#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define REG_MAX     16
#define NUM_SCALES  3
#define MAX_DETS    300

static const int STRIDES[3] = {8, 16, 32};
static const int HS[3] = {80, 40, 20};
static const int WS[3] = {80, 40, 20};

typedef struct {
    float *boxes;
    float *scores;
    int    count;
    int    capacity;
} BoxList;

static BoxList* boxlist_create(int cap) {
    BoxList *bl = (BoxList*)malloc(sizeof(BoxList));
    bl->boxes  = (float*)malloc(cap * 4 * sizeof(float));
    bl->scores = (float*)malloc(cap * sizeof(float));
    bl->count    = 0;
    bl->capacity = cap;
    return bl;
}

static void boxlist_add(BoxList *bl, float x1, float y1, float x2, float y2, float s) {
    if (bl->count >= bl->capacity) {
        bl->capacity *= 2;
        bl->boxes  = (float*)realloc(bl->boxes,  bl->capacity * 4 * sizeof(float));
        bl->scores = (float*)realloc(bl->scores, bl->capacity * sizeof(float));
    }
    int i = bl->count;
    bl->boxes[i*4+0] = x1;
    bl->boxes[i*4+1] = y1;
    bl->boxes[i*4+2] = x2;
    bl->boxes[i*4+3] = y2;
    bl->scores[i] = s;
    bl->count++;
}

static void boxlist_free(BoxList *bl) {
    free(bl->boxes);
    free(bl->scores);
    free(bl);
}

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* DFL decode: NCHW tensor (1, 64, h, w) -> (4, h, w) */
static void dfl_decode(const float *bbox_reg, int h, int w, float *dist_out) {
    int spatial = h * w;
    for (int g = 0; g < 4; g++) {
        for (int i = 0; i < spatial; i++) {
            float vals[REG_MAX];
            float mx = -1e30f;
            for (int k = 0; k < REG_MAX; k++) {
                float v = bbox_reg[(g * REG_MAX + k) * spatial + i];
                vals[k] = v;
                if (v > mx) mx = v;
            }
            float sum = 0.0f;
            for (int k = 0; k < REG_MAX; k++) {
                vals[k] = expf(vals[k] - mx);
                sum += vals[k];
            }
            float wsum = 0.0f;
            for (int k = 0; k < REG_MAX; k++) {
                wsum += (vals[k] / sum) * (float)k;
            }
            dist_out[g * spatial + i] = wsum;
        }
    }
}

static int nms_greedy(float *boxes, float *scores, int n, float thresh, int *keep) {
    if (n <= 0) return 0;

    int *idx = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (scores[idx[j]] < scores[idx[j+1]]) {
                int t = idx[j];
                idx[j] = idx[j+1];
                idx[j+1] = t;
            }
        }
    }

    float *areas = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float bw = boxes[idx[i]*4+2] - boxes[idx[i]*4+0] + 1.0f;
        float bh = boxes[idx[i]*4+3] - boxes[idx[i]*4+1] + 1.0f;
        areas[i] = (bw > 0.0f ? bw : 0.0f) * (bh > 0.0f ? bh : 0.0f);
    }

    int k = 0;
    for (int i = 0; i < n; i++) {
        int ii = idx[i];
        if (scores[ii] < 0.0f) continue;
        keep[k++] = ii;

        float bx1 = boxes[ii*4+0], by1 = boxes[ii*4+1];
        float bx2 = boxes[ii*4+2], by2 = boxes[ii*4+3];
        float ba = areas[i];

        for (int j = i + 1; j < n; j++) {
            int jj = idx[j];
            if (scores[jj] < 0.0f) continue;

            float xx1 = fmaxf(bx1, boxes[jj*4+0]);
            float yy1 = fmaxf(by1, boxes[jj*4+1]);
            float xx2 = fminf(bx2, boxes[jj*4+2]);
            float yy2 = fminf(by2, boxes[jj*4+3]);
            float iw = xx2 - xx1 + 1.0f;
            float ih = yy2 - yy1 + 1.0f;
            if (iw <= 0.0f || ih <= 0.0f) continue;

            float inter = iw * ih;
            float iou = inter / (ba + areas[j] - inter);
            if (iou > thresh) scores[jj] = -1.0f;
        }
    }

    free(idx);
    free(areas);
    return k;
}


/* =================================================================
 *  主接口：逐个接收 6 个 float* 输出指针
 *
 *  outputs[0..5] : NCHW float32 buffers
 *    0: (64, 80, 80)  bbox stride 8
 *    1: (1,  80, 80)  cls  stride 8
 *    2: (64, 40, 40)  bbox stride 16
 *    3: (1,  40, 40)  cls  stride 16
 *    4: (64, 20, 20)  bbox stride 32
 *    5: (1,  20, 20)  cls  stride 32
 *
 *   input_size : 模型输入边长 (640)
 *   ori_w, ori_h : 原始图像宽高
 *   conf_thres, nms_thres : 阈值
 *
 *   返回 JSON 字符串，需用 fire_postprocess_free 释放
 * ================================================================= */
char* fire_postprocess(
    const float *output0, const float *output1,
    const float *output2, const float *output3,
    const float *output4, const float *output5,
    int input_size,
    int ori_w, int ori_h,
    float conf_thres, float nms_thres)
{
    const float *bbox_outs[3] = {output0, output2, output4};
    const float *cls_outs[3]  = {output1, output3, output5};

    BoxList *bl = boxlist_create(64);

    float r = (float)input_size / fmaxf((float)ori_w, (float)ori_h);
    float pad_x = ((float)input_size - (float)ori_w * r) * 0.5f;
    float pad_y = ((float)input_size - (float)ori_h * r) * 0.5f;

    for (int s = 0; s < NUM_SCALES; s++) {
        int h = HS[s], w = WS[s];
        int stride = STRIDES[s];
        int spatial = h * w;

        const float *bbox_reg = bbox_outs[s];
        const float *cls_data = cls_outs[s];

        float *dist = (float*)malloc(4 * spatial * sizeof(float));
        if (!dist) continue;
        dfl_decode(bbox_reg, h, w, dist);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int pi = y * w + x;

                float score = sigmoid_f(cls_data[pi]);
                if (score <= conf_thres) continue;

                float cx = (float)x * (float)stride;
                float cy = (float)y * (float)stride;

                float left   = dist[0 * spatial + pi] * (float)stride;
                float top    = dist[1 * spatial + pi] * (float)stride;
                float right  = dist[2 * spatial + pi] * (float)stride;
                float bottom = dist[3 * spatial + pi] * (float)stride;

                float x1 = cx - left;
                float y1 = cy - top;
                float x2 = cx + right;
                float y2 = cy + bottom;

                x1 = (x1 - pad_x) / r;
                y1 = (y1 - pad_y) / r;
                x2 = (x2 - pad_x) / r;
                y2 = (y2 - pad_y) / r;

                if (x1 < 0.0f) x1 = 0.0f;
                if (y1 < 0.0f) y1 = 0.0f;
                if (x2 > (float)ori_w) x2 = (float)ori_w;
                if (y2 > (float)ori_h) y2 = (float)ori_h;

                if (x2 - x1 < 2.0f || y2 - y1 < 2.0f) continue;

                boxlist_add(bl, x1, y1, x2, y2, score);
            }
        }
        free(dist);
    }

    int n = bl->count;
    int *keep = NULL;
    int nkeep = 0;
    if (n > 0) {
        keep = (int*)malloc(n * sizeof(int));
        nkeep = nms_greedy(bl->boxes, bl->scores, n, nms_thres, keep);
        if (nkeep > MAX_DETS) nkeep = MAX_DETS;
    }

    // Build JSON using snprintf into a growing buffer
    size_t cap = 256;
    char *json = (char*)malloc(cap);
    if (!json) {
        if (keep) free(keep);
        boxlist_free(bl);
        return strdup("[]");
    }

    int off = 0;
    off += snprintf(json + off, cap - off, "[");
    for (int i = 0; i < nkeep; i++) {
        int ki = keep[i];
        int need = snprintf(NULL, 0,
            "%s{\"bbox\":[%.1f,%.1f,%.1f,%.1f],\"score\":%.4f,\"name\":\"fire\"}",
            i > 0 ? "," : "",
            bl->boxes[ki*4+0], bl->boxes[ki*4+1],
            bl->boxes[ki*4+2], bl->boxes[ki*4+3],
            bl->scores[ki]);
        if (off + need + 1 > (int)cap) {
            cap = off + need + 256;
            json = (char*)realloc(json, cap);
        }
        off += snprintf(json + off, cap - off,
            "%s{\"bbox\":[%.1f,%.1f,%.1f,%.1f],\"score\":%.4f,\"name\":\"fire\"}",
            i > 0 ? "," : "",
            bl->boxes[ki*4+0], bl->boxes[ki*4+1],
            bl->boxes[ki*4+2], bl->boxes[ki*4+3],
            bl->scores[ki]);
    }
    off += snprintf(json + off, cap - off, "]");

    if (keep) free(keep);
    boxlist_free(bl);

    return json;
}


void fire_postprocess_free(char *s) {
    free(s);
}
