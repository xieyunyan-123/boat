#!/usr/bin/env python3
"""RDK X5 USB LiDAR Web Viewer"""
import os, sys, struct, time, json, argparse, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import serial, serial.tools.list_ports

latest_scan = []
scan_lock = threading.Lock()

def find_lidar_port():
    for p in serial.tools.list_ports.comports():
        if p.vid and p.vid in (0x1A86, 0x10C4, 0x067B, 0x2E3C): return p.device
    for p in serial.tools.list_ports.comports():
        if 'USB' in p.description.upper() or 'usb' in p.device.lower(): return p.device
    return '/dev/ttyUSB0'

def parse_points(data):
    points = []
    i = 0
    while i < len(data) - 5:
        if data[i] == 0x55 and data[i+1] == 0xAA:
            hdr = 6  # skip 55 aa 07 0c ae 59
            remaining = len(data) - i - hdr
            n_points = remaining // 4
            for j in range(n_points):
                offset = i + hdr + j * 4
                if offset + 3 >= len(data): break
                a_raw = struct.unpack_from('<H', data, offset)[0]
                d_raw = struct.unpack_from('<H', data, offset + 2)[0]
                a = a_raw * 0.01
                if a >= 360: a -= 360.0
                if a < 0: a += 360.0
                if d_raw > 0 and d_raw < 12000 and 0 <= a < 360:
                    points.append({"a": round(a, 2), "d": round(d_raw / 1000.0, 3)})
            i += hdr
        else:
            i += 1
    return points

def lidar_reader(port, baud):
    global latest_scan
    ser = serial.Serial(port, baud, timeout=0.1)
    buf = bytearray()
    print(f"[INFO] LiDAR reader started on {port} @ {baud}")
    while True:
        try:
            if ser.in_waiting:
                buf.extend(ser.read(ser.in_waiting))
        except Exception:
            time.sleep(0.1); continue
        if len(buf) > 4096: buf = buf[-2048:]
        pts = parse_points(buf)
        if pts:
            with scan_lock:
                latest_scan = pts

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RDK X5 LiDAR Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;display:flex;justify-content:center;align-items:center;height:100vh;font-family:sans-serif;overflow:hidden}
canvas{display:block;background:#16213e;border-radius:12px;box-shadow:0 0 40px rgba(0,150,255,0.15)}
.info{position:absolute;top:16px;left:50%;transform:translateX(-50%);color:#8899aa;font-size:14px;text-align:center;pointer-events:none}
.info span{color:#00d4ff;font-weight:bold}
</style>
</head>
<body>
<div class="container">
  <div class="info">Points: <span id="ptCount">0</span> | Frame: <span id="fps">0</span> FPS</div>
  <canvas id="c"></canvas>
</div>
<script>
const SIZE = Math.min(window.innerWidth, window.innerHeight) - 40;
const c = document.getElementById('c'), ctx = c.getContext('2d');
c.width = c.height = SIZE;
const cx=SIZE/2, cy=SIZE/2, maxR=SIZE/2-20;
let fc=0, lft=performance.now();
function draw(){
  fetch('/data').then(r=>r.json()).then(d=>{
    ctx.clearRect(0,0,SIZE,SIZE); ctx.save(); ctx.translate(cx,cy);
    for(let r=1;r<=5;r++){ ctx.beginPath(); ctx.arc(0,0,(r/6)*maxR,0,Math.PI*2); ctx.strokeStyle='rgba(255,255,255,0.05)'; ctx.stroke() }
    for(let a=0;a<360;a+=30){ const rad=a*Math.PI/180; ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(Math.cos(rad)*maxR,Math.sin(rad)*maxR); ctx.strokeStyle='rgba(255,255,255,0.04)'; ctx.stroke() }
    const pts=d.points||[]; document.getElementById('ptCount').textContent=pts.length;
    for(const p of pts){
      const rad=p.a*Math.PI/180, dist=(p.d/12)*maxR, x=Math.cos(rad)*dist, y=Math.sin(rad)*dist;
      ctx.fillStyle='#00d4ff'; ctx.beginPath(); ctx.arc(x,y,2,0,Math.PI*2); ctx.fill()
    }
    ctx.restore(); fc++; const n=performance.now();
    if(n-lft>=1000){ document.getElementById('fps').textContent=fc; fc=0; lft=n }
  }).catch(()=>{}); requestAnimationFrame(draw)
}
draw();
</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/data':
            with scan_lock: data = json.dumps({"points": latest_scan})
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            self.wfile.write(data.encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())
    def log_message(self, fmt, *args): pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port','-p',default=None)
    parser.add_argument('--baud','-b',type=int,default=230400)
    parser.add_argument('--http-port',type=int,default=8080)
    parser.add_argument('--host',default='0.0.0.0')
    args = parser.parse_args()
    port = args.port or find_lidar_port()
    print(f"[INFO] LiDAR on {port} @ {args.baud}")
    print(f"[INFO] Open http://<RDK_IP>:{args.http_port} in browser")
    t = threading.Thread(target=lidar_reader, args=(port,args.baud), daemon=True)
    t.start()
    server = HTTPServer((args.host, args.http_port), Handler)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n[INFO] Shutdown"); server.shutdown()

if __name__ == '__main__': main()
