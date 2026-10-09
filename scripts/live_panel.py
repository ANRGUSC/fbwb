"""Live/replay desktop companion. Reads recorded facts; never synthesizes activity."""
import argparse
from collections import deque
import ctypes
import ctypes.wintypes
import json
import math
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

BG, CARD, TEXT, MUTED = '#0b1422','#132237','#e7eef8','#9eafc6'
CYAN, GREEN, ORANGE = '#62c8ef','#70d6a4','#f2ad6c'


class Panel:
    def __init__(self, directory, run_id=None, world=None, replay=False):
        self.directory = Path(directory)
        self.run_id = run_id
        self.world = world
        self.root = tk.Tk()
        self.root.title('Flybrain • live inputs, neural activity, outputs')
        self.root.configure(bg=BG)
        self.root.tk.call('tk','scaling',1.15)
        self.root.minsize(650,660)
        area = ctypes.wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(48,0,ctypes.byref(area),0)
        self.area = area
        self.width = max(680,round((area.right-area.left)*0.43))
        self.root.geometry(f'{self.width}x{area.bottom-area.top-10}+{area.right-self.width}+{area.top}')
        toolbar = tk.Frame(self.root,bg=BG)
        toolbar.pack(fill='x',padx=10,pady=6)
        for label,command in [('Live',self.live),('Replay',self.replay),('Pause view',self.pause),('Stop run',self.stop),('Open logs',lambda:os.startfile(self.directory))]:
            tk.Button(toolbar,text=label,command=command,bg=CARD,fg=TEXT,activebackground='#23415d',activeforeground='white',relief='flat',padx=9).pack(side='left',padx=3)
        self.slider = ttk.Scale(toolbar,from_=0,to=1,command=self.scrub)
        self.slider.pack(side='right',fill='x',expand=True,padx=6)
        navigation = tk.Frame(self.root,bg=BG)
        navigation.pack(fill='x',padx=14,pady=(0,4))
        tk.Label(navigation,text='Jump to recorded phase:',bg=BG,fg=MUTED).pack(side='left')
        self.phase_picker = ttk.Combobox(navigation,state='readonly',width=28)
        self.phase_picker.pack(side='left',padx=8)
        self.phase_picker.bind('<<ComboboxSelected>>',self.jump_phase)
        tk.Button(navigation,text='Load phases',command=self.load_phases,bg=CARD,fg=TEXT,relief='flat').pack(side='left')
        body = tk.Frame(self.root,bg=BG)
        body.pack(fill='both',expand=True)
        self.canvas = tk.Canvas(body,bg=BG,highlightthickness=0)
        scrollbar = ttk.Scrollbar(body,orient='vertical',command=self.canvas.yview)
        scrollbar.pack(side='right',fill='y')
        self.canvas.pack(fill='both',expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set,scrollregion=(0,0,800,970))
        self.canvas.bind('<MouseWheel>',lambda e:self.canvas.yview_scroll(-int(e.delta/120),'units'))
        self.canvas.bind('<Configure>',lambda e:self.draw())
        self.packet = None
        self.image = None
        self.last_seq = None
        self.history = deque(maxlen=160)
        self.playback = False
        self.paused = False
        self.records = []
        self.cursor = 0
        self.updating_slider = False
        self.status = 'Waiting for simulator / loading full connectome…'
        self.root.after(400,self.dock)
        if replay:
            self.root.after(200,self.replay)
        self.root.after(150,self.poll)

    def text(self,x,y,text,size=13,color=TEXT,**kw):
        return self.canvas.create_text(x,y,text=text,fill=color,font=('Segoe UI',-size),anchor='nw',**kw)

    def card(self,y,h,title):
        w = self.canvas.winfo_width()-12
        self.canvas.create_rectangle(10,y,w,y+h,fill=CARD,outline='')
        self.text(22,y+9,title,12,CYAN)

    def bar(self,x,y,width,value,limit=1,color=CYAN):
        self.canvas.create_rectangle(x,y,x+width,y+8,fill='#25364d',outline='')
        self.canvas.create_rectangle(x,y,x+width*max(0,min(1,value/limit)),y+8,fill=color,outline='')

    def draw(self):
        c = self.canvas
        c.delete('all')
        w = max(630,c.winfo_width())
        self.text(18,6,'Flybrain — watch the signal path',21)
        self.text(18,34,self.status,12,MUTED,width=w-36)
        if self.packet is None:
            self.text(20,100,'The panel will update when the controller publishes its first frame.\nThe full connectome can take several seconds to load.',15,width=w-40)
            return
        p = self.packet
        state, applied, motor = p['state'],p['applied'],p['decoded']
        brain = p.get('brain')
        anatomical = bool(brain and brain.get('decode',{}).get('decoder','').startswith('anatomical'))
        self.card(65,192,'1   VISION  →  FOUR OPTICAL-FLOW FEATURES')
        try:
            self.image = tk.PhotoImage(data=p['camera_png'])
            c.create_image(24,96,image=self.image,anchor='nw')
        except tk.TclError:
            self.text(24,110,'Camera frame unavailable')
        x = 184
        self.text(24,242,p['phase'].get('camera','onboard camera'),10,MUTED)
        for i,(name,value) in enumerate(zip(['Radial / outward','Left image motion','Right image motion','Vertical ratio'],p['inputs'])):
            y = 96+i*33
            self.text(x,y,name,12)
            self.text(w-68,y,f'{value:.3f}',12,CYAN)
            self.bar(x,y+18,w-x-40,value)
        self.card(267,142,'2   RECEPTOR ACTIVITY — SAMPLED NEURONS, NOT ANATOMICAL OMMATIDIA')
        if brain:
            receptors = brain['receptors']
            offset = 0
            group_width = (w-50)/3
            colors = [CYAN,ORANGE,GREEN]
            for group_idx,group in enumerate(brain['groups']):
                gx = 22+group_idx*group_width
                labels = ['R1-6 • anatomical left','R1-6 • anatomical right','R7/R8 • vertical'] if anatomical else ['R1-6 A • left proxy','R1-6 B • right proxy','R7/R8 • vertical']
                self.text(gx,294,labels[group_idx],12,colors[group_idx])
                self.text(gx,313,f"Drive {group['target_hz']:.0f} Hz · measured {group['observed_hz']:.0f} Hz",11,MUTED)
                n = brain['sample_group_sizes'][group_idx]
                cell = min(9,(group_width-8)/24)
                for j,rec in enumerate(receptors[offset:offset+n]):
                    xx = gx+(j%24)*cell
                    yy = 336+(j//24)*12
                    q = min(1,rec['hz']/300)
                    rgb = tuple(int(int(colors[group_idx][k:k+2],16)*q+30*(1-q)) for k in [1,3,5])
                    item = c.create_rectangle(xx,yy,xx+cell-1,yy+9,fill='#%02x%02x%02x'%rgb,outline='')
                    detail = f"Cell {rec['id']} | {rec['hz']:.0f} Hz measured | {rec['target_hz']:.0f} Hz drive | {rec['drive_events']} input events"
                    c.tag_bind(item,'<Enter>',lambda e,s=detail:self.show_cell(s))
                offset += n
            self.cell_label = self.text(22,369,'Hover a square for cell ID and rate. Brightness: 0–300 Hz; 20 ms measurement window.',11,MUTED,width=w-45)
        else:
            self.text(22,308,'Brain inactive during takeoff, or detailed telemetry unavailable in this mode.',13,MUTED,width=w-45)
        self.card(419,139,'3   NEURAL PROCESSING')
        if brain:
            self.text(22,447,f"{brain['spike_count']:,} spikes / {brain['window_ms']:.0f} ms",18,GREEN)
            self.text(22,472,f"{brain['active_neurons']:,} active neurons",13)
            rates = brain['population_hz']
            self.text(22,496,f"Mean Hz/cell: receptors {rates['receptors']:.1f}  ·  motion {rates['motion']:.1f}  ·  DN {rates['descending']:.1f}",12,MUTED,width=w-40)
            self.text(22,530,f"Minimum voltage {brain['voltage_min']:.3f} V — physiology remains experimental",11,ORANGE)
            gx,gy,gw,gh = w*0.58,448,w*0.23,43
            hist = list(self.history)
            if len(hist)>1 and hist[-1][0]>hist[0][0]:
                peak = max(1,max(v for _,v in hist))
                coords = []
                for t,v in hist:
                    coords.extend([gx+gw*(t-hist[0][0])/(hist[-1][0]-hist[0][0]),gy+gh*(1-v/peak)])
                c.create_line(*coords,fill=GREEN,width=2)
            self.text(gx,494,'Recent spikes / neural ms',10,MUTED)
            hx,hy = w-65,471
            c.create_oval(hx-25,hy-25,hx+25,hy+25,outline=MUTED)
            c.create_line(hx,hy,hx-23*math.sin(state['yaw']),hy-23*math.cos(state['yaw']),fill=ORANGE,width=3,arrow=tk.LAST)
            self.text(hx-34,503,'Drone heading',10,MUTED)
        else:
            self.text(22,458,'No neural computation yet. Takeoff uses the ordinary flight controller.',13,MUTED,width=w-45)
        self.card(568,211,'4   OUTPUTS — NEURAL DECODE  →  APPLIED COMMAND')
        self.text(22,594,'Axis',11,MUTED)
        self.text(180,594,'Neural RC [−100, +100]',11,MUTED)
        self.text(390,594,'Actually applied',11,MUTED)
        altitude_owner = 'fixture + PID' if p['phase']['held'] or p['phase'].get('turn_trial') else 'PID'
        descriptions = [('Forward / reverse','forward',f"{applied['vx']:+.3f} m/s",'fixture' if p['phase']['held'] or p['phase'].get('turn_trial') else 'brain + filter'),('Turn left / right','yaw',f"{applied['yaw']:+.3f} rad/s",'fixture' if p['phase']['held'] else '+ = left / CCW'),('Up / down','vertical',f"target {applied['height']:.2f} m",f"actual {state['z']:.2f} m; {altitude_owner}"),('Sideways','roll',f"{applied['vy']:+.3f} m/s",'disabled')]
        for i,(name,key,value,owner) in enumerate(descriptions):
            y = 617+i*28
            self.text(22,y,name,12)
            self.text(180,y,f'{motor[key]:+.1f}',13,ORANGE)
            midpoint = 300
            c.create_line(250,y+9,350,y+9,fill='#35475e',width=7)
            c.create_line(midpoint,y+9,midpoint+max(-100,min(100,motor[key]))/2,y+9,fill=ORANGE,width=7)
            c.create_line(midpoint,y+3,midpoint,y+15,fill=TEXT)
            self.text(390,y,value,12,GREEN)
            self.text(520,y,owner,11,MUTED,width=w-545)
        speeds = '   '.join(f'M{i+1}: {v:+.1f}' for i,v in enumerate(applied['motors']))
        self.text(22,733,'Motor speeds (rad/s)   '+speeds,12,MUTED)
        self.text(22,752,f"Measured heading {state['yaw']*180/3.141592653589793:+.1f} degrees (+ = left / CCW)",12,GREEN)
        self.card(783,150,'5   WHAT IS HAPPENING NOW')
        self.text(22,811,p['narrative'],12,TEXT,width=w-48)
        self.text(20,940,'Measured signals explain the decoder; they do not prove a biological neuron-by-neuron causal story.',11,MUTED,width=w-38)
        c.configure(scrollregion=(0,0,w,980))

    def show_cell(self,text):
        if hasattr(self,'cell_label'):
            self.canvas.itemconfigure(self.cell_label,text=text)

    def accept(self,packet):
        if self.run_id and packet.get('run_id') != self.run_id:
            return
        self.packet = packet
        b = packet.get('brain')
        t = packet['state']['t']
        if self.history and t < self.history[-1][0]:
            self.history.clear()
        if b and (not self.history or self.history[-1][0] != t):
            self.history.append((t,b['spike_count']/b['window_ms']))
        label = 'REPLAY' if self.playback else 'LIVE'
        held = 'POSE HELD • outputs not applied' if packet['phase']['held'] else 'YAW RIG • neural turning' if packet['phase'].get('turn_trial') else 'flight enabled'
        self.status = f"{label}   {t:.2f} s simulated   |   {packet['phase']['name']}   |   {held}"
        self.draw()

    def poll(self):
        if not self.paused:
            if self.playback and self.records:
                self.accept(self.records[self.cursor])
                self.updating_slider = True
                self.slider.set(self.cursor)
                self.updating_slider = False
                self.cursor = min(self.cursor+1,len(self.records)-1)
            elif not self.playback:
                try:
                    packet = json.loads((self.directory/'latest.json').read_text(encoding='utf-8'))
                    if (not self.run_id or packet.get('run_id') == self.run_id) and packet.get('seq') != self.last_seq:
                        self.accept(packet)
                        self.last_seq = packet.get('seq')
                except (OSError,ValueError):
                    pass
                try:
                    status = json.loads((self.directory/'status.json').read_text())
                    if status.get('run_id') == self.run_id and status['state'] in ('finished','failed'):
                        self.status = status['state'].upper()+' — use Replay and the slider to inspect the recording.'
                        self.draw()
                except (OSError,ValueError,KeyError):
                    pass
        self.root.after(160,self.poll)

    def live(self):
        self.playback = self.paused = False
        self.last_seq = None
        self.history.clear()

    def replay(self):
        try:
            records = []
            with (self.directory/'live.ndjson').open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        break  # live recording may end with an incomplete append
                    if not self.run_id or record.get('run_id') == self.run_id:
                        records.append(record)
            if not records:
                return
            self.records = records
            self.phase_picker.configure(values=list(dict.fromkeys(p['phase']['name'] for p in records)))
            self.slider.configure(to=len(records)-1)
            self.cursor = 0
            self.history.clear()
            self.playback,self.paused = True,False
        except OSError:
            pass

    def pause(self):
        self.paused = not self.paused

    def load_phases(self):
        self.replay()
        self.paused = True
        self.phase_picker.configure(values=list(dict.fromkeys(p['phase']['name'] for p in self.records)))

    def jump_phase(self,event=None):
        name = self.phase_picker.get()
        indices = [i for i,p in enumerate(self.records) if p['phase']['name'] == name]
        if indices:
            self.cursor = indices[0]
            self.playback,self.paused = True,False
            self.history.clear()
            self.accept(self.records[self.cursor])

    def scrub(self,value):
        if self.updating_slider or not self.records:
            return
        self.playback,self.paused = True,True
        self.cursor = min(len(self.records)-1,int(float(value)))
        self.accept(self.records[self.cursor])

    def stop(self):
        if self.run_id:
            (self.directory/'control.json').write_text(json.dumps({'run_id':self.run_id,'action':'stop'}))

    def dock(self):
        if self.world:
            expected = str(Path(self.world)).lower().replace('/','\\')
            user32 = ctypes.windll.user32
            callback = ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.wintypes.HWND,ctypes.wintypes.LPARAM)
            found = []
            def visit(hwnd,lparam):
                length = user32.GetWindowTextLengthW(hwnd)
                buffer = ctypes.create_unicode_buffer(length+1)
                user32.GetWindowTextW(hwnd,buffer,length+1)
                if expected in buffer.value.lower() and 'webots' in buffer.value.lower():
                    found.append(hwnd)
                return True
            user32.EnumWindows(callback(visit),0)
            for hwnd in found:
                user32.ShowWindow(hwnd,9)
                user32.SetWindowPos(hwnd,0,self.area.left,self.area.top,self.area.right-self.area.left-self.width,self.area.bottom-self.area.top,0x0014)
            if not found:
                self.root.after(1000,self.dock)


def main():
    import ctypes.wintypes
    ctypes.windll.user32.SetProcessDPIAware()
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--world')
    parser.add_argument('--replay',action='store_true')
    args = parser.parse_args()
    panel = Panel(args.directory,args.run_id,args.world,args.replay)
    panel.root.mainloop()


if __name__ == '__main__':
    main()
