import math, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont
W,H,FPS=1080,1920,30
OUT='segundo_turno_2026.mp4'
B='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'; R='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
F=lambda s,b=True: ImageFont.truetype(B if b else R,s)
BG=(14,18,32); FG=(240,242,248); MUT=(150,158,180)
RED=(214,48,64); BLUE=(40,110,220); YEL=(245,190,40); GRY=(110,116,135)
def ease(t): t=max(0,min(1,t)); return 1-(1-t)**3
def ctext(d,y,s,f,c=FG,a=1.0):
    c=tuple(int(BG[i]+(c[i]-BG[i])*a) for i in range(3))
    w=d.textlength(s,font=f); d.text(((W-w)/2,y),s,font=f,fill=c)
def header(d,t,title,a):
    ctext(d,150,title,F(56),YEL,a)
    d.rectangle([W/2-80,235,W/2+80,241],fill=tuple(int(BG[i]+(YEL[i]-BG[i])*a) for i in range(3)))
def bar(d,y,label,val,maxv,col,p,a):
    ctext(d,y,label,F(46),FG,a)
    x0,x1=110,W-110; d.rounded_rectangle([x0,y+70,x1,y+150],20,fill=(32,38,58))
    v=val*ease(p); xe=x0+(x1-x0)*v/maxv
    if xe>x0+40: d.rounded_rectangle([x0,y+70,xe,y+150],20,fill=col)
    d.text((x0+25 if xe<x0+220 else xe-200,y+82),f"{v:.0f}%",font=F(54),fill=FG)
SC=[]
def scene(dur):
    def deco(fn): SC.append((dur,fn)); return fn
    return deco
@scene(6)
def s1(d,t):
    a=ease(t/0.8)
    ctext(d,560,"ELEIÇÕES 2026",F(60),YEL,a)
    ctext(d,700,"SEGUNDO TURNO",F(96),FG,ease((t-0.4)/0.8))
    ctext(d,830,"Quem vai ganhar?",F(80),FG,ease((t-0.9)/0.8))
    a2=ease((t-1.8)/0.8)
    d.rounded_rectangle([140,1050,940,1190],30,fill=tuple(int(BG[i]+(RED[i]-BG[i])*a2) for i in range(3)))
    ctext(d,1088,"Domingo, 25 de outubro",F(50),FG,a2)
    ctext(d,1260,"faltam 2 dias",F(56,False),MUT,ease((t-2.5)/0.8))
@scene(8)
def s2(d,t):
    header(d,t,"A DISPUTA",ease(t/0.6))
    for i,(n,p,c) in enumerate([("LULA","PT",RED),("FLÁVIO BOLSONARO","PL",BLUE)]):
        a=ease((t-0.5-i*0.8)/0.8); y=520+i*560
        d.rounded_rectangle([110,y,W-110,y+300],36,fill=tuple(int(BG[k]+(c[k]-BG[k])*a) for k in range(3)))
        ctext(d,y+80,n,F(72 if i==0 else 62),FG,a); ctext(d,y+185,p,F(48,False),FG,a)
    ctext(d,885,"X",F(110),YEL,ease((t-1.6)/0.6))
@scene(12)
def s3(d,t):
    header(d,t,"DATAFOLHA · 20 a 22/out",ease(t/0.6))
    ctext(d,330,"Votos válidos",F(48,False),MUT,ease((t-0.3)/0.6))
    bar(d,420,"Lula",50,100,RED,(t-0.6)/1.5,ease((t-0.4)/0.5))
    bar(d,640,"Flávio Bolsonaro",50,100,BLUE,(t-0.9)/1.5,ease((t-0.7)/0.5))
    a=ease((t-4.5)/0.8)
    ctext(d,940,"Votos totais",F(48,False),MUT,a)
    rows=[("Flávio",47,BLUE),("Lula",46,RED),("Brancos/nulos",5,GRY),("Indecisos",2,GRY)]
    for i,(n,v,c) in enumerate(rows):
        aa=ease((t-5-i*0.4)/0.6); y=1030+i*120
        cc=tuple(int(BG[k]+(c[k]-BG[k])*aa) for k in range(3))
        d.rounded_rectangle([160,y,180,y+80],6,fill=cc)
        d.text((210,y+10),n,font=F(50),fill=tuple(int(BG[k]+(FG[k]-BG[k])*aa) for k in range(3)))
        s=f"{v}%"; d.text((W-160-d.textlength(s,font=F(56)),y+6),s,font=F(56),fill=tuple(int(BG[k]+(FG[k]-BG[k])*aa) for k in range(3)))
    ctext(d,1560,"Margem de erro: 2 pontos",F(44,False),MUT,ease((t-7.5)/0.8))
    ctext(d,1640,"EMPATE TÉCNICO",F(76),YEL,ease((t-8.3)/0.8))
@scene(9)
def s4(d,t):
    header(d,t,"QUAEST · 17 a 20/out",ease(t/0.6))
    ctext(d,330,"Votos totais (estimulada)",F(48,False),MUT,ease((t-0.3)/0.6))
    bar(d,450,"Flávio Bolsonaro",45,60,BLUE,(t-0.6)/1.5,ease((t-0.4)/0.5))
    bar(d,700,"Lula",44,60,RED,(t-0.9)/1.5,ease((t-0.7)/0.5))
    ctext(d,1050,"Diferença de 1 ponto",F(58),FG,ease((t-3)/0.8))
    ctext(d,1140,"dentro da margem de erro",F(48,False),MUT,ease((t-3.4)/0.8))
    ctext(d,1350,"EMPATE TÉCNICO",F(76),YEL,ease((t-4.5)/0.8))
@scene(10)
def s5(d,t):
    header(d,t,"A TENDÊNCIA",ease(t/0.6))
    ctext(d,330,"Vantagem de Flávio no Datafolha",F(46,False),MUT,ease((t-0.3)/0.6))
    pts=[("1ª rodada","4 pts",4),("15/out","2 pts*",2),("22/out","0 pt*",0)]
    x0,x1,yb,yt=200,880,1150,550
    a=ease((t-0.8)/0.6)
    d.line([x0-40,yb,x1+40,yb],fill=tuple(int(BG[k]+(GRY[k]-BG[k])*a) for k in range(3)),width=4)
    prog=ease((t-1.2)/3.0)*2
    xy=[(x0+(x1-x0)*i/2, yb-(yb-yt)*v/4) for i,(_,_,v) in enumerate(pts)]
    for i in range(2):
        f=max(0,min(1,prog-i))
        if f>0:
            (ax,ay),(bx,by)=xy[i],xy[i+1]; d.line([ax,ay,ax+(bx-ax)*f,ay+(by-ay)*f],fill=BLUE,width=10)
    for i,((x,y),(lab,val,_)) in enumerate(zip(xy,pts)):
        aa=ease((t-1.2-i*1.5)/0.5)
        if aa<=0: continue
        d.ellipse([x-22,y-22,x+22,y+22],fill=BLUE)
        w=d.textlength(val,font=F(48)); d.text((x-w/2,y-95),val,font=F(48),fill=FG)
        w=d.textlength(lab,font=F(38,False)); d.text((x-w/2,yb+30),lab,font=F(38,False),fill=MUT)
    ctext(d,1330,"*em votos válidos",F(34,False),MUT,ease((t-5)/0.6))
    ctext(d,1470,"Lula encostou na reta final",F(58),FG,ease((t-5.5)/0.8))
    ctext(d,1560,"a disputa ficou aberta",F(50,False),MUT,ease((t-6)/0.8))
@scene(8)
def s6(d,t):
    header(d,t,"O QUE PODE DECIDIR",ease(t/0.6))
    ctext(d,360,"Governo Lula (Datafolha)",F(46,False),MUT,ease((t-0.3)/0.6))
    ctext(d,440,"49% aprovam  ·  48% desaprovam",F(52),FG,ease((t-0.6)/0.6))
    items=["Abstenção: quem vai votar","Os 2% de indecisos","Brancos e nulos","Últimos debates e campanha"]
    for i,s in enumerate(items):
        a=ease((t-1.8-i*0.6)/0.6); y=700+i*190
        d.rounded_rectangle([90,y,W-90,y+140],28,fill=tuple(int(BG[k]+((36,44,68)[k]-BG[k])*a) for k in range(3)))
        d.ellipse([130,y+50,170,y+90],fill=tuple(int(BG[k]+(YEL[k]-BG[k])*a) for k in range(3)))
        d.text((245,y+42),s,font=F(44),fill=tuple(int(BG[k]+(FG[k]-BG[k])*a) for k in range(3)))
@scene(7)
def s7(d,t):
    ctext(d,480,"QUEM VAI GANHAR?",F(84),YEL,ease(t/0.7))
    ctext(d,650,"As pesquisas não cravam:",F(56),FG,ease((t-0.8)/0.7))
    ctext(d,740,"é empate técnico.",F(56),FG,ease((t-1.2)/0.7))
    ctext(d,920,"Últimas pesquisas: sábado (24)",F(48,False),MUT,ease((t-2.2)/0.7))
    a=ease((t-3)/0.7)
    d.rounded_rectangle([140,1080,940,1230],34,fill=tuple(int(BG[i]+(RED[i]-BG[i])*a) for i in range(3)))
    ctext(d,1122,"Quem decide é você, dia 25",F(50),FG,a)
    ctext(d,1700,"Fontes: Datafolha, Quaest (out/2026)",F(32,False),MUT,ease((t-3.5)/0.7))
tot=sum(s for s,_ in SC)
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
  '-f','lavfi','-i',f'aevalsrc=0.05*sin(2*PI*110*t)*(0.6+0.4*sin(2*PI*0.25*t))+0.035*sin(2*PI*164.8*t)+0.025*sin(2*PI*220*t)*(0.5+0.5*sin(2*PI*0.5*t)):s=44100:d={tot}',
  '-af',f'afade=t=in:d=2,afade=t=out:st={tot-2}:d=2','-c:v','libx264','-pix_fmt','yuv420p','-crf','20','-c:a','aac','-shortest',OUT],stdin=subprocess.PIPE)
for dur,fn in SC:
    for i in range(int(dur*FPS)):
        t=i/FPS; im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
        fn(d,t)
        # progress bar + fade
        g=sum(s for s,_ in SC[:SC.index((dur,fn))])+t
        d.rectangle([0,H-14,W*g/tot,H],fill=YEL)
        arr=np.asarray(im).astype(np.float32)
        fade=min(1,t/0.3,(dur-t)/0.3)
        arr=np.asarray(BG,np.float32)+(arr-np.asarray(BG,np.float32))*fade
        p.stdin.write(arr.astype(np.uint8).tobytes())
p.stdin.close(); p.wait(); print('ok',tot)
