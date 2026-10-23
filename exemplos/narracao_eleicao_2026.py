from kokoro_onnx import Kokoro; import soundfile as sf, numpy as np
k=Kokoro('../modelos/kokoro/kokoro-v1.0.onnx','../modelos/kokoro/voices-v1.0.bin')
S=[(6,"Faltam só dois dias para o segundo turno. Quem vai ganhar a presidência?"),
(8,"A disputa é entre Lula, do PT, e Flávio Bolsonaro, do PL."),
(12,"No Datafolha, os dois empatam com cinquenta por cento dos votos válidos. Nos votos totais, Flávio tem quarenta e sete, e Lula, quarenta e seis. Empate técnico."),
(9,"Na Quaest, Flávio tem quarenta e cinco por cento, e Lula, quarenta e quatro. De novo, dentro da margem de erro."),
(10,"E a tendência? A vantagem de Flávio no Datafolha caiu de quatro pontos para zero. Lula encostou na reta final."),
(8,"O governo Lula tem quarenta e nove por cento de aprovação. Abstenção, indecisos e votos brancos podem decidir."),
(7,"Resumo: empate técnico. Quem decide é você, no domingo, dia vinte e cinco.")]
SR=24000; out=np.zeros(60*SR,np.float32); t0=0
for dur,txt in S:
    sp=1.05
    while True:
        a,sr=k.create(txt,voice='pm_alex',speed=sp,lang='pt-br')
        if len(a)/sr<=dur-0.7 or sp>1.4: break
        sp+=0.05
    st=int((t0+0.35)*SR); out[st:st+len(a)]+=a[:len(out)-st]
    print(dur,round(len(a)/sr,2),sp); t0+=dur
out/=max(1e-6,np.abs(out).max())/0.9
sf.write('narracao.wav',out,SR)
