"""Render complete page buffers atomically for private visual review."""
from pathlib import Path
from io import BytesIO
import subprocess
from PIL import Image,ImageDraw
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'output/Agentic_HCARF_Photothermal_Digital_Twin_Manuscript.pdf';out=ROOT/'qa/pdf_release';out.mkdir(exist_ok=True)
    for i in range(1,len(PdfReader(source).pages)+1):
        r=subprocess.run(['pdftoppm','-f',str(i),'-l',str(i),'-singlefile','-scale-to','1400','-png',str(source)],capture_output=True,check=True)
        Image.open(BytesIO(r.stdout)).load();p=out/f'page-{i:02}.png';p.write_bytes(r.stdout)
    for name in ['pdf_release','word_verified']:
        p=ROOT/'qa'/name;files=sorted(p.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
        for batch in range((len(files)+5)//6):
            canvas=Image.new('RGB',(1500,1390),'#DFE4E9')
            for j,f in enumerate(files[6*batch:6*batch+6]):
                im=Image.open(f).convert('RGB');im.thumbnail((482,646));x=(j%3)*500+(500-im.width)//2;y=(j//3)*695+10
                canvas.paste(im,(x,y));ImageDraw.Draw(canvas).text(((j%3)*500+12,(j//3)*695+668),f.name,fill='black')
            canvas.save(p/f'contact_{batch+1}.png')
        print(name,len(files),'complete page images')

if __name__=='__main__':main()
