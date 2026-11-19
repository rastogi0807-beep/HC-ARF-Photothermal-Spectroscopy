from pathlib import Path
import subprocess,hashlib
from io import BytesIO
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
def main():
    source='Agentic_HCARF_Photothermal_Digital_Twin.tex';job='Agentic_HCARF_Review'
    for _ in range(3):
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname='+job,source],cwd=ROOT/'latex',capture_output=True)
        if result.returncode:raise RuntimeError(result.stdout.decode(errors='replace')[-3500:])
    log=(ROOT/'latex'/f'{job}.log').read_text();(ROOT/'latex/final_build.log').write_text(log)
    data=(ROOT/'latex'/f'{job}.pdf').read_bytes();reader=PdfReader(BytesIO(data));assert len(reader.pages)>5
    target=ROOT/'output/Agentic_HCARF_Photothermal_Digital_Twin_Manuscript.pdf'
    temporary=target.with_suffix('.pdf.tmp');temporary.write_bytes(data);temporary.replace(target)
    print('PDF pages',len(reader.pages),'bytes',len(data),'SHA256',hashlib.sha256(data).hexdigest())
    for line in log.splitlines():
        if any(word in line for word in ['Overfull','Warning:']):print(line)
if __name__=='__main__':main()
