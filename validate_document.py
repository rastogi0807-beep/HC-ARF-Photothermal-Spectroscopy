"""Audit manuscript structure and consistency with retained executed outputs."""
from pathlib import Path
from io import BytesIO
import hashlib,json,re,zipfile
from lxml import etree
from pypdf import PdfReader
from PIL import Image
import pandas as pd
from manuscript_content import content

ROOT=Path(__file__).resolve().parents[1]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}

def main():
    paper=content();blocks=paper['blocks'];equations=[b for b in blocks if b['kind']=='equation']
    figs=[b for b in blocks if b['kind']=='figure'];tables=[b for b in blocks if b['kind']=='table']
    refs=[b for b in blocks if b['kind']=='reference'];abstract=next(b['text'] for b in blocks if b['kind']=='abstract')
    assert len(equations)==27 and [e['number'] for e in equations]==list(range(1,28))
    assert len(figs)==10 and len(tables)==8 and len(refs)==26
    body='\n'.join(b.get('text','') for b in blocks if b['kind']!='reference')
    assert set(map(int,re.findall(r'\[(\d+)\]',body)))==set(range(1,27))
    assert not re.search(r'\b(TODO|TBD|PLACEHOLDER)\b',body)
    assert 150<=len(abstract.split())<=250
    for t in tables:
        assert abs(sum(t['widths'])-7)<1e-9
        assert all(len(r)==len(t['headers']) for r in t['rows'])
        assert not any('$' in str(x) for row in t['rows'] for x in row)
    s=json.loads((ROOT/'data/results_summary.json').read_text());g=s['primary']['Goal agent']
    assert g['n']==240 and g['path_covered']['count']==238 and g['report']['count']==114
    assert abs(g['RMSE_ppm']-.2614626323977914)<1e-12
    assert abs(g['mean_stop_ms']-745)<1e-10
    frame=pd.read_csv(ROOT/'data/episode_metrics.csv')
    assert len(frame)==2112
    assert s['challenges']['Spectral interferent']['Goal agent']['terminal_covered']['count']==7
    assert s['challenges']['Line strength +10%']['Goal agent']['final_flag']['count']==0
    for p,v in s['primary'].items():
        f=frame[(frame.policy==p)&(frame.setting=='Nominal')]
        assert len(f)==240 and f.report.sum()==v['report']['count']
    research=json.loads((ROOT/'references/references_verified.json').read_text())
    assert len({r['doi'] for r in research})==25
    assert sum(r['year']==2026 for r in research)==9
    docx=ROOT/'output/Agentic_HCARF_Photothermal_Digital_Twin_Manuscript.docx'
    with zipfile.ZipFile(docx) as z:
        assert z.testzip() is None
        xml=etree.fromstring(z.read('word/document.xml'))
        fields=[n for n in xml.findall('.//w:fldSimple',NS) if 'SEQ Equation' in n.get('{'+NS['w']+'}instr','')]
        assert [int(''.join(n.itertext())) for n in fields]==list(range(1,28))
        assert len([t for t in xml.findall('.//w:tbl',NS) if t.find('.//m:oMath',NS) is not None])==27
        assert len(xml.findall('.//wp:inline',NS))==10
        assert all(e.get('descr') for e in xml.findall('.//wp:docPr',NS))
        assert xml.find('.//w:sectPr/w:cols',NS).get('{'+NS['w']+'}num')=='1'
        assert '\\begin{' not in ''.join(xml.itertext())
        rel=etree.fromstring(z.read('word/_rels/document.xml.rels'))
        assert len([e for e in rel if e.get('Target','').startswith('https://doi.org/')])==25
    pdf=ROOT/'output/Agentic_HCARF_Photothermal_Digital_Twin_Manuscript.pdf'
    reader=PdfReader(BytesIO(pdf.read_bytes()));assert 10<=len(reader.pages)<=20
    text='\n'.join(p.extract_text() for p in reader.pages)
    assert '\ufffd' not in text and '??' not in text
    assert all('['+str(i)+']' in text for i in range(1,27))
    log=(ROOT/'latex/final_build.log').read_text()
    assert 'Overfull' not in log and 'undefined' not in log.lower()
    assets=[]
    for stem in [f['name'] for f in figs]+['Graphical_Abstract']:
        assert all((ROOT/'figures'/f'{stem}.{ext}').is_file() for ext in ['pdf','svg','png'])
        assert len(PdfReader(ROOT/'figures'/f'{stem}.pdf').pages)==1
        with Image.open(ROOT/'figures'/f'{stem}.png') as im:
            im.load();assert min(im.info['dpi'])>299
            assets.append(dict(name=stem,pixels=list(im.size),dpi=list(im.info['dpi'])))
    scientific=json.loads((ROOT/'data/results_validation.json').read_text());assert scientific['passed']
    assert json.loads((ROOT/'data/physics_validation.json').read_text())['passed']
    report=dict(passed=True,numbered_native_equations=27,manuscript_figures=10,data_tables=8,
                algorithm_blocks=1,standalone_graphical_abstract=True,research_references=25,
                research_references_2026=9,research_references_2024_onward=19,AI_system_reference=1,
                abstract_words=len(abstract.split()),prose_words=sum(len(b.get('text','').split()) for b in blocks if b['kind'] in ['p','abstract']),
                journal_style_PDF_pages=len(reader.pages),single_column_Word=True,
                native_equation_fields_sequential=True,DOI_hyperlinks=25,
                no_overfull_latex_boxes=True,all_latex_citations_resolved=True,
                scientific_replay_counts=scientific['counts'],figure_assets=assets,
                release_documents_SHA256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [docx,pdf]},
                scope='Document and numerical consistency checks, not certification of novelty, experimental validity or journal acceptance.')
    rendered=list((ROOT/'qa/word_verified').glob('*.pdf'))
    if rendered:report['rendered_Word_pages']=len(PdfReader(rendered[0]).pages)
    (ROOT/'VALIDATION_REPORT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='figure_assets'},indent=2))

if __name__=='__main__':main()
