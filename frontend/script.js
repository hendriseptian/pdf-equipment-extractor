const API_BASE = "https://pdf-equipment-extractor.side-gs78.workers.dev";
const pdfInput = document.getElementById('pdfInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const statusEl = document.getElementById('status');
const resultCard = document.getElementById('resultCard');
const resultTitle = document.getElementById('resultTitle');
const meta = document.getElementById('meta');
const tbody = document.getElementById('tbody');
const downloadBtn = document.getElementById('downloadBtn');
let selectedFile = null;
let rows = [];

const pdfjsLibPromise = import('https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.mjs').then(m=>{
  m.GlobalWorkerOptions.workerSrc='https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.mjs';
  return m;
});

function setStatus(t, cls=''){statusEl.textContent=t;statusEl.className='status '+cls}
function b64(dataUrl){return dataUrl.split(',')[1]}
async function canvasData(canvas){return canvas.toDataURL('image/jpeg',0.82)}

async function renderPage(page, pageNo){
  const baseViewport = page.getViewport({scale:3.0});
  const full = document.createElement('canvas'); full.width=Math.ceil(baseViewport.width); full.height=Math.ceil(baseViewport.height);
  await page.render({canvasContext:full.getContext('2d'),viewport:baseViewport}).promise;
  const fullUrl=await canvasData(full);
  const cols=3, rowsN=2, overlap=.12;
  const tiles=[];
  for(let r=0;r<rowsN;r++) for(let c=0;c<cols;c++){
    const x0=Math.max(0,Math.floor(c*full.width/cols-overlap*full.width/cols));
    const x1=Math.min(full.width,Math.ceil((c+1)*full.width/cols+overlap*full.width/cols));
    const y0=Math.max(0,Math.floor(r*full.height/rowsN-overlap*full.height/rowsN));
    const y1=Math.min(full.height,Math.ceil((r+1)*full.height/rowsN+overlap*full.height/rowsN));
    const c2=document.createElement('canvas'); c2.width=x1-x0;c2.height=y1-y0;
    c2.getContext('2d').drawImage(full,x0,y0,c2.width,c2.height,0,0,c2.width,c2.height);
    tiles.push({id:`P${pageNo}-R${r+1}C${c+1}`,mime_type:'image/jpeg',data:b64(await canvasData(c2))});
  }
  return {page_no:pageNo,full_page:{mime_type:'image/jpeg',data:b64(fullUrl)},tiles};
}

function renderRows(){
  tbody.innerHTML='';
  rows.forEach((r,i)=>{
    const tr=document.createElement('tr');
    const vals=[i+1,r.tag_no,r.pid_no,r.from,r.to,r.size];
    vals.forEach((v,j)=>{const td=document.createElement('td'); if(j===0){td.textContent=v}else{const inp=document.createElement('input');inp.value=v??'';inp.dataset.i=i;inp.dataset.k=['tag_no','pid_no','from','to','size'][j-1];inp.addEventListener('input',e=>rows[+e.target.dataset.i][e.target.dataset.k]=e.target.value);td.appendChild(inp)} tr.appendChild(td)});
    tbody.appendChild(tr);
  });
}

pdfInput.addEventListener('change',()=>{selectedFile=pdfInput.files?.[0]||null;analyzeBtn.disabled=!selectedFile;setStatus(selectedFile?`${selectedFile.name} ready for analysis.`:'No PDF selected.')});
clearBtn.addEventListener('click',()=>{selectedFile=null;pdfInput.value='';rows=[];resultCard.hidden=true;analyzeBtn.disabled=true;setStatus('No PDF selected.')});

analyzeBtn.addEventListener('click',async()=>{
  if(!selectedFile)return;
  analyzeBtn.disabled=true; resultCard.hidden=true;
  try{
    setStatus('Loading PDF renderer...');
    const pdfjsLib=await pdfjsLibPromise;
    const data=new Uint8Array(await selectedFile.arrayBuffer());
    const pdf=await pdfjsLib.getDocument({data}).promise;
    const pages=[];
    for(let p=1;p<=pdf.numPages;p++){
      setStatus(`Rendering page ${p}/${pdf.numPages} at high resolution...`);
      pages.push(await renderPage(await pdf.getPage(p),p));
    }
    setStatus(`AI is scanning ${pdf.numPages} page(s) with overlapping visual tiles...`);
    const resp=await fetch(`${API_BASE}/api/analyze-piping`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:selectedFile.name,pages})});
    const dataJson=await resp.json();
    if(!resp.ok) throw new Error(dataJson.detail||`HTTP ${resp.status}`);
    rows=dataJson.piping_tags||[];
    resultTitle.textContent=`${rows.length} piping tag${rows.length===1?'':'s'} detected`;
    meta.textContent=`P&ID No.: ${dataJson.drawing_no||'-'} · PDF: ${selectedFile.name}`;
    renderRows(); resultCard.hidden=false;
    setStatus(`Analysis completed. ${rows.length} piping tag(s) detected.`);
  }catch(e){console.error(e);setStatus(`Error: ${e.message}`,'error')}finally{analyzeBtn.disabled=!selectedFile}
});

downloadBtn.addEventListener('click',async()=>{
  try{
    if(!rows.length) return;
    const wb=new ExcelJS.Workbook();
    const res=await fetch('./template/piping.xlsx');
    if(!res.ok) throw new Error('Template piping.xlsx tidak ditemukan di GitHub Pages.');
    await wb.xlsx.load(await res.arrayBuffer());
    const ws=wb.getWorksheet(1);
    if(!ws) throw new Error('Sheet template tidak ditemukan.');
    const styleSource = ws.getRow(4);
    const sourceStyles = [];
    for(let c=1;c<=5;c++){ const cell=styleSource.getCell(c); sourceStyles.push({style:cell.style?JSON.parse(JSON.stringify(cell.style)):null,numFmt:cell.numFmt}); }
    while(ws.rowCount>3) ws.spliceRows(4,ws.rowCount-3);
    rows.forEach((r,idx)=>{
      const row=ws.getRow(4+idx);
      row.values=[r.tag_no,r.pid_no,r.from||'',r.to||'',r.size];
      row.height=styleSource.height;
      for(let c=1;c<=5;c++){
        const dst=row.getCell(c); const src=sourceStyles[c-1];
        if(src.style) dst.style=JSON.parse(JSON.stringify(src.style));
        if(src.numFmt) dst.numFmt=src.numFmt;
      }
    });
    ws.getColumn(1).width=18.285;ws.getColumn(2).width=9.855;ws.getColumn(3).width=19;ws.getColumn(4).width=21.71;ws.getColumn(5).width=12.425;
    const out=await wb.xlsx.writeBuffer();
    const blob=new Blob([out],{type:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'});
    const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='piping_extracted.xlsx';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  }catch(e){alert(e.message)}
});
