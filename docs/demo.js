const canvas=document.getElementById('demo-canvas');
const ctx=canvas.getContext('2d');
const thresholdInput=document.getElementById('threshold');
const thresholdValue=document.getElementById('threshold-value');
const metricThreshold=document.getElementById('metric-threshold');
const candidateCount=document.getElementById('candidate-count');
const highestScore=document.getElementById('highest-score');
const statusText=document.getElementById('demo-status');
const runButton=document.getElementById('run-demo');
const resetButton=document.getElementById('reset-demo');
const stageEls=[...document.querySelectorAll('.stage')];

const candidates=[
{x:160,y:132,length:34,angle:-.45,score:.94},{x:294,y:215,length:29,angle:.52,score:.88},
{x:432,y:128,length:36,angle:1.05,score:.81},{x:598,y:302,length:31,angle:-.78,score:.76},
{x:756,y:168,length:27,angle:.18,score:.69},{x:225,y:418,length:30,angle:.84,score:.62},
{x:485,y:448,length:25,angle:-.18,score:.56},{x:778,y:425,length:28,angle:-1.12,score:.47}
];
const backgroundCells=[[88,95,36],[205,78,28],[338,104,42],[535,74,30],[702,92,37],[870,108,30],[105,254,31],[245,291,44],[391,258,29],[520,238,36],[675,246,43],[840,272,35],[74,438,41],[337,455,34],[625,452,39],[870,440,31],[390,535,27],[710,530,32]];
let finished=false;let runToken=0;

function drawField(showBoxes=false){
 const gradient=ctx.createLinearGradient(0,0,canvas.width,canvas.height);gradient.addColorStop(0,'#d9e9f4');gradient.addColorStop(.55,'#c9deeb');gradient.addColorStop(1,'#e6eff5');ctx.fillStyle=gradient;ctx.fillRect(0,0,canvas.width,canvas.height);
 for(const [x,y,r] of backgroundCells){const g=ctx.createRadialGradient(x-r*.2,y-r*.2,4,x,y,r);g.addColorStop(0,'rgba(97,126,163,.25)');g.addColorStop(1,'rgba(80,111,150,.05)');ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill()}
 ctx.fillStyle='rgba(68,96,128,.12)';for(let i=0;i<50;i+=1){const x=(i*137+41)%canvas.width;const y=(i*83+67)%canvas.height;ctx.beginPath();ctx.arc(x,y,2+(i%4),0,Math.PI*2);ctx.fill()}
 candidates.forEach(drawRod);
 if(showBoxes){const threshold=Number(thresholdInput.value);const retained=candidates.filter(item=>item.score>=threshold);retained.forEach(drawBox);updateMetrics(retained)}else updateMetrics([])
}
function drawRod(item){const half=item.length/2;ctx.save();ctx.translate(item.x,item.y);ctx.rotate(item.angle);ctx.lineCap='round';ctx.strokeStyle='rgba(126,20,36,.20)';ctx.lineWidth=9;ctx.beginPath();ctx.moveTo(-half,0);ctx.lineTo(half,0);ctx.stroke();ctx.strokeStyle='#b41f3c';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(-half,0);ctx.lineTo(half,0);ctx.stroke();ctx.restore()}
function drawBox(item){const boxW=item.length+28,boxH=28;ctx.save();ctx.translate(item.x,item.y);ctx.rotate(item.angle);ctx.strokeStyle='#f4f7fb';ctx.lineWidth=6;ctx.strokeRect(-boxW/2,-boxH/2,boxW,boxH);ctx.strokeStyle='#9d2f38';ctx.lineWidth=3;ctx.strokeRect(-boxW/2,-boxH/2,boxW,boxH);ctx.restore();const label=item.score.toFixed(2);ctx.font='700 15px ui-monospace, SFMono-Regular, Menlo, monospace';const width=ctx.measureText(label).width+16;ctx.fillStyle='#9d2f38';ctx.fillRect(item.x-width/2,item.y-36,width,24);ctx.fillStyle='#fff';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(label,item.x,item.y-24)}
function updateMetrics(retained){candidateCount.textContent=String(retained.length);highestScore.textContent=retained.length?retained[0].score.toFixed(2):'—';metricThreshold.textContent=Number(thresholdInput.value).toFixed(2)}
function setStage(index,state){stageEls.forEach((element,stageIndex)=>{element.classList.remove('active','done');if(stageIndex<index)element.classList.add('done');if(stageIndex===index&&state==='active')element.classList.add('active');if(stageIndex<=index&&state==='done')element.classList.add('done')})}
function wait(ms){return new Promise(resolve=>setTimeout(resolve,ms))}
async function runDemo(){const token=++runToken;finished=false;runButton.disabled=true;thresholdInput.disabled=true;drawField(false);const steps=[['Loading synthetic microscopy field…',0,420],['Dividing the field into inference tiles…',1,520],['Scoring illustrative AFB candidates…',2,620],['Applying confidence threshold and overlap filtering…',3,560],['Presenting retained candidates for expert review.',4,360]];for(const [message,index,delay] of steps){if(token!==runToken)return;statusText.textContent=message;setStage(index,'active');await wait(delay)}if(token!==runToken)return;finished=true;setStage(4,'done');statusText.textContent='Demonstration complete';drawField(true);runButton.disabled=false;thresholdInput.disabled=false}
function resetDemo(){runToken+=1;finished=false;runButton.disabled=false;thresholdInput.disabled=false;statusText.textContent='Ready';stageEls.forEach(element=>element.classList.remove('active','done'));drawField(false)}
thresholdInput.addEventListener('input',()=>{const value=Number(thresholdInput.value).toFixed(2);thresholdValue.textContent=value;metricThreshold.textContent=value;if(finished)drawField(true)});
runButton.addEventListener('click',runDemo);
resetButton.addEventListener('click',resetDemo);
drawField(false);
