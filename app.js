/* Wine & Food Pairing — Neural Recommender (tf.js)
 * - Loads CSV (PapaParse)
 * - EDA (tfjs-vis + light HTML tables)
 * - Preprocess: build vocabularies, one-hot tensors
 * - Model: Dense NN (5-way softmax), metrics, confusion matrix, per-class F1
 * - Prototype: interactive prediction + "Suggest Better Wine"
 */

const state = {
  raw: [],
  df: [],
  X: null, y: null,
  Xtrain: null, ytrain: null,
  Xtest: null, ytest: null,
  model: null,
  vocabs: {},
  cats: ["wine_type","wine_category","food_item","food_category","cuisine"],
  classes: [1,2,3,4,5],
};

// ---------- Helpers ----------
const $ = (sel) => document.querySelector(sel);
const setStatus = (msg) => $("#status").textContent = msg;

function head(arr, n=10){ return arr.slice(0, n); }
function uniq(arr){ return [...new Set(arr)]; }
function countBy(arr, key){
  const m = new Map();
  for (const r of arr){ const k = r[key]; m.set(k, (m.get(k)||0)+1); }
  return [...m.entries()].sort((a,b)=>b[1]-a[1]);
}
function toTable(rows, headers){
  const htmlHead = `<tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr>`;
  const htmlRows = rows.map(r=>`<tr>${headers.map(h=>`<td>${r[h]}</td>`).join("")}</tr>`).join("");
  return `<table>${htmlHead}${htmlRows}</table>`;
}
function showHTML(id, html){ $(id).innerHTML = html; }

// ---------- Load CSV ----------
$("#csv-file").addEventListener("change", (e)=>{
  const file = e.target.files[0];
  if(!file){ return; }
  Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: (res)=>{
      state.raw = res.data.filter(r =>
        r.wine_type && r.wine_category && r.food_item && r.food_category && r.cuisine && r.pairing_quality);
      state.df = state.raw.map(r => ({
        wine_type: String(r.wine_type).trim(),
        wine_category: String(r.wine_category).trim(),
        food_item: String(r.food_item).trim(),
        food_category: String(r.food_category).trim(),
        cuisine: String(r.cuisine).trim(),
        pairing_quality: Number(r.pairing_quality),
        quality_label: r.quality_label ? String(r.quality_label).trim() : String(r.pairing_quality),
        description: r.description ? String(r.description).trim() : ""
      }));
      setStatus(`Loaded ${state.df.length} rows`);
      $("#btn-preview").disabled = false;
      $("#btn-eda").disabled = false;
      $("#btn-prep").disabled = false;
    }
  });
});

// ---------- Preview ----------
$("#btn-preview").addEventListener("click", ()=>{
  const rows = head(state.df, 20);
  const headers = ["wine_type","wine_category","food_item","food_category","cuisine","pairing_quality","quality_label","description"];
  showHTML("#preview", toTable(rows, headers));
});

// ---------- EDA ----------
$("#btn-eda").addEventListener("click", ()=>{
  const el = $("#eda");
  el.innerHTML = "";

  // Basic stats
  const n = state.df.length;
  const qualCounts = countBy(state.df, "pairing_quality");
  const wc = countBy(state.df, "wine_category");
  const fc = countBy(state.df, "food_category");
  const cu = countBy(state.df, "cuisine");

  const statCard = document.createElement("div");
  statCard.className = "row";
  statCard.innerHTML = `
    <div class="card"><b>Rows:</b> ${n}<br/><b>Distinct wines:</b> ${uniq(state.df.map(r=>r.wine_type)).length}
      <br/><b>Food items:</b> ${uniq(state.df.map(r=>r.food_item)).length}
      <br/><b>Cuisines:</b> ${uniq(state.df.map(r=>r.cuisine)).length}
    </div>
    <div class="card"><b>Class balance (1–5):</b><div id="eda-qual"></div></div>
  `;
  el.appendChild(statCard);

  // tfjs-vis bar charts
  const surface1 = { name: 'Quality distribution', tab: 'EDA' };
  const qualSeries = qualCounts.map(([k,v])=>({ index:k, value:v }));
  tfvis.render.barchart(surface1, qualSeries, { xLabel:'Quality', yLabel:'Count' });

  const surface2 = { name: 'Wine categories', tab: 'EDA' };
  tfvis.render.barchart(surface2, wc.slice(0,15).map(([k,v])=>({index:k, value:v})), { xLabel:'Wine category', yLabel:'Count' });

  const surface3 = { name: 'Food categories', tab: 'EDA' };
  tfvis.render.barchart(surface3, fc.slice(0,20).map(([k,v])=>({index:k, value:v})), { xLabel:'Food category', yLabel:'Count' });

  const surface4 = { name: 'Cuisines', tab: 'EDA' };
  tfvis.render.barchart(surface4, cu.slice(0,20).map(([k,v])=>({index:k, value:v})), { xLabel:'Cuisine', yLabel:'Count' });

  // HTML top-10 tables
  const tblRow = document.createElement("div");
  tblRow.className = "row-3";
  const mkTableCard = (title, pairs) => `
    <div class="card">
      <h3 style="margin-top:0">${title}</h3>
      ${toTable(pairs.slice(0,10).map(([k,v])=>({Key:k, Count:v})), ["Key","Count"])}
    </div>`;
  tblRow.innerHTML = mkTableCard("Top Wine Types", countBy(state.df, "wine_type"))
                   + mkTableCard("Top Food Items", countBy(state.df, "food_item"))
                   + mkTableCard("Top Cuisines", cu);
  el.appendChild(tblRow);
});

// ---------- Preprocess ----------
$("#btn-prep").addEventListener("click", ()=>{
  // Build vocabularies
  const vocabs = {};
  for(const c of state.cats){ vocabs[c] = uniq(state.df.map(r=>r[c])).sort(); }
  state.vocabs = vocabs;

  // UI selects
  const fillSel = (id, arr)=>{ const s=$(id); s.innerHTML=""; for(const v of arr){ const opt=document.createElement("option"); opt.value=v; opt.textContent=v; s.appendChild(opt);} };
  fillSel("#sel-wine-type", vocabs.wine_type);
  fillSel("#sel-wine-cat",  vocabs.wine_category);
  fillSel("#sel-food-item", vocabs.food_item);
  fillSel("#sel-food-cat",  vocabs.food_category);
  fillSel("#sel-cuisine",   vocabs.cuisine);

  // One-hot encode
  const featSize = Object.values(vocabs).reduce((s,arr)=>s+arr.length, 0);
  const rows = state.df.length;
  const X = tf.buffer([rows, featSize], 'float32');
  const y = tf.buffer([rows, 5], 'float32'); // classes 1..5 → index 0..4

  function encRow(r){
    const v = new Float32Array(featSize);
    let offset = 0;
    for(const c of state.cats){
      const idx = state.vocabs[c].indexOf(r[c]);
      if(idx>=0){ v[offset + idx] = 1; }
      offset += state.vocabs[c].length;
    }
    return v;
  }

  state.df.forEach((r, i)=>{
    const vec = encRow(r);
    vec.forEach((val,j)=>X.set(val, i, j));
    const clazz = Math.max(1, Math.min(5, Number(r.pairing_quality))) - 1;
    y.set(1, i, clazz);
  });

  state.X = X.toTensor();
  state.y = y.toTensor();

// Train/Test split (80/20)
// здесь мы НЕ объявляем rows заново — он уже есть выше
const idx = tf.util.createShuffledIndices(rows);
const nTrain = Math.floor(rows * 0.8);

// делаем обычные JS-массивы, потом int32-тензоры индексов
const tr = Array.from(idx.slice(0, nTrain));
const te = Array.from(idx.slice(nTrain));

const trIdx = tf.tensor1d(tr, 'int32');
const teIdx = tf.tensor1d(te, 'int32');

state.Xtrain = tf.gather(state.X, trIdx);
state.ytrain = tf.gather(state.y, trIdx);
state.Xtest  = tf.gather(state.X, teIdx);
state.ytest  = tf.gather(state.y, teIdx);

trIdx.dispose();
teIdx.dispose();

setStatus(`Preprocessed: features=${featSize}, train=${nTrain}, test=${rows - nTrain}`);
$("#btn-build").disabled = false;

});

// ---------- Build Model ----------
$("#btn-build").addEventListener("click", ()=>{
  const inDim = state.X.shape[1];
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: Math.min(256, Math.round(inDim*0.75)+32), activation:'relu', inputShape:[inDim] }));
  m.add(tf.layers.dropout({ rate: 0.25 }));
  m.add(tf.layers.dense({ units: 128, activation:'relu' }));
  m.add(tf.layers.dropout({ rate: 0.2 }));
  m.add(tf.layers.dense({ units: 5, activation:'softmax' }));
  m.compile({ optimizer: tf.train.adam(0.003), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  state.model = m;

  // Summary
  const div = $("#model-summary");
  div.innerHTML = "";
  tfvis.show.modelSummary({ name:'Model Summary', tab:'Model' }, m);
  $("#btn-train").disabled = false;
});

// ---------- Train ----------
$("#btn-train").addEventListener("click", async ()=>{
  $("#btn-train").disabled = true;
  const h = await state.model.fit(state.Xtrain, state.ytrain, {
    epochs: 30,
    batchSize: 32,
    validationSplit: 0.15,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training', tab: 'Model' },
      ['loss','val_loss','acc','val_acc'],
      { callbacks: ['onEpochEnd'] }
    )
  });
  $("#btn-eval").disabled = false;
});

// ---------- Evaluate ----------
$("#btn-eval").addEventListener("click", async ()=>{
  const preds = state.model.predict(state.Xtest);
  const yTrue = await state.ytest.argMax(-1).data();
  const yPred = await preds.argMax(-1).data();

  // Confusion matrix (5x5)
  const C = Array.from({length:5},()=>Array(5).fill(0));
  for(let i=0;i<yTrue.length;i++){ C[yTrue[i]][yPred[i]]++; }

  // Metrics
  const perClass = [];
  let sumF1 = 0;
  for(let k=0;k<5;k++){
    const tp = C[k][k];
    const fp = C.map(row=>row[k]).reduce((a,b)=>a+b,0) - tp;
    const fn = C[k].reduce((a,b)=>a+b,0) - tp;
    const prec = tp/(tp+fp+1e-9), rec = tp/(tp+fn+1e-9);
    const f1 = 2*prec*rec/(prec+rec+1e-9);
    perClass.push({cls:k+1, prec:prec.toFixed(3), rec:rec.toFixed(3), f1:f1.toFixed(3)});
    sumF1 += f1;
  }
  const macroF1 = (sumF1/5).toFixed(3);

  // Overall accuracy
  const acc = (yTrue.filter((v,i)=>v===yPred[i]).length / yTrue.length).toFixed(3);

  $("#m-acc").textContent = acc;
  $("#m-f1").textContent = macroF1;

  // Render CM
  tfvis.render.confusionMatrix(
    { name:'Confusion Matrix (test)', tab:'Metrics' },
    { values: C, tickLabels: ['1','2','3','4','5'] },
    { shadeDiagonal: true }
  );

  // Classification report
  const repTable = `<table>
    <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
    ${perClass.map(r=>`<tr><td>${r.cls}</td><td>${r.prec}</td><td>${r.rec}</td><td>${r.f1}</td></tr>`).join("")}
  </table>`;
  $("#class-report").innerHTML = repTable;

  $("#btn-export").disabled = false;
  $("#btn-predict").disabled = false;
  $("#btn-suggest").disabled = false;
});

// ---------- Interactive Prototype ----------
function oneHotFromSelections(sel){
  const r = {
    wine_type: $("#sel-wine-type").value,
    wine_category: $("#sel-wine-cat").value,
    food_item: $("#sel-food-item").value,
    food_category: $("#sel-food-cat").value,
    cuisine: $("#sel-cuisine").value
  };
  let offset = 0;
  const inDim = Object.values(state.vocabs).reduce((s,a)=>s+a.length,0);
  const vec = new Float32Array(inDim);
  for(const c of state.cats){
    const idx = state.vocabs[c].indexOf(r[c]);
    if(idx>=0) vec[offset+idx]=1;
    offset += state.vocabs[c].length;
  }
  return tf.tensor2d(vec, [1, inDim]);
}

$("#btn-predict").addEventListener("click", async ()=>{
  const x = oneHotFromSelections();
  const p = state.model.predict(x);
  const probs = Array.from(await p.data());
  const best = probs.indexOf(Math.max(...probs)) + 1;
  $("#pred-out").innerHTML = `
    <div class="card">
      <b>Predicted quality:</b> ${best} / 5<br/>
      <small>Probabilities: [${probs.map(v=>v.toFixed(2)).join(", ")}]</small>
    </div>`;
  x.dispose(); p.dispose();
});

// Suggest better wine (keep dish/cuisine fixed, scan wines)
$("#btn-suggest").addEventListener("click", async ()=>{
  const chosen = {
    wine_type: $("#sel-wine-type").value,
    wine_category: $("#sel-wine-cat").value,
    food_item: $("#sel-food-item").value,
    food_category: $("#sel-food-cat").value,
    cuisine: $("#sel-cuisine").value
  };
  const candidates = state.vocabs.wine_type; // try all wine types
  const inDim = Object.values(state.vocabs).reduce((s,a)=>s+a.length,0);

  const rows = [];
  for(const wt of candidates){
    const r = {...chosen, wine_type: wt, wine_category: guessWineCategory(wt) || $("#sel-wine-cat").value };
    let offset=0; const vec=new Float32Array(inDim);
    for(const c of state.cats){
      const idx = state.vocabs[c].indexOf(r[c]);
      if(idx>=0) vec[offset+idx]=1;
      offset += state.vocabs[c].length;
    }
    rows.push(vec);
  }
  const X = tf.tensor2d(rows);
  const P = state.model.predict(X);
  const probs = await P.array();
  const scored = probs.map((arr, i)=>({ wine_type: candidates[i], score: arr[0]*1 + arr[1]*2 + arr[2]*3 + arr[3]*4 + arr[4]*5 }));
  scored.sort((a,b)=>b.score-a.score);
  const top = scored.slice(0,8);
  $("#pred-out").innerHTML = `<div class="card"><b>Top suggestions (fixed dish):</b>
    ${toTable(top.map(t=>({Wine:t.wine_type, Score:t.score.toFixed(2)})), ["Wine","Score"])}</div>`;
  X.dispose(); P.dispose();
});

// heuristic: map wine_type → wine_category (fallback if unknown)
function guessWineCategory(wineType){
  const t = wineType.toLowerCase();
  if(t.includes("cabernet")||t.includes("pinot noir")||t.includes("merlot")||t.includes("syrah")||t.includes("malbec")) return "Red";
  if(t.includes("riesling")||t.includes("sauvignon blanc")||t.includes("chardonnay")||t.includes("pinot grigio")||t.includes("albarino")) return "White";
  if(t.includes("rose")||t.includes("rosé")) return "Rosé";
  if(t.includes("champagne")||t.includes("cava")||t.includes("prosecco")||t.includes("sparkling")) return "Sparkling";
  if(t.includes("port")||t.includes("sherry")||t.includes("madeira")) return "Fortified";
  if(t.includes("moscato")||t.includes("late harvest")||t.includes("sauternes")||t.includes("ice wine")) return "Dessert";
  return null;
}

// ---------- Export ----------
$("#btn-export").addEventListener("click", async ()=>{
  await state.model.save('downloads://wine_food_nn');
  alert('Model files saved (JSON + weights BIN).');
});
