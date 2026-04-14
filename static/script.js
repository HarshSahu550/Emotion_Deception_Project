const EMOTION_COLORS = {
  Angry:    "#dc1414",
  Disgust:  "#008c00",
  Fear:     "#820082",
  Happy:    "#ffd200",
  Neutral:  "#a0a0a0",
  Sad:      "#c86400",
  Surprise: "#ff8c00"
}

const DECEPTION_COLORS = {
  "Low":          "#00c800",
  "Medium":       "#ffa500",
  "High":         "#dc1414",
  "Analyzing...": "#a0a0a0"
}

function buildBars(allProbs) {
  const container = document.getElementById('bars')
  container.innerHTML = ''
  for (const [emotion, prob] of Object.entries(allProbs)) {
    const pct   = Math.round(prob * 100)
    const color = EMOTION_COLORS[emotion] || '#888'
    container.innerHTML += `
      <div class="bar-row">
        <span class="bar-label">${emotion}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%; background:${color}"></div>
        </div>
        <span class="bar-pct">${pct}%</span>
      </div>`
  }
}

async function poll() {
  try {
    const res  = await fetch('/data')
    const data = await res.json()

    document.getElementById('emotion').textContent    = data.emotion
    document.getElementById('confidence').textContent = `${Math.round(data.confidence * 100)}% confidence`
    
    const dlabel = document.getElementById('deception_label')
    dlabel.textContent = data.deception_label
    dlabel.style.color = DECEPTION_COLORS[data.deception_label] || '#e0e0e0'

    document.getElementById('deception_score').textContent =
      `Score: ${data.deception_score.toFixed(2)}`

    if (data.all_probs && Object.keys(data.all_probs).length)
      buildBars(data.all_probs)

  } catch(e) {
    console.error('Poll error:', e)
  }
}

setInterval(poll, 500)
poll()
```

---

### STEP 6 — `requirements.txt` (add one line)
```
flask