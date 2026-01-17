// Demo App Logic

const API_ENDPOINT = '/api/analyze';

async function analyzeNews() {
    const input = document.getElementById('news-input').value.trim();
    if (!input) return;

    // UI States
    const btn = document.querySelector('.btn-analyze');
    const loader = document.getElementById('btn-loader');
    const btnText = document.querySelector('.btn-text');
    
    // Reset View
    document.getElementById('result-card').querySelector('.empty-state').classList.add('hidden');
    document.getElementById('results-content').classList.add('hidden');
    
    // Loading State
    btn.disabled = true;
    loader.style.display = 'block';
    btnText.style.opacity = '0';
    
    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input })
        });
        
        const data = await response.json();
        
        // Simulate network delay for UX if local mock is too fast
        if(window.location.protocol === 'file:') {
            // Fallback for file:// usage without backend
            await new Promise(r => setTimeout(r, 1500)); 
            displayResults(getMockData(input));
        } else {
            if (data.error) throw new Error(data.error);
            displayResults(data);
        }
        
    } catch (e) {
        console.error("Analysis failed", e);
        // Fallback to mock if server not running for demo
        displayResults(getMockData(input));
    } finally {
        // Reset Button
        btn.disabled = false;
        loader.style.display = 'none';
        btnText.style.opacity = '1';
    }
}

function displayResults(data) {
    const content = document.getElementById('results-content');
    const verdictText = document.getElementById('verdict-text');
    const confidenceScore = document.getElementById('confidence-score');
    const confidenceFill = document.getElementById('confidence-fill');
    const heatmap = document.getElementById('heatmap-container');
    
    content.classList.remove('hidden');
    
    // 1. Verdict
    const isFake = data.prediction.toLowerCase() === 'fake';
    verdictText.innerText = isFake ? 'LIKELY FAKE' : 'LIKELY REAL';
    verdictText.className = isFake ? 'verdict-fake' : 'verdict-real';
    
    // 2. Confidence
    const scorePct = Math.round(data.confidence * 100);
    confidenceScore.innerText = scorePct;
    confidenceFill.style.width = `${scorePct}%`;
    confidenceFill.style.background = isFake 
        ? 'linear-gradient(90deg, #ef4444, #f87171)' 
        : 'linear-gradient(90deg, #10b981, #34d399)';
        
    // 3. Heatmap Explanation
    heatmap.innerHTML = '';
    
    if (data.tokens && data.attributions) {
        data.tokens.forEach((token, index) => {
            const attr = data.attributions[index];
            const span = document.createElement('span');
            span.className = 'token';
            span.innerText = token + ' ';
            
            // Color mapping based on importance
            // Red for suspicious (high importance in Fake), Green/Transparent for neutral
            // For demo: higher attribute = darker red
            const opacity = Math.min(Math.abs(attr) * 1.5, 1); // boost visibility
            
            if (isFake) {
                span.style.backgroundColor = `rgba(239, 68, 68, ${opacity})`;
            } else {
                 span.style.backgroundColor = `rgba(16, 185, 129, ${opacity * 0.5})`;
            }
            
            heatmap.appendChild(span);
        });
    }
}

function clearInput() {
    document.getElementById('news-input').value = '';
    document.getElementById('results-content').classList.add('hidden');
    document.getElementById('result-card').querySelector('.empty-state').classList.remove('hidden');
}

// Fallback Mock Data generator for client-side demo
function getMockData(text) {
    const isFake = /miracle|shocking|secret|guaranteed|viruses/i.test(text);
    const tokens = text.split(/\s+/);
    const attributions = tokens.map(t => {
        if (/miracle|shocking|secret|guaranteed|viruses/i.test(t)) return 0.9;
        return Math.random() * 0.2;
    });
    
    return {
        prediction: isFake ? 'Fake' : 'Real',
        confidence: 0.85 + Math.random() * 0.1,
        tokens: tokens,
        attributions: attributions
    };
}
