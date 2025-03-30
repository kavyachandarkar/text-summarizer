async function summarizeText() {
    const text = document.getElementById('inputText').value;
    const method = document.getElementById('method').value;
    
    const response = await fetch('http://127.0.0.1:5000/summarize', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text, method })
    });
    
    const data = await response.json();
    document.getElementById('summaryOutput').innerText = data.summary || 'Error summarizing text';
}