import React, { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [method, setMethod] = useState("nltk");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);

  const summarizeText = async () => {
    if (!text) {
      alert("Please enter text to summarize!");
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch("http://127.0.0.1:5000/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, method }),
      });

      const data = await response.json();
      setSummary(data.summary || "Error summarizing text.");
    } catch (error) {
      console.error("Error:", error);
      setSummary("Failed to connect to the server.");
    }

    setLoading(false);
  };

  return (
    <div style={{ maxWidth: "600px", margin: "50px auto", textAlign: "center" }}>
      <h2>Text Summarizer</h2>
      <textarea 
        rows="6"
        placeholder="Enter text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{ width: "100%", padding: "10px", marginBottom: "10px" }}
      />
      
      <select value={method} onChange={(e) => setMethod(e.target.value)}>
        <option value="nltk">NLTK</option>
        <option value="spacy">SpaCy</option>
        <option value="word2vec">Word2Vec</option>
      </select>
      
      <button onClick={summarizeText} style={{ marginLeft: "10px", padding: "5px 10px" }}>
        {loading ? "Summarizing..." : "Summarize"}
      </button>
      
      <h3>Summary:</h3>
      <p>{summary}</p>
    </div>
  );
}

export default App;
