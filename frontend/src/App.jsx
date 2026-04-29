
import { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [language, setLanguage] = useState('en');
  const [showTestModal, setShowTestModal] = useState(false);

  // Dynamic API URL - works for both localhost and production
  const API_URL = import.meta.env.MODE === 'production' 
    ? 'https://fakenewsdetection-majorproject.onrender.com'
    : 'http://localhost:5001';

  const languages = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'hi', name: 'हिंदी', flag: '🇮🇳' },
    { code: 'gu', name: 'ગુજરાતી', flag: '🇮🇳' },
    { code: 'mr', name: 'मराठी', flag: '🇮🇳' }
  ];

  const testExamples = {
    en: [
      {
        title: 'Real News',
        text: 'Scientists at Stanford University published peer-reviewed research in Nature Journal demonstrating that renewable energy adoption reduces carbon emissions by 40%. The study involved 500 research institutions across 50 countries over 8 years.'
      },
      {
        title: 'Fake News',
        text: 'SHOCKING! Scientists discover that 5G towers control your mind! Multiple sources confirm government conspiracy. Doctors hate this one weird trick. Share before Facebook removes it!'
      }
    ],
    hi: [
      {
        title: 'Real News',
        text: 'भारतीय सरकार ने राष्ट्रीय शिक्षा नीति 2023 की घोषणा की जो 5 लाख स्कूलों में लागू होगी। यह नीति शिक्षा विशेषज्ञों की एक अंतरराष्ट्रीय टीम द्वारा तैयार की गई है।'
      },
      {
        title: 'Fake News',
        text: 'ब्रेकिंग न्यूज! विदेशी देश भारत को कमजोर करने के लिए साजिश कर रहे हैं! अज्ञात स्रोत सांगते हैं। इसे साझा करो!'
      }
    ],
    gu: [
      {
        title: 'Real News',
        text: 'ગુજરાત સરકારે નવો ડિજિટલ શિક્ષા કાર્યક્રમ શરૂ કર્યો છે જે 1000 શાળાઓમાં લાગુ થશે. આ કાર્યક્રમ યુનેસ્કોના માર્ગદર્શન અનુસાર તૈયાર છે.'
      },
      {
        title: 'Fake News',
        text: 'આવશ્યક! ગુજરાતમાં વિદેશી દેશો એક ગુપ્ત યોજના બનાવી રહ્યા છે! અજ્ઞાત સ્ત્રોત। શેર કરો!'
      }
    ],
    mr: [
      {
        title: 'Real News',
        text: 'महाराष्ट्र सरकारने नवीन स्मार्ट सिटी प्रकल्प जाहीर केला जो 500000 नागरिकांना सेवा देणार आहे. हा प्रकल्प भारतीय तंत्रज्ञान संस्थानांद्वारे डिजाइन केला आहे।'
      },
      {
        title: 'Fake News',
        text: 'धक्कादायक! काहीजण महाराष्ट्रांचा विनाश करण्यासाठी संयोजन करत आहेत! अज्ञात स्रोत। शेयर करा!'
      }
    ]
  };

  const analyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }
    setLoading(true);
    setError('');
    try {
      console.log(`📡 Calling: ${API_URL}/predict`);
      
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language })
      });
      
      if (!res.ok) {
        throw new Error(`Backend error: ${res.status}`);
      }
      
      const data = await res.json();
      console.log('✅ Response received:', data);
      setResult(data);
    } catch (e) {
      console.error('❌ Error:', e);
      setError(`Connection failed: ${e.message}. Backend URL: ${API_URL}`);
    }
    setLoading(false);
  };

  const loadTestExample = (example) => {
    setText(example.text);
    setResult(null);
    setShowTestModal(false);
  };

  const clearAll = () => {
    setText('');
    setResult(null);
    setError('');
  };

  const wordCount = text.trim().split(/\s+/).filter(w => w).length;

  return (
    <div className="app">
      <div className="gradient-bg">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>
      </div>

      <div className="container">
        {/* Header */}
        <div className="header">
          <div className="header-left">
            <div className="badge">AI-POWERED VERIFICATION</div>
            <h1>FactCheck AI</h1>
            <p>Analyze multilingual news text with a refined credibility experience built for modern misinformation screening.</p>
          </div>
          <div className="header-right">
            <div className="feature-badge">Confidence-aware output</div>
            <div className="feature-badge">Fake / Real / Uncertain</div>
            <div className="feature-badge">Fast Flask API integration</div>
          </div>
        </div>

        {/* Language Selector */}
        <div className="language-selector">
          {languages.map(lang => (
            <button
              key={lang.code}
              className={`lang-btn ${language === lang.code ? 'active' : ''}`}
              onClick={() => setLanguage(lang.code)}
            >
              <span>{lang.flag}</span>
              <span>{lang.name}</span>
            </button>
          ))}
        </div>

        {/* Main Content */}
        <div className="main-grid">
          {/* Left Panel - Input */}
          <div className="input-panel">
            <div className="panel-header">
              <div className="section-label">INPUT</div>
              <div className="char-count">{wordCount} characters</div>
            </div>

            <h2>Paste a claim or article</h2>

            <div className="input-label">News content</div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste a news article, viral message, or suspicious claim here..."
              rows="8"
            />

            {error && <div className="error-msg">{error}</div>}

            <div className="input-footer">
              <span>Try full articles for stronger confidence.</span>
              <span className="shortcut">Ctrl/Cmd + Enter to analyze</span>
            </div>

            <button
              onClick={analyze}
              disabled={loading || !text.trim()}
              className="btn-analyze"
            >
              {loading ? '⏳ Analyzing...' : '+ Analyze'}
            </button>

            <button onClick={() => setShowTestModal(true)} className="btn-test">
              🧪 Load Test Example
            </button>
          </div>

          {/* Right Panel - Features */}
          {!result ? (
            <div className="features-panel">
              <div className="panel-header">
                <div className="section-label">WHAT YOU GET</div>
              </div>

              <h2>Premium feedback, not just a label</h2>

              <ul className="features-list">
                <li>
                  <span className="dot"></span>
                  <span>Color-coded verdicts with smooth animated transitions</span>
                </li>
                <li>
                  <span className="dot"></span>
                  <span>Confidence score with a visual probability meter</span>
                </li>
                <li>
                  <span className="dot"></span>
                  <span>Optional reason messaging for uncertain or low-quality input</span>
                </li>
              </ul>

              <div className="stats">
                <div className="stat">
                  <div className="stat-number">3</div>
                  <div className="stat-label">Prediction states</div>
                </div>
                <div className="stat">
                  <div className="stat-number">2</div>
                  <div className="stat-label">Signal bars</div>
                </div>
                <div className="stat">
                  <div className="stat-number">1</div>
                  <div className="stat-label">Clean API call</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="result-panel">
             <div
  className={`result-badge ${
    result.prediction === 'Real'
      ? 'real'
      : result.prediction === 'Fake'
      ? 'fake'
      : 'uncertain'
  }`}
>
  {result.prediction === 'Real'
    ? '✓ AUTHENTIC'
    : result.prediction === 'Fake'
    ? '✕ SUSPICIOUS'
    : '⚠️ UNCERTAIN'}
</div>
{result.prediction === 'Uncertain' && (
  <div style={{ marginBottom: '15px', color: '#facc15', fontSize: '13px' }}>
    ⚠️ {result.reason || 'Low confidence prediction'}
  </div>
)}

              <div className="confidence-meter">
                <svg viewBox="0 0 180 180">
                  <circle cx="90" cy="90" r="85" className="bg" />
                  <circle
                    cx="90"
                    cy="90"
                    r="85"
                    className="progress"
                    style={{
                      strokeDasharray: `${result.confidence * 534} 534`
                    }}
                  />
                </svg>
                <div className="confidence-text">
                  <div className="percent">{(result.confidence * 100).toFixed(1)}%</div>
                  <div className="label">Confidence</div>
                </div>
              </div>

              <div className="probability-bars">
                <div className={`bar real ${result.probabilities.Real > result.probabilities.Fake ? 'dominant' : ''}`}>
                  <div className="bar-label">Authentic</div>
                  <div className="bar-value">{(result.probabilities.Real * 100).toFixed(1)}%</div>
                  <div className="bar-bg">
                    <div className="bar-fill" style={{width: `${result.probabilities.Real * 100}%`}}></div>
                  </div>
                </div>
                <div className={`bar fake ${result.probabilities.Fake > result.probabilities.Real ? 'dominant' : ''}`}>
                  <div className="bar-label">Suspicious</div>
                  <div className="bar-value">{(result.probabilities.Fake * 100).toFixed(1)}%</div>
                  <div className="bar-bg">
                    <div className="bar-fill" style={{width: `${result.probabilities.Fake * 100}%`}}></div>
                  </div>
                </div>
              </div>

              <div className="result-meta">
                <div><span>Language:</span> {result.language.toUpperCase()}</div>
                <div><span>Words:</span> {wordCount}</div>
                <div><span>Model:</span> Ensemble</div>
              </div>

              <button onClick={clearAll} className="btn-new">Start New Analysis</button>
            </div>
          )}
        </div>

        {/* Test Modal */}
        {showTestModal && (
          <div className="modal-overlay" onClick={() => setShowTestModal(false)}>
            <div className="modal" onClick={e => e.stopPropagation()}>
              <div className="modal-header">
                <h2>Test Examples</h2>
                <button onClick={() => setShowTestModal(false)} className="close-btn">✕</button>
              </div>
              <div className="test-grid">
                {testExamples[language].map((example, i) => (
                  <button
                    key={i}
                    onClick={() => loadTestExample(example)}
                    className={`test-example ${example.title.includes('Real') ? 'real' : 'fake'}`}
                  >
                    <div className="test-type">{example.title}</div>
                    <div className="test-preview">{example.text.substring(0, 80)}...</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
