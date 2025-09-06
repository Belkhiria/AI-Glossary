import React, { useState, useMemo } from 'react'
import { aiTerms, categories } from './data/aiTerms.js'
import './App.css'

export default function App() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('All')
  const [expandedTerms, setExpandedTerms] = useState(new Set())

  // Filter terms based on search and category
  const filteredTerms = useMemo(() => {
    return aiTerms.filter(term => {
      const matchesSearch = term.term.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           term.definition.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesCategory = selectedCategory === 'All' || term.category === selectedCategory
      return matchesSearch && matchesCategory
    })
  }, [searchQuery, selectedCategory])

  // Toggle term expansion
  const toggleTerm = (termId) => {
    const newExpanded = new Set(expandedTerms)
    if (newExpanded.has(termId)) {
      newExpanded.delete(termId)
    } else {
      newExpanded.add(termId)
    }
    setExpandedTerms(newExpanded)
  }

  // Highlight search text
  const highlightText = (text, query) => {
    if (!query) return text
    const parts = text.split(new RegExp(`(${query})`, 'gi'))
    return parts.map((part, index) => 
      part.toLowerCase() === query.toLowerCase() ? 
        <mark key={index}>{part}</mark> : part
    )
  }

  // Clear all filters
  const clearAll = () => {
    setSearchQuery('')
    setSelectedCategory('All')
    setExpandedTerms(new Set())
  }

  return (
    <main className="app">
      <header className="header">
        <h1>🤖 AI Glossary</h1>
        <p className="subtitle">
          Complex AI terms made simple. Search, explore, and understand AI concepts in seconds.
        </p>
      </header>

      <div className="search-section">
        <div className="search-container">
          <input
            type="text"
            placeholder="Search AI terms... (e.g., 'neural network', 'transformer')"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
            autoFocus
          />
          {(searchQuery || selectedCategory !== 'All') && (
            <button onClick={clearAll} className="clear-button">
              Clear All
            </button>
          )}
        </div>

        <div className="category-filters">
          {categories.map(category => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`category-button ${selectedCategory === category ? 'active' : ''}`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      <div className="results-section">
        {filteredTerms.length === 0 ? (
          <div className="no-results">
            <p>No terms found matching "{searchQuery}"</p>
            <p>Try searching for terms like "neural network", "GPT", or "machine learning"</p>
          </div>
        ) : (
          <>
            <div className="results-header">
              <span className="results-count">
                {filteredTerms.length} term{filteredTerms.length !== 1 ? 's' : ''} found
              </span>
            </div>
            
            <div className="terms-grid">
              {filteredTerms.map(term => (
                <div key={term.id} className="term-card">
                  <div 
                    className="term-header"
                    onClick={() => toggleTerm(term.id)}
                  >
                    <h3 className="term-title">
                      {highlightText(term.term, searchQuery)}
                    </h3>
                    <div className="term-meta">
                      <span className="category-tag">{term.category}</span>
                      <span className={`expand-icon ${expandedTerms.has(term.id) ? 'expanded' : ''}`}>
                        ▼
                      </span>
                    </div>
                  </div>
                  
                  <div className="term-preview">
                    {highlightText(term.definition.substring(0, 100), searchQuery)}...
                  </div>

                  {expandedTerms.has(term.id) && (
                    <div className="term-details">
                      <p className="term-definition">
                        {highlightText(term.definition, searchQuery)}
                      </p>
                      <div className="term-examples">
                        <strong>Examples:</strong>
                        <ul>
                          {term.examples.map((example, index) => (
                            <li key={index}>{example}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <footer className="footer">
        <p>
          Built for developers, students, and tech enthusiasts. 
          <span className="tip"> Tip: Click any term to expand its full definition.</span>
        </p>
      </footer>
    </main>
  )
}
