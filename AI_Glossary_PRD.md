# Product Requirements Document (PRD) – AI Glossary

## 🎯 Purpose
**Build a lightweight, client-side AI glossary that makes complex AI terms instantly accessible to developers, students, and tech enthusiasts.**

**Core Value:** Search → Find → Understand AI concepts in seconds, not minutes.

---

## 🚀 Prototype Goals (30-60 min build)
- **Primary:** Functional search + term display
- **Secondary:** Basic filtering by category
- **Success:** User can search "neural network" and get a clear definition

---

## 👤 Key User Flows

### Flow 1: Quick Search (Primary)
1. User lands on page → sees search box prominently
2. Types "transformer" → sees filtered results instantly
3. Clicks term → definition expands inline
4. **Result:** Question answered in <10 seconds

### Flow 2: Browse by Category (Secondary)
1. User clicks "Machine Learning" filter
2. Sees curated list of ML terms
3. Scans and clicks interesting terms
4. **Result:** Discovers related concepts organically

---

## 🛠 MVP Feature Breakdown

### Core Features (Must Have - 30 min)
- **Search Box**: Real-time filter as user types
- **Term Cards**: Title + short definition (expandable)
- **Static Data**: 20-30 curated AI terms in JSON/array
- **Responsive Layout**: Works on mobile + desktop

### Enhanced Features (Nice to Have - 15 min)
- **Category Filters**: ML, NLP, Computer Vision, Ethics
- **Highlight Search**: Bold matching text in results
- **Smooth Animations**: Expand/collapse transitions

### Polish Features (If Time Allows - 15 min)
- **Dark Mode Toggle**: Simple CSS class switch
- **URL State**: Shareable links for specific terms
- **Clear All**: Reset search/filters button

---

## 📋 Content Structure
```javascript
// Sample data structure
{
  id: "neural-network",
  term: "Neural Network", 
  definition: "A computing system inspired by biological neural networks...",
  category: "Machine Learning",
  examples: ["Image recognition", "Natural language processing"]
}
```

### Initial Term List (20 terms minimum)
**Machine Learning:** Neural Network, Deep Learning, Supervised Learning, Unsupervised Learning, Reinforcement Learning
**NLP:** Transformer, BERT, GPT, Tokenization, Embedding
**Computer Vision:** CNN, Object Detection, Image Classification, Segmentation
**Ethics/General:** AI Bias, Explainable AI, AGI, Training Data, Model Drift

---

## ⚡ Technical Constraints (Optimized for Speed)

### Requirements
- **No Backend**: Pure client-side (HTML/CSS/JS or React)
- **No External APIs**: All data embedded/imported
- **No Database**: JSON array or JavaScript objects
- **No Authentication**: Public access only

### Recommended Stack
- **React + Vite** (already set up) for fast development
- **CSS Modules or Styled Components** for styling
- **Local Storage** for user preferences (theme, last search)

---

## 🎯 Success Criteria

### Functional Success (MVP Done)
- [ ] User can search and find terms
- [ ] Definitions display clearly
- [ ] Works on mobile and desktop
- [ ] No console errors

### User Experience Success
- [ ] Search results appear instantly (<100ms)
- [ ] Interface is intuitive (no instructions needed)
- [ ] Content is accurate and helpful

### Development Success
- [ ] Built within time constraints
- [ ] Code is clean and maintainable
- [ ] Easy to add new terms

---

## 🔄 Iteration Plan

### Phase 1 (Core - 30 min)
Static layout + hardcoded terms + basic search

### Phase 2 (Enhanced - 15 min) 
Categories + better styling + animations

### Phase 3 (Polish - 15 min)
Dark mode + URL state + performance optimization

**Total Target: 60 minutes from start to deployable prototype**
