/**
 * Support Ticket Classifier Dashboard - Controller Script
 * Handles UI bindings, event handlers, and links to the ClassifierEngine
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- App State ---
  let tickets = [];
  let vectorizer = null;
  let model = null;
  let isTrained = false;
  let testSet = [];
  
  // Unique Categories List
  const CATEGORIES = [
    "Technical Issue",
    "Billing Issue",
    "Account Issue",
    "Feature Request",
    "General Inquiry"
  ];

  // --- UI Elements ---
  const modelStatusIndicator = document.getElementById("model-status-indicator");
  const modelStatusText = document.getElementById("model-status-text");
  
  const ticketInput = document.getElementById("ticket-input");
  const classifyBtn = document.getElementById("classify-btn");
  const clearInputBtn = document.getElementById("clear-input-btn");
  
  const predictionResults = document.getElementById("prediction-results");
  const predCategory = document.getElementById("pred-category");
  const predPriority = document.getElementById("pred-priority");
  const probChart = document.getElementById("prob-chart");
  const preprocRaw = document.getElementById("preproc-raw");
  const preprocClean = document.getElementById("preproc-clean");
  const preprocKeywords = document.getElementById("preproc-keywords");
  
  const trainModelBtn = document.getElementById("train-model-btn");
  const trainVocabLabel = document.getElementById("train-vocab-label");
  
  const statTotalTickets = document.getElementById("stat-total-tickets");
  const statAccuracy = document.getElementById("stat-accuracy");
  const statTestSamples = document.getElementById("stat-test-samples");
  
  const evaluationPanel = document.getElementById("evaluation-panel");
  const metricsTableBody = document.getElementById("metrics-table-body");
  const confusionMatrixGrid = document.getElementById("confusion-matrix-grid");
  
  const searchInput = document.getElementById("search-input");
  const filterCategory = document.getElementById("filter-category");
  const filterPriority = document.getElementById("filter-priority");
  const datasetTableBody = document.getElementById("dataset-table-body");
  const datasetCountLabel = document.getElementById("dataset-count-label");
  const resetDatasetBtn = document.getElementById("reset-dataset-btn");
  const addTicketBtn = document.getElementById("add-ticket-btn");
  
  // Modal Elements
  const ticketModal = document.getElementById("ticket-modal");
  const closeModalBtn = document.getElementById("close-modal-btn");
  const cancelModalBtn = document.getElementById("cancel-modal-btn");
  const saveTicketBtn = document.getElementById("save-ticket-btn");
  const newTicketText = document.getElementById("new-ticket-text");
  const newTicketCategory = document.getElementById("new-ticket-category");
  const newTicketPriority = document.getElementById("new-ticket-priority");

  // --- Initializer ---
  function init() {
    loadDataset();
    bindEvents();
    
    // Automatically train model on startup so dashboard works out of the box
    trainModel();
  }

  // --- Load Dataset ---
  function loadDataset() {
    const stored = localStorage.getItem("support_tickets");
    if (stored) {
      try {
        tickets = JSON.parse(stored);
      } catch (e) {
        console.error("Error parsing stored tickets, resetting...", e);
        tickets = [...INITIAL_TICKETS];
        saveDatasetToStorage();
      }
    } else {
      tickets = [...INITIAL_TICKETS];
      saveDatasetToStorage();
    }
    
    updateStats();
    renderDatasetExplorer();
  }

  function saveDatasetToStorage() {
    localStorage.setItem("support_tickets", JSON.stringify(tickets));
  }

  function updateStats() {
    statTotalTickets.textContent = tickets.length;
  }

  // --- Bind UI Events ---
  function bindEvents() {
    // Classifier simulator
    classifyBtn.addEventListener("click", classifyCurrentInput);
    clearInputBtn.addEventListener("click", () => {
      ticketInput.value = "";
      predictionResults.style.display = "none";
    });
    
    // Classify dynamically with a debounce as the user types
    let debounceTimer;
    ticketInput.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        if (ticketInput.value.trim().length > 3) {
          classifyCurrentInput();
        } else {
          predictionResults.style.display = "none";
        }
      }, 300);
    });

    // Train Model
    trainModelBtn.addEventListener("click", () => {
      trainModel();
      // Simple visual trigger
      trainModelBtn.classList.add("btn-secondary");
      trainModelBtn.classList.remove("btn-primary");
      setTimeout(() => {
        trainModelBtn.classList.add("btn-primary");
        trainModelBtn.classList.remove("btn-secondary");
      }, 300);
    });

    // Search and filters
    searchInput.addEventListener("input", renderDatasetExplorer);
    filterCategory.addEventListener("change", renderDatasetExplorer);
    filterPriority.addEventListener("change", renderDatasetExplorer);
    
    // Reset dataset
    resetDatasetBtn.addEventListener("click", () => {
      if (confirm("Are you sure you want to restore the default dataset of 150 tickets? Your additions will be deleted.")) {
        localStorage.removeItem("support_tickets");
        loadDataset();
        trainModel();
        // Clear classifier simulation inputs
        ticketInput.value = "";
        predictionResults.style.display = "none";
      }
    });

    // Modal Triggers
    addTicketBtn.addEventListener("click", () => {
      newTicketText.value = "";
      ticketModal.classList.add("active");
      newTicketText.focus();
    });

    const closeModal = () => {
      ticketModal.classList.remove("active");
    };
    
    closeModalBtn.addEventListener("click", closeModal);
    cancelModalBtn.addEventListener("click", closeModal);
    
    // Save new ticket
    saveTicketBtn.addEventListener("click", () => {
      const text = newTicketText.value.trim();
      const category = newTicketCategory.value;
      const priority = newTicketPriority.value;

      if (!text) {
        alert("Please enter ticket content description.");
        newTicketText.focus();
        return;
      }

      tickets.unshift({ ticket: text, category, priority });
      saveDatasetToStorage();
      updateStats();
      renderDatasetExplorer();
      closeModal();
      
      // Auto-retrain on new ticket add so model fits updated dataset
      trainModel();
    });
  }

  // --- Train Model Loop ---
  function trainModel() {
    if (tickets.length < 10) {
      alert("Dataset is too small to train a model. Please add more tickets.");
      return;
    }

    // 1. Perform Train/Test Split (80% Train, 20% Test)
    const splits = ClassifierEngine.trainTestSplit(tickets, 0.2);
    testSet = splits.test;

    // 2. Preprocess ticket strings
    const trainCleaned = splits.train.map(t => ClassifierEngine.cleanText(t.ticket));
    const testCleaned = splits.test.map(t => ClassifierEngine.cleanText(t.ticket));
    
    const trainLabels = splits.train.map(t => t.category);
    const testLabels = splits.test.map(t => t.category);

    // 3. Fit TF-IDF Vectorizer
    vectorizer = new ClassifierEngine.TfidfVectorizer();
    vectorizer.fit(trainCleaned);

    // 4. Transform documents to TF-IDF vectors
    const X_train = vectorizer.transform(trainCleaned);
    const X_test = vectorizer.transform(testCleaned);

    // 5. Fit Multinomial Naive Bayes model
    model = new ClassifierEngine.MultinomialNB(1.0); // Laplace alpha = 1.0
    model.fit(X_train, trainLabels);

    // 6. Predict on Test Set
    const predictions = X_test.map(vec => model.predict(vec));

    // 7. Calculate evaluation metrics
    const metrics = ClassifierEngine.calculateMetrics(testLabels, predictions, CATEGORIES);

    // 8. Render Results in Dashboard
    isTrained = true;
    modelStatusIndicator.classList.add("ready");
    modelStatusText.textContent = "Model Trained & Active";

    trainVocabLabel.textContent = `Vocabulary: ${vectorizer.vocabList.length} features`;
    statAccuracy.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
    statTestSamples.textContent = splits.test.length;

    renderPerformanceMetrics(metrics.classReports);
    renderConfusionMatrix(metrics.confusionMatrix, testLabels);

    evaluationPanel.style.display = "block";
  }

  // --- Classify Live Input ---
  function classifyCurrentInput() {
    if (!isTrained || !model || !vectorizer) {
      alert("Model is not trained. Please train the model first.");
      return;
    }

    const rawText = ticketInput.value.trim();
    if (!rawText) {
      predictionResults.style.display = "none";
      return;
    }

    // 1. Preprocess
    const cleaned = ClassifierEngine.cleanText(rawText);

    // 2. Vectorize
    const vectors = vectorizer.transform([cleaned]);
    const vector = vectors[0];

    // 3. Classify (Category & Probabilities)
    const category = model.predict(vector);
    const probabilities = model.predictProba(vector);

    // 4. Assign Priority
    const priority = ClassifierEngine.assignPriority(rawText);

    // 5. Display output badges
    predCategory.textContent = category;
    // Set text class based on category
    predCategory.className = "pred-badge-value " + getCategoryColorClass(category);
    
    predPriority.textContent = priority;
    predPriority.className = "pred-badge-value " + getPriorityColorClass(priority);

    // 6. Draw category probabilities bar chart
    probChart.innerHTML = "";
    CATEGORIES.forEach(cat => {
      const prob = probabilities[cat] || 0;
      const pct = (prob * 100).toFixed(1);
      
      const probRow = document.createElement("div");
      probRow.className = "prob-row";
      
      const labelClass = cat === category ? "prob-label " + getCategoryColorClass(cat) : "prob-label";
      const fontStyle = cat === category ? "font-weight: 600;" : "";
      
      probRow.innerHTML = `
        <span class="${labelClass}" style="${fontStyle}" title="${cat}">${cat}</span>
        <div class="prob-track">
          <div class="prob-bar ${getProgressBarClass(cat)}" style="width: ${pct}%"></div>
        </div>
        <span class="prob-pct" style="${fontStyle}">${pct}%</span>
      `;
      probChart.appendChild(probRow);
    });

    // 7. Render NLP processing step summaries
    preprocRaw.textContent = `"${rawText}"`;
    preprocClean.textContent = cleaned ? `[ ${cleaned.split(" ").join(", ")} ]` : "[ empty after stop-words removal ]";
    
    // Show priority-associated keywords found in text
    const matchedKeywords = getMatchedPriorityKeywords(rawText);
    preprocKeywords.textContent = matchedKeywords.length > 0 
      ? `Keywords: ${matchedKeywords.join(", ")}` 
      : "No priority keywords matched. Defaulting to Low.";

    predictionResults.style.display = "block";
  }

  // --- Render Dataset Table Explorer ---
  function renderDatasetExplorer() {
    const query = searchInput.value.toLowerCase().trim();
    const catFilter = filterCategory.value;
    const priFilter = filterPriority.value;

    // Filter tickets array
    const filtered = tickets.filter(t => {
      const matchesSearch = t.ticket.toLowerCase().includes(query);
      const matchesCategory = catFilter === "" || t.category === catFilter;
      const matchesPriority = priFilter === "" || t.priority === priFilter;
      return matchesSearch && matchesCategory && matchesPriority;
    });

    datasetCountLabel.textContent = `Showing ${filtered.length} of ${tickets.length} tickets`;
    
    // Clear existing table content
    datasetTableBody.innerHTML = "";

    if (filtered.length === 0) {
      datasetTableBody.innerHTML = `
        <tr>
          <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 30px;">
            No tickets found matching filters.
          </td>
        </tr>
      `;
      return;
    }

    // Render table rows
    filtered.forEach((item, index) => {
      const tr = document.createElement("tr");
      
      // Get the absolute index in the master tickets array for deleting
      const originalIndex = tickets.indexOf(item);

      tr.innerHTML = `
        <td style="font-weight: 400; color: var(--text-main); word-break: break-word;">${escapeHtml(item.ticket)}</td>
        <td><span class="badge-pill ${getPillCategoryClass(item.category)}">${item.category}</span></td>
        <td><span class="badge-pill ${getPillPriorityClass(item.priority)}">${item.priority}</span></td>
        <td>
          <button class="action-icon-btn delete-ticket-btn" data-index="${originalIndex}" title="Delete Ticket">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
              <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
            </svg>
          </button>
        </td>
      `;
      datasetTableBody.appendChild(tr);
    });

    // Hook delete buttons
    const deleteButtons = datasetTableBody.querySelectorAll(".delete-ticket-btn");
    deleteButtons.forEach(btn => {
      btn.addEventListener("click", (e) => {
        const index = parseInt(btn.getAttribute("data-index"));
        if (confirm("Are you sure you want to delete this ticket from the dataset?")) {
          tickets.splice(index, 1);
          saveDatasetToStorage();
          updateStats();
          renderDatasetExplorer();
          
          // Re-train model to reflect deleted ticket features
          trainModel();
        }
      });
    });
  }

  // --- Render Class Performance Report Table ---
  function renderPerformanceMetrics(classReports) {
    metricsTableBody.innerHTML = "";
    
    CATEGORIES.forEach(cat => {
      const report = classReports[cat] || { precision: 0, recall: 0, f1: 0, support: 0 };
      
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 500;" class="${getCategoryColorClass(cat)}">${cat}</td>
        <td>${(report.precision * 100).toFixed(1)}%</td>
        <td>${(report.recall * 100).toFixed(1)}%</td>
        <td style="font-weight: 600;">${(report.f1 * 100).toFixed(1)}%</td>
        <td style="color: var(--text-muted);">${report.support}</td>
      `;
      metricsTableBody.appendChild(tr);
    });
  }

  // --- Render Confusion Matrix Visual Grid ---
  function renderConfusionMatrix(matrix, testLabels) {
    confusionMatrixGrid.innerHTML = "";
    
    // Determine the maximum value in the matrix for color density scaling
    let maxVal = 1;
    CATEGORIES.forEach(tClass => {
      CATEGORIES.forEach(pClass => {
        const val = matrix[tClass][pClass] || 0;
        if (val > maxVal) maxVal = val;
      });
    });

    // 1. Column Header Corner (Empty space)
    const cornerCell = document.createElement("div");
    cornerCell.className = "matrix-cell matrix-header";
    cornerCell.innerHTML = `<span style="font-size: 0.65rem; text-transform: uppercase;">True \\ Pred</span>`;
    confusionMatrixGrid.appendChild(cornerCell);

    // 2. Column Header Labels (Predicted Categories)
    CATEGORIES.forEach(pClass => {
      const headerCell = document.createElement("div");
      headerCell.className = "matrix-cell matrix-header matrix-header-col";
      headerCell.textContent = getCategoryAbbreviation(pClass);
      headerCell.title = `Predicted: ${pClass}`;
      confusionMatrixGrid.appendChild(headerCell);
    });

    // 3. Grid Rows
    CATEGORIES.forEach(tClass => {
      // Row Label (True Category)
      const labelCell = document.createElement("div");
      labelCell.className = "matrix-cell matrix-header matrix-header-row";
      labelCell.textContent = getCategoryAbbreviation(tClass);
      labelCell.title = `Actual: ${tClass}`;
      confusionMatrixGrid.appendChild(labelCell);

      // Data Cells for this True Category
      CATEGORIES.forEach(pClass => {
        const val = matrix[tClass][pClass] || 0;
        const cell = document.createElement("div");
        
        // Compute density class
        let weightClass = "cell-weight-0";
        if (val > 0) {
          const ratio = val / maxVal;
          if (ratio <= 0.25) weightClass = "cell-weight-1";
          else if (ratio <= 0.5) weightClass = "cell-weight-2";
          else if (ratio <= 0.75) weightClass = "cell-weight-3";
          else weightClass = "cell-weight-4";
        }
        
        cell.className = `matrix-cell ${weightClass}`;
        cell.innerHTML = `<span class="matrix-val">${val}</span>`;
        cell.title = `Actual: ${tClass}\nPredicted: ${pClass}\nCount: ${val}`;
        
        confusionMatrixGrid.appendChild(cell);
      });
    });
  }

  // --- Utility Functions ---
  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function getCategoryAbbreviation(cat) {
    switch (cat) {
      case "Technical Issue": return "Tech";
      case "Billing Issue": return "Billing";
      case "Account Issue": return "Account";
      case "Feature Request": return "Feature";
      case "General Inquiry": return "Inquiry";
      default: return cat;
    }
  }

  function getCategoryColorClass(cat) {
    switch (cat) {
      case "Technical Issue": return "cat-tech";
      case "Billing Issue": return "cat-billing";
      case "Account Issue": return "cat-account";
      case "Feature Request": return "cat-feature";
      case "General Inquiry": return "cat-inquiry";
      default: return "";
    }
  }

  function getPriorityColorClass(pri) {
    switch (pri) {
      case "High": return "pri-high";
      case "Medium": return "pri-medium";
      case "Low": return "pri-low";
      default: return "";
    }
  }

  function getPillCategoryClass(cat) {
    switch (cat) {
      case "Technical Issue": return "pill-tech";
      case "Billing Issue": return "pill-billing";
      case "Account Issue": return "pill-account";
      case "Feature Request": return "pill-feature";
      case "General Inquiry": return "pill-inquiry";
      default: return "";
    }
  }

  function getPillPriorityClass(pri) {
    switch (pri) {
      case "High": return "pill-high";
      case "Medium": return "pill-medium";
      case "Low": return "pill-low";
      default: return "";
    }
  }

  function getProgressBarClass(cat) {
    switch (cat) {
      case "Technical Issue": return "prob-bar-tech";
      case "Billing Issue": return "prob-bar-billing";
      case "Account Issue": return "prob-bar-account";
      case "Feature Request": return "prob-bar-feature";
      case "General Inquiry": return "prob-bar-inquiry";
      default: return "";
    }
  }

  function getMatchedPriorityKeywords(text) {
    const highKeywords = [
      "error", "failed", "refund", "payment", "server down", "crash",
      "locked out", "suspicious", "declined", "unauthorized", "deadlock",
      "timeout", "freeze", "double charge", "security", "breach", "urgent",
      "urgently", "fatal", "broken", "loss", "money", "charged twice", "charge twice", "blocked"
    ];

    const mediumKeywords = [
      "login", "password", "account", "slow", "verify", "link expired",
      "update", "slack", "sync", "loading slowly", "reset link", "integrate",
      "recap", "how to delete", "cannot change", "promo code", "receipt", "invoice"
    ];

    const cleanedText = text.toLowerCase();
    const matched = [];

    highKeywords.forEach(kw => {
      if (cleanedText.includes(kw) && !matched.includes(`${kw} (High)`)) {
        matched.push(`${kw} (High)`);
      }
    });

    mediumKeywords.forEach(kw => {
      if (cleanedText.includes(kw) && !matched.includes(`${kw} (Med)`)) {
        // Only push if not already marked High
        const simpleKw = kw;
        if (!highKeywords.some(hk => simpleKw.includes(hk))) {
          matched.push(`${kw} (Medium)`);
        }
      }
    });

    return matched;
  }

  // Launch Dashboard
  init();
});
