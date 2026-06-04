/**
 * Support Ticket Classifier Engine
 * Pure Vanilla JavaScript implementation of TF-IDF Vectorizer and Multinomial Naive Bayes
 */

// Custom stop words list
const STOP_WORDS = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent",
  "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
  "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont",
  "down", "during", "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have",
  "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him",
  "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt",
  "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not",
  "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
  "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some",
  "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there",
  "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to",
  "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "werent",
  "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom",
  "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve",
  "your", "yours", "yourself", "yourselves"
]);

/**
 * Preprocess text: lowercase, remove punctuation, filter stop words
 */
function cleanText(text) {
  if (!text) return "";
  // Lowercase
  let cleaned = text.toLowerCase();
  // Remove non-alphabetic characters (keep spaces)
  cleaned = cleaned.replace(/[^a-z\s]/g, "");
  // Tokenize by whitespace
  const tokens = cleaned.split(/\s+/);
  // Filter out stop words and empty tokens
  const filtered = tokens.filter(word => word.length > 0 && !STOP_WORDS.has(word));
  return filtered.join(" ");
}

/**
 * TF-IDF Vectorizer
 */
class TfidfVectorizer {
  constructor() {
    this.vocabulary = {}; // word -> index
    this.idf = {};        // word -> idf value
    this.vocabList = [];  // index -> word
    this.isFitted = false;
  }

  /**
   * Fit vocabulary and calculate IDF from a corpus of preprocessed text documents
   * @param {string[]} cleanDocuments Array of cleaned document strings
   */
  fit(cleanDocuments) {
    const nSamples = cleanDocuments.length;
    const documentFrequencies = {};
    const tempVocab = new Set();

    // 1. Find document frequency (df) for each word
    cleanDocuments.forEach(doc => {
      const tokens = doc.split(/\s+/).filter(t => t.length > 0);
      const uniqueTokensInDoc = new Set(tokens);
      uniqueTokensInDoc.forEach(token => {
        tempVocab.add(token);
        documentFrequencies[token] = (documentFrequencies[token] || 0) + 1;
      });
    });

    // 2. Build vocabulary mapping
    this.vocabList = Array.from(tempVocab).sort();
    this.vocabulary = {};
    this.vocabList.forEach((word, idx) => {
      this.vocabulary[word] = idx;
    });

    // 3. Compute IDF with scikit-learn smoothing formula:
    // idf(t) = log((1 + nSamples) / (1 + df(t))) + 1
    this.idf = {};
    this.vocabList.forEach(word => {
      const df = documentFrequencies[word] || 0;
      this.idf[word] = Math.log((1 + nSamples) / (1 + df)) + 1;
    });

    this.isFitted = true;
    return this;
  }

  /**
   * Transform documents into L2-normalized TF-IDF vectors
   * @param {string[]} cleanDocuments Array of cleaned document strings
   * @returns {number[][]} Array of vectors (shape: [num_docs, vocab_size])
   */
  transform(cleanDocuments) {
    if (!this.isFitted) {
      throw new Error("TfidfVectorizer must be fitted before transforming data.");
    }

    const vocabSize = this.vocabList.length;

    return cleanDocuments.map(doc => {
      const vector = new Array(vocabSize).fill(0);
      const tokens = doc.split(/\s+/).filter(t => t.length > 0);
      if (tokens.length === 0) return vector;

      // Count term frequencies (TF) in this document
      const tfCounts = {};
      tokens.forEach(token => {
        tfCounts[token] = (tfCounts[token] || 0) + 1;
      });

      // Calculate raw TF-IDF for words in vocabulary
      let sumSquares = 0;
      for (const word in tfCounts) {
        if (word in this.vocabulary) {
          const idx = this.vocabulary[word];
          const tf = tfCounts[word];
          const idf = this.idf[word];
          vector[idx] = tf * idf;
          sumSquares += vector[idx] * vector[idx];
        }
      }

      // L2 Normalization: divide vector by its Euclidean norm
      const l2Norm = Math.sqrt(sumSquares);
      if (l2Norm > 0) {
        for (let i = 0; i < vocabSize; i++) {
          vector[i] = vector[i] / l2Norm;
        }
      }

      return vector;
    });
  }

  /**
   * Fit and transform in one step
   */
  fitTransform(cleanDocuments) {
    return this.fit(cleanDocuments).transform(cleanDocuments);
  }
}

/**
 * Multinomial Naive Bayes Classifier
 */
class MultinomialNB {
  constructor(alpha = 1.0) {
    this.alpha = alpha; // Laplace smoothing parameter
    this.classes = [];
    this.classLogPriors = {};
    this.featureLogProbabilities = {}; // class -> array of log probabilities of length vocab_size
    this.isTrained = false;
  }

  /**
   * Train the classifier on TF-IDF vectors and labels
   * @param {number[][]} X TF-IDF vectors, shape [nSamples, nFeatures]
   * @param {string[]} y Target class labels of length nSamples
   */
  fit(X, y) {
    const nSamples = X.length;
    if (nSamples === 0) return;
    const nFeatures = X[0].length;

    // 1. Identify distinct classes
    this.classes = Array.from(new Set(y));
    const nClasses = this.classes.length;

    // Count class frequencies
    const classCounts = {};
    y.forEach(label => {
      classCounts[label] = (classCounts[label] || 0) + 1;
    });

    // 2. Compute Class Log Priors
    this.classLogPriors = {};
    this.classes.forEach(cls => {
      this.classLogPriors[cls] = Math.log(classCounts[cls] / nSamples);
    });

    // 3. Compute Feature Log Probabilities with Laplace smoothing
    this.featureLogProbabilities = {};
    this.classes.forEach(cls => {
      // Sum the TF-IDF vectors for documents belonging to this class
      const featureSums = new Array(nFeatures).fill(0);
      let totalFeatureSum = 0;

      for (let i = 0; i < nSamples; i++) {
        if (y[i] === cls) {
          for (let j = 0; j < nFeatures; j++) {
            featureSums[j] += X[i][j];
            totalFeatureSum += X[i][j];
          }
        }
      }

      // Feature probabilities P(x_i | c) = (sum(x_ij for j in class c) + alpha) / (total_sum(class c) + alpha * nFeatures)
      const denominator = totalFeatureSum + this.alpha * nFeatures;
      const logProbs = new Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        const numerator = featureSums[j] + this.alpha;
        logProbs[j] = Math.log(numerator / denominator);
      }

      this.featureLogProbabilities[cls] = logProbs;
    });

    this.isTrained = true;
    return this;
  }

  /**
   * Predict class probabilities for a single TF-IDF vector
   * @param {number[]} vector TF-IDF vector of length nFeatures
   * @returns {Object} Mapping from class label to probability
   */
  predictProba(vector) {
    if (!this.isTrained) {
      throw new Error("Model has not been trained yet.");
    }

    const logPosteriors = {};
    let maxLogPosterior = -Infinity;

    // Calculate P(c) * PROD( P(x_i | c)^w_i ) in log space:
    // log(P(c)) + SUM( w_i * log(P(x_i | c)) )
    this.classes.forEach(cls => {
      let score = this.classLogPriors[cls];
      const logLikelihoods = this.featureLogProbabilities[cls];

      for (let i = 0; i < vector.length; i++) {
        if (vector[i] > 0) {
          score += vector[i] * logLikelihoods[i];
        }
      }

      logPosteriors[cls] = score;
      if (score > maxLogPosterior) {
        maxLogPosterior = score;
      }
    });

    // Exponentials subtraction trick to prevent overflow/underflow
    const exps = {};
    let sumExps = 0;
    this.classes.forEach(cls => {
      const expVal = Math.exp(logPosteriors[cls] - maxLogPosterior);
      exps[cls] = expVal;
      sumExps += expVal;
    });

    // Normalization to get final probabilities
    const probabilities = {};
    this.classes.forEach(cls => {
      probabilities[cls] = sumExps > 0 ? exps[cls] / sumExps : 1 / this.classes.length;
    });

    return probabilities;
  }

  /**
   * Predict the highest probability class label for a single TF-IDF vector
   * @param {number[]} vector TF-IDF vector of length nFeatures
   * @returns {string} The predicted class name
   */
  predict(vector) {
    const probas = this.predictProba(vector);
    let bestClass = "";
    let maxProba = -1;

    for (const cls in probas) {
      if (probas[cls] > maxProba) {
        maxProba = probas[cls];
        bestClass = cls;
      }
    }

    return bestClass;
  }
}

/**
 * Keyword-based priority assignment logic
 * @param {string} ticketText Uncleaned raw support ticket text
 * @returns {string} Priority: "High", "Medium", or "Low"
 */
function assignPriority(ticketText) {
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

  const ticket = ticketText.toLowerCase();

  // Search for phrases/keywords
  if (highKeywords.some(word => ticket.includes(word))) {
    return "High";
  } else if (mediumKeywords.some(word => ticket.includes(word))) {
    return "Medium";
  } else {
    return "Low";
  }
}

/**
 * Splits dataset into random training and testing sets
 * @param {Object[]} data Full dataset array
 * @param {number} testSize Proportion of test data (e.g. 0.2)
 * @returns {{train: Object[], test: Object[]}} Train and test splits
 */
function trainTestSplit(data, testSize = 0.2) {
  // Use a predictable seed layout or shuffle
  // Copy array
  const shuffled = [...data];
  // Fisher-Yates Shuffle with a fixed random_state if we want repeatable results
  // For the client-side dashboard, we can use standard Math.random or seeded random.
  // Standard Math.random works great to demonstrate live changes.
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const splitIndex = Math.floor(shuffled.length * (1 - testSize));
  const train = shuffled.slice(0, splitIndex);
  const test = shuffled.slice(splitIndex);

  return { train, test };
}

/**
 * Computes model performance evaluation metrics
 * @param {string[]} trueLabels Actual labels
 * @param {string[]} predictedLabels Predicted labels
 * @param {string[]} uniqueClasses List of unique categories
 */
function calculateMetrics(trueLabels, predictedLabels, uniqueClasses) {
  let correct = 0;
  const total = trueLabels.length;

  // Accuracy
  for (let i = 0; i < total; i++) {
    if (trueLabels[i] === predictedLabels[i]) {
      correct++;
    }
  }
  const accuracy = total > 0 ? correct / total : 0;

  // Initialize confusion matrix
  // matrix[trueLabel][predictedLabel] = count
  const confusionMatrix = {};
  uniqueClasses.forEach(tClass => {
    confusionMatrix[tClass] = {};
    uniqueClasses.forEach(pClass => {
      confusionMatrix[tClass][pClass] = 0;
    });
  });

  // Fill confusion matrix
  for (let i = 0; i < total; i++) {
    const tLabel = trueLabels[i];
    const pLabel = predictedLabels[i];
    if (confusionMatrix[tLabel] && confusionMatrix[tLabel][pLabel] !== undefined) {
      confusionMatrix[tLabel][pLabel]++;
    }
  }

  // Calculate Precision, Recall, F1 for each class
  const classReports = {};
  uniqueClasses.forEach(cls => {
    let tp = 0; // True Positive
    let fp = 0; // False Positive
    let fn = 0; // False Negative

    for (let i = 0; i < total; i++) {
      if (trueLabels[i] === cls && predictedLabels[i] === cls) {
        tp++;
      } else if (trueLabels[i] !== cls && predictedLabels[i] === cls) {
        fp++;
      } else if (trueLabels[i] === cls && predictedLabels[i] !== cls) {
        fn++;
      }
    }

    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    const support = trueLabels.filter(l => l === cls).length;

    classReports[cls] = { precision, recall, f1, support };
  });

  return {
    accuracy,
    confusionMatrix,
    classReports
  };
}

// Export modules globally for UI script access
window.ClassifierEngine = {
  cleanText,
  TfidfVectorizer,
  MultinomialNB,
  assignPriority,
  trainTestSplit,
  calculateMetrics
};
