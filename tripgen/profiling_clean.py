import json
import re
import logging
import difflib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

# Import necessary NLP libraries with better error handling
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.util import ngrams
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from langdetect import detect, DetectorFactory
    
    # Set seed for consistent language detection
    DetectorFactory.seed = 0
    
except ImportError as e:
    missing_libs = []
    if 'nltk' in str(e):
        missing_libs.append('nltk')
    if 'sklearn' in str(e):
        missing_libs.append('scikit-learn')
    if 'numpy' in str(e):
        missing_libs.append('numpy')
    if 'langdetect' in str(e):
        missing_libs.append('langdetect')
    
    print(f"Missing required libraries: {', '.join(missing_libs)}")
    print("Please install with: pip install scikit-learn nltk numpy langdetect")
    print("\nAlso run these commands once to download NLTK data:")
    print("import nltk")
    print("nltk.download('punkt')")
    print("nltk.download('stopwords')")
    print("nltk.download('wordnet')")
    print("nltk.download('omw-1.4')")
    print("nltk.download('vader_lexicon')")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProfilerConfig:
    """Configuration class for the profiler"""
    num_city_tags: int = 5
    min_reviews_per_place: int = 2
    min_review_length: int = 50
    sentiment_threshold: float = 0.1
    tfidf_max_features: int = 1000
    cluster_count: int = 8
    min_term_frequency: float = 0.01
    max_term_frequency: float = 0.5
    enable_clustering: bool = True
    enable_sentiment_analysis: bool = True
    enable_language_detection: bool = True
    fallback_tags: List[str] = None
    
    def __post_init__(self):
        if self.fallback_tags is None:
            self.fallback_tags = ['Historic', 'Cultural', 'Tourist', 'Scenic', 'Memorable']

class RobustCityProfiler:
    """
    An enhanced, robust system to profile cities with sophisticated text analysis,
    sentiment consideration, error handling, and nuanced keyword extraction.
    """

    def __init__(self, config: ProfilerConfig = None):
        self.config = config or ProfilerConfig()
        self.lemmatizer = None
        self.sentiment_analyzer = None
        self.stopwords = set()
        
        self._initialize_nlp_components()
        
        # Enhanced tourism categories with weights and context
        self.tourism_concepts = self._load_base_tourism_concepts()
        self.negative_indicators = self._load_negative_indicators()
        
        # Initialize TF-IDF vectorizer for semantic analysis
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            min_df=self.config.min_term_frequency,
            max_df=self.config.max_term_frequency,
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            stop_words='english'
        )
        
        # Statistics tracking
        self.stats = {
            'total_reviews': 0,
            'processed_reviews': 0,
            'languages_detected': defaultdict(int),
            'sentiment_distribution': defaultdict(int),
            'errors_encountered': []
        }

    def _initialize_nlp_components(self):
        """Initialize NLP components with error handling"""
        try:
            # Download required NLTK data if not present
            required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
            if self.config.enable_sentiment_analysis:
                required_nltk_data.append('vader_lexicon')
            
            for data in required_nltk_data:
                try:
                    nltk.data.find(f'tokenizers/{data}')
                except LookupError:
                    logger.info(f"Downloading NLTK data: {data}")
                    nltk.download(data, quiet=True)
            
            # Initialize components
            self.lemmatizer = WordNetLemmatizer()
            
            if self.config.enable_sentiment_analysis:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Enhanced stopwords
            basic_stopwords = set(stopwords.words('english'))
            additional_stopwords = {
                'place', 'time', 'people', 'good', 'great', 'nice', 'beautiful', 'amazing', 
                'best', 'visit', 'see', 'go', 'come', 'much', 'one', 'also', 'really', 
                'very', 'would', 'could', 'should', 'make', 'take', 'get', 'day', 'way',
                'lot', 'many', 'well', 'back', 'around', 'inside', 'outside', 'top', 
                'area', 'side', 'part', 'thing', 'something', 'everything', 'anything',
                'quite', 'pretty', 'definitely', 'probably', 'maybe', 'perhaps'
            }
            self.stopwords = basic_stopwords.union(additional_stopwords)
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP components: {e}")
            raise
        
    def _load_base_tourism_concepts(self) -> Dict[str, Dict[str, Any]]:
        """Load base tourism concept seeds for dynamic expansion"""
        return {
            'architecture': {
                'core_concepts': ['architecture', 'building', 'design', 'structure'],
                'semantic_clusters': ['visual_appeal', 'historical_significance', 'craftsmanship'],
                'weight': 1.0,
                'discovery_patterns': [r'\b\w*architect\w*\b', r'\b\w*design\w*\b', r'\b\w*built\w*\b']
            },
            'heritage': {
                'core_concepts': ['heritage', 'history', 'culture', 'tradition'],
                'semantic_clusters': ['temporal_significance', 'cultural_value', 'authenticity'],
                'weight': 1.0,
                'discovery_patterns': [r'\b\w*histor\w*\b', r'\b\w*cultur\w*\b', r'\b\w*tradition\w*\b']
            },
            'experience': {
                'core_concepts': ['experience', 'visit', 'enjoy', 'explore'],
                'semantic_clusters': ['engagement', 'satisfaction', 'memorable_moments'],
                'weight': 1.1,
                'discovery_patterns': [r'\b\w*experienc\w*\b', r'\b\w*enjoy\w*\b', r'\b\w*visit\w*\b']
            },
            'nature': {
                'core_concepts': ['nature', 'natural', 'scenic', 'landscape'],
                'semantic_clusters': ['environmental_beauty', 'outdoor_activities', 'tranquility'],
                'weight': 1.2,
                'discovery_patterns': [r'\b\w*scenic\w*\b', r'\b\w*natural\w*\b', r'\b\w*landscape\w*\b']
            },
            'activity': {
                'core_concepts': ['activity', 'adventure', 'sports', 'recreation'],
                'semantic_clusters': ['physical_engagement', 'thrill_seeking', 'skill_based'],
                'weight': 1.1,
                'discovery_patterns': [r'\b\w*activit\w*\b', r'\b\w*adventure\w*\b', r'\b\w*sport\w*\b']
            },
            'social': {
                'core_concepts': ['social', 'people', 'community', 'interaction'],
                'semantic_clusters': ['human_connection', 'cultural_exchange', 'group_dynamics'],
                'weight': 0.9,
                'discovery_patterns': [r'\b\w*social\w*\b', r'\b\w*people\w*\b', r'\b\w*crowd\w*\b']
            }
        }

    def _load_negative_indicators(self) -> List[str]:
        """Load words that indicate negative experiences"""
        return [
            'disappointing', 'overpriced', 'crowded', 'dirty', 'unsafe', 'scam', 'tourist-trap',
            'avoid', 'waste', 'terrible', 'awful', 'horrible', 'worst', 'disgusting',
            'boring', 'commercial', 'fake', 'poor', 'bad', 'worst', 'never', 'don\'t',
            'closed', 'broken', 'damaged', 'expensive', 'rude', 'unfriendly'
        ]

    def _detect_language(self, text: str) -> str:
        """Detect language of the text with error handling"""
        if not self.config.enable_language_detection:
            return 'en'
        
        try:
            if len(text.strip()) < 10:
                return 'en'  # Default for very short texts
            
            detected = detect(text)
            self.stats['languages_detected'][detected] += 1
            return detected
        except:
            return 'en'  # Default fallback

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text with error handling"""
        if not self.config.enable_sentiment_analysis or not self.sentiment_analyzer:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Classify sentiment
            if scores['compound'] >= self.config.sentiment_threshold:
                sentiment_class = 'positive'
            elif scores['compound'] <= -self.config.sentiment_threshold:
                sentiment_class = 'negative'
            else:
                sentiment_class = 'neutral'
            
            self.stats['sentiment_distribution'][sentiment_class] += 1
            return scores
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}

    def _preprocess_text(self, text: str, consider_sentiment: bool = True) -> Tuple[List[str], Dict]:
        """Enhanced text preprocessing with sentiment consideration"""
        try:
            if not text or len(text.strip()) < self.config.min_review_length:
                return [], {}
            
            # Language detection
            language = self._detect_language(text)
            if language != 'en':
                logger.debug(f"Non-English text detected: {language}")
            
            # Sentiment analysis
            sentiment = self._analyze_sentiment(text) if consider_sentiment else {}
            
            # Text cleaning
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            if not text:
                return [], sentiment
            
            # Tokenization
            tokens = word_tokenize(text)
            
            # Enhanced filtering
            filtered_tokens = []
            for word in tokens:
                if (len(word) > 2 and 
                    word not in self.stopwords and 
                    word.isalpha() and
                    not word.isdigit()):
                    
                    lemmatized = self.lemmatizer.lemmatize(word)
                    filtered_tokens.append(lemmatized)
            
            return filtered_tokens, sentiment
            
        except Exception as e:
            logger.warning(f"Text preprocessing failed for text '{text[:50]}...': {e}")
            self.stats['errors_encountered'].append(f"Preprocessing error: {e}")
            return [], {}

    def _discover_semantic_keywords(self, all_reviews_text):
        """Dynamically discover keywords using semantic analysis and pattern recognition"""
        try:
            if not all_reviews_text or len(all_reviews_text.strip()) < 10:
                logger.warning("Insufficient text for semantic keyword discovery")
                return {}
            
            # Get preprocessed words and metadata
            processed_words, metadata = self._preprocess_text(all_reviews_text)
            
            if not processed_words:
                logger.warning("No words after preprocessing")
                return {}
            
            # Phase 1: Pattern-based discovery using regex patterns
            pattern_keywords = self._discover_keywords_by_patterns(all_reviews_text)
            
            # Phase 2: Semantic clustering of terms
            semantic_clusters = self._discover_semantic_clusters(processed_words)
            
            # Phase 3: Context-aware scoring
            contextual_scores = self._analyze_keyword_contexts(all_reviews_text, pattern_keywords)
            
            # Phase 4: Dynamic theme discovery
            discovered_themes = self._discover_themes_from_data(semantic_clusters, contextual_scores)
            
            # Combine all discoveries
            comprehensive_keywords = self._merge_keyword_discoveries(
                pattern_keywords, semantic_clusters, contextual_scores, discovered_themes
            )
            
            logger.info(f"Discovered {len(comprehensive_keywords)} semantic keyword groups")
            
            return comprehensive_keywords
            
        except Exception as e:
            logger.error(f"Error in semantic keyword discovery: {str(e)}")
            return {}
    
    def _discover_keywords_by_patterns(self, text):
        """Discover keywords using regex patterns and linguistic rules"""
        discovered = {}
        
        for concept_name, concept_data in self.tourism_concepts.items():
            discovered[concept_name] = []
            
            # Apply discovery patterns
            for pattern in concept_data['discovery_patterns']:
                matches = re.findall(pattern, text.lower(), re.IGNORECASE)
                discovered[concept_name].extend(matches)
            
            # Remove duplicates and empty matches
            discovered[concept_name] = list(set([m for m in discovered[concept_name] if m.strip()]))
        
        return discovered
    
    def _discover_semantic_clusters(self, processed_words):
        """Group semantically similar words using statistical clustering"""
        from collections import Counter, defaultdict
        import difflib
        
        # Count word frequencies
        word_freq = Counter(processed_words)
        significant_words = [word for word, freq in word_freq.items() 
                           if freq >= 2 and len(word) > 3]
        
        # Group similar words
        clusters = defaultdict(list)
        processed = set()
        
        for word in significant_words:
            if word in processed:
                continue
                
            # Find similar words using fuzzy matching
            similar_words = [w for w in significant_words 
                           if w not in processed and 
                           difflib.SequenceMatcher(None, word, w).ratio() > 0.7]
            
            if similar_words:
                cluster_key = min(similar_words, key=len)  # Use shortest as cluster key
                clusters[cluster_key].extend(similar_words)
                processed.update(similar_words)
        
        # Convert to semantic clusters with scores
        semantic_clusters = {}
        for cluster_key, words in clusters.items():
            total_freq = sum(word_freq[word] for word in words)
            semantic_clusters[cluster_key] = {
                'words': words,
                'total_frequency': total_freq,
                'cluster_size': len(words),
                'avg_frequency': total_freq / len(words)
            }
        
        return semantic_clusters
    
    def _analyze_keyword_contexts(self, text, pattern_keywords):
        """Analyze the context around keywords for sentiment and modifiers"""
        contextual_scores = {}
        sentences = re.split(r'[.!?]+', text)
        
        # Define context modifiers
        positive_modifiers = ['amazing', 'beautiful', 'excellent', 'fantastic', 'incredible', 
                            'outstanding', 'wonderful', 'perfect', 'stunning', 'magnificent']
        negative_modifiers = ['awful', 'terrible', 'horrible', 'disappointing', 'poor', 
                            'bad', 'worst', 'overpriced', 'crowded', 'dirty']
        intensity_modifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'quite', 
                             'rather', 'somewhat', 'really', 'truly', 'genuinely']
        
        for concept, keywords in pattern_keywords.items():
            contextual_scores[concept] = {}
            
            for keyword in keywords:
                if not keyword.strip():
                    continue
                    
                context_analysis = {
                    'positive_context': 0,
                    'negative_context': 0,
                    'intensity_score': 0,
                    'frequency': 0,
                    'confidence': 0
                }
                
                # Analyze each sentence containing the keyword
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        context_analysis['frequency'] += 1
                        
                        # Check for modifiers in context window
                        words = sentence.lower().split()
                        keyword_indices = [i for i, word in enumerate(words) 
                                         if keyword.lower() in word]
                        
                        for idx in keyword_indices:
                            # Check context window (±3 words)
                            start = max(0, idx - 3)
                            end = min(len(words), idx + 4)
                            context_words = words[start:end]
                            
                            # Score positive/negative context
                            positive_count = sum(1 for mod in positive_modifiers 
                                               if any(mod in word for word in context_words))
                            negative_count = sum(1 for mod in negative_modifiers 
                                               if any(mod in word for word in context_words))
                            intensity_count = sum(1 for mod in intensity_modifiers 
                                                if any(mod in word for word in context_words))
                            
                            context_analysis['positive_context'] += positive_count
                            context_analysis['negative_context'] += negative_count
                            context_analysis['intensity_score'] += intensity_count
                
                # Calculate confidence based on frequency and context clarity
                total_context = context_analysis['positive_context'] + context_analysis['negative_context']
                if total_context > 0:
                    context_analysis['confidence'] = min(1.0, 
                        (context_analysis['frequency'] * 0.3) + 
                        (total_context * 0.4) + 
                        (context_analysis['intensity_score'] * 0.3)
                    )
                else:
                    context_analysis['confidence'] = context_analysis['frequency'] * 0.1
                
                contextual_scores[concept][keyword] = context_analysis
        
        return contextual_scores
    
    def _discover_themes_from_data(self, semantic_clusters, contextual_scores):
        """Discover themes dynamically from the data using clustering and correlation"""
        discovered_themes = {}
        
        # Combine semantic and contextual information
        all_keywords = []
        keyword_features = []
        
        # Collect features for each keyword
        for concept, context_data in contextual_scores.items():
            for keyword, analysis in context_data.items():
                if analysis['confidence'] > 0.1:  # Only consider confident keywords
                    all_keywords.append((concept, keyword))
                    features = [
                        analysis['positive_context'],
                        analysis['negative_context'], 
                        analysis['intensity_score'],
                        analysis['frequency'],
                        analysis['confidence']
                    ]
                    keyword_features.append(features)
        
        if len(keyword_features) < 3:
            return {}
        
        # Use simple clustering to discover themes
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Normalize features
            features_array = np.array(keyword_features)
            if features_array.size > 0:
                features_normalized = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)
                
                # Determine optimal cluster count
                n_clusters = min(5, max(2, len(all_keywords) // 3))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_normalized)
                
                # Group keywords by cluster
                for i, (concept, keyword) in enumerate(all_keywords):
                    cluster_id = cluster_labels[i]
                    theme_name = f"discovered_theme_{cluster_id}"
                    
                    if theme_name not in discovered_themes:
                        discovered_themes[theme_name] = {
                            'keywords': [],
                            'source_concepts': set(),
                            'avg_confidence': 0,
                            'keyword_count': 0
                        }
                    
                    discovered_themes[theme_name]['keywords'].append(keyword)
                    discovered_themes[theme_name]['source_concepts'].add(concept)
                    discovered_themes[theme_name]['keyword_count'] += 1
                
                # Calculate average confidence for each theme
                for theme_name, theme_data in discovered_themes.items():
                    confidences = []
                    for keyword in theme_data['keywords']:
                        for concept in theme_data['source_concepts']:
                            if concept in contextual_scores and keyword in contextual_scores[concept]:
                                confidences.append(contextual_scores[concept][keyword]['confidence'])
                    
                    if confidences:
                        theme_data['avg_confidence'] = sum(confidences) / len(confidences)
                        theme_data['source_concepts'] = list(theme_data['source_concepts'])
                
        except Exception as e:
            logger.warning(f"Could not perform theme clustering: {e}")
        
        return discovered_themes
    
    def _merge_keyword_discoveries(self, pattern_keywords, semantic_clusters, contextual_scores, discovered_themes):
        """Merge all keyword discovery methods into a comprehensive result"""
        merged_result = {}
        
        # Start with base concepts from pattern discovery
        for concept_name, concept_data in self.tourism_concepts.items():
            merged_result[concept_name] = {
                'core_keywords': pattern_keywords.get(concept_name, []),
                'semantic_expansions': [],
                'contextual_scores': contextual_scores.get(concept_name, {}),
                'confidence_score': 0,
                'weight': concept_data['weight']
            }
            
            # Add semantic expansions
            concept_keywords = set(pattern_keywords.get(concept_name, []))
            for cluster_key, cluster_data in semantic_clusters.items():
                # Check if any cluster words relate to this concept
                if any(keyword in cluster_data['words'] for keyword in concept_keywords):
                    merged_result[concept_name]['semantic_expansions'].extend(cluster_data['words'])
            
            # Calculate overall confidence
            if merged_result[concept_name]['contextual_scores']:
                confidences = [score['confidence'] for score in 
                             merged_result[concept_name]['contextual_scores'].values()]
                merged_result[concept_name]['confidence_score'] = sum(confidences) / len(confidences)
        
        # Add discovered themes as new concepts
        for theme_name, theme_data in discovered_themes.items():
            if theme_data['avg_confidence'] > 0.3:  # Only add confident themes
                merged_result[theme_name] = {
                    'core_keywords': theme_data['keywords'],
                    'semantic_expansions': [],
                    'contextual_scores': {},
                    'confidence_score': theme_data['avg_confidence'],
                    'weight': 1.0,
                    'source_concepts': theme_data['source_concepts']
                }
        
        return merged_result

    def _categorize_semantic_discoveries_into_themes(self, semantic_keywords):
        """Categorize semantic discoveries into tourism themes using confidence-weighted scoring."""
        theme_scores = defaultdict(float)
        term_to_theme = {}
        
        if not semantic_keywords:
            return theme_scores, term_to_theme
        
        for theme_name, theme_data in semantic_keywords.items():
            confidence = theme_data.get('confidence_score', 0)
            weight = theme_data.get('weight', 1.0)
            
            # Calculate weighted theme score
            core_keywords_count = len(theme_data.get('core_keywords', []))
            semantic_expansions_count = len(theme_data.get('semantic_expansions', []))
            
            # Score based on keyword richness and confidence
            theme_score = (
                (core_keywords_count * 2.0) +  # Core keywords are more important
                (semantic_expansions_count * 1.0) +  # Expansions add breadth
                (confidence * 5.0)  # Confidence multiplier
            ) * weight
            
            if theme_score > 0:
                theme_scores[theme_name] = theme_score
                
                # Map individual keywords to themes
                all_keywords = theme_data.get('core_keywords', []) + theme_data.get('semantic_expansions', [])
                for keyword in all_keywords:
                    term_to_theme[keyword] = theme_name
        
        # Debug output
        logger.info(f"Categorized {len(term_to_theme)} semantic terms into {len(theme_scores)} themes")
        for theme, score in sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {theme}: {score:.3f}")
        
        return theme_scores, term_to_theme

    # Removed old _generate_city_tags method - replaced with _generate_city_tags_from_themes

    def _generate_city_tags_from_themes(self, theme_scores):
        """Generate city tags from theme scores using intelligent dynamic mapping."""
        if not theme_scores:
            return self.config.fallback_tags[:self.config.num_city_tags]
        
        # Sort themes by strength
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Dynamic theme to tag mapping that adapts to discovered themes
        base_theme_to_tag = {
            'architecture': 'Architectural',
            'heritage': 'Heritage',
            'experience': 'Experiential',
            'nature': 'Scenic',
            'activity': 'Adventure',
            'social': 'Social'
        }
        
        city_tags = []
        for theme, score in sorted_themes:
            if len(city_tags) >= self.config.num_city_tags:
                break
            
            # Handle base themes
            if theme in base_theme_to_tag:
                tag = base_theme_to_tag[theme]
            # Handle discovered themes
            elif theme.startswith('discovered_theme_'):
                # Create meaningful tag from discovered theme characteristics
                # This could be enhanced by analyzing the keywords within the theme
                theme_num = theme.split('_')[-1]
                tag = f"Theme-{theme_num.capitalize()}"
            else:
                # Convert theme name to title case tag
                tag = theme.replace('_', ' ').title()
            
            # Ensure uniqueness
            if tag not in city_tags:
                city_tags.append(tag)
        
        # Fill remaining slots with fallback tags if needed
        while len(city_tags) < self.config.num_city_tags:
            for fallback_tag in self.config.fallback_tags:
                if fallback_tag not in city_tags:
                    city_tags.append(fallback_tag)
                    break
            else:
                break
        
        return city_tags[:self.config.num_city_tags]

    def _categorize_places_by_semantic_content(self, places_data, semantic_keywords):
        """Categorize places using intelligent semantic analysis."""
        
        # Dynamic theme to tag mapping based on discovered themes
        dynamic_theme_to_tag = {}
        base_mappings = {
            'architecture': 'Architectural',
            'heritage': 'Heritage', 
            'experience': 'Experiential',
            'nature': 'Scenic',
            'activity': 'Adventure',
            'social': 'Social'
        }
        
        # Create mappings for all discovered themes
        for theme_name in semantic_keywords.keys():
            if theme_name in base_mappings:
                dynamic_theme_to_tag[theme_name] = base_mappings[theme_name]
            elif theme_name.startswith('discovered_theme_'):
                # For discovered themes, create meaningful names based on top keywords
                theme_data = semantic_keywords[theme_name]
                top_keywords = theme_data.get('core_keywords', [])[:3]
                if top_keywords:
                    tag_name = f"{top_keywords[0].capitalize()}-Related"
                    dynamic_theme_to_tag[theme_name] = tag_name
                else:
                    dynamic_theme_to_tag[theme_name] = "Unique"
            else:
                dynamic_theme_to_tag[theme_name] = theme_name.replace('_', ' ').title()
        
        place_categorization = defaultdict(list)
        
        for place_data in places_data:
            place_name = place_data['name']
            reviews_text = ' '.join(place_data['review_texts'])
            
            if not reviews_text.strip():
                place_categorization['Heritage'].append(place_name)
                continue
            
            # Intelligent semantic matching
            place_theme_scores = defaultdict(float)
            tokens, sentiment_data = self._preprocess_text(reviews_text)
            
            # Score each theme using semantic keywords and context
            for theme_name, theme_data in semantic_keywords.items():
                theme_score = 0
                confidence_multiplier = theme_data.get('confidence_score', 0.5)
                weight = theme_data.get('weight', 1.0)
                
                # Score core keywords with higher weight
                core_keywords = theme_data.get('core_keywords', [])
                for keyword in core_keywords:
                    keyword_frequency = sum(1 for token in tokens if keyword.lower() in token.lower())
                    theme_score += keyword_frequency * 2.0  # Core keywords get double weight
                
                # Score semantic expansions
                semantic_expansions = theme_data.get('semantic_expansions', [])
                for keyword in semantic_expansions:
                    keyword_frequency = sum(1 for token in tokens if keyword.lower() in token.lower())
                    theme_score += keyword_frequency * 1.0
                
                # Apply contextual analysis from the discovery phase
                contextual_scores = theme_data.get('contextual_scores', {})
                for keyword, context_data in contextual_scores.items():
                    if any(keyword.lower() in token.lower() for token in tokens):
                        # Boost positive context, reduce negative context
                        context_boost = (
                            context_data.get('positive_context', 0) * 0.5 - 
                            context_data.get('negative_context', 0) * 0.3
                        )
                        theme_score += context_boost
                
                # Apply confidence and weight multipliers
                final_score = theme_score * confidence_multiplier * weight
                
                if final_score > 0:
                    place_theme_scores[theme_name] = final_score
            
            # Assign place to strongest theme with confidence threshold
            if place_theme_scores:
                # Only assign if the score is above a minimum threshold
                max_score = max(place_theme_scores.values())
                if max_score >= 1.0:  # Minimum confidence threshold
                    strongest_theme = max(place_theme_scores.keys(), key=lambda x: place_theme_scores[x])
                    
                    if strongest_theme in dynamic_theme_to_tag:
                        city_tag = dynamic_theme_to_tag[strongest_theme]
                        place_categorization[city_tag].append(place_name)
                        logger.debug(f"{place_name}: {city_tag} (theme: {strongest_theme}, score: {max_score:.2f})")
                    else:
                        place_categorization['Unique'].append(place_name)
                        logger.debug(f"{place_name}: Unique (unmapped theme: {strongest_theme})")
                else:
                    # Low confidence, use fallback
                    place_categorization['General'].append(place_name)
                    logger.debug(f"{place_name}: General (low confidence: {max_score:.2f})")
            else:
                # No semantic matches, use fallback
                place_categorization['Heritage'].append(place_name)
                logger.debug(f"{place_name}: Heritage (no matches)")
        
        return place_categorization

    def _validate_input_data(self, api_data: Dict) -> bool:
        """Validate the structure and content of input data"""
        try:
            if not isinstance(api_data, dict):
                logger.error("Input data must be a dictionary")
                return False
            
            if 'places' not in api_data:
                logger.error("Input data must contain 'places' key")
                return False
            
            places = api_data['places']
            if not isinstance(places, list) or len(places) == 0:
                logger.error("Places must be a non-empty list")
                return False
            
            valid_places = 0
            for place in places:
                if (isinstance(place, dict) and 
                    'displayName' in place and 
                    'reviews' in place and
                    isinstance(place['reviews'], list)):
                    valid_places += 1
            
            if valid_places == 0:
                logger.error("No valid places found in input data")
                return False
            
            logger.info(f"Validated {valid_places} places out of {len(places)}")
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False

    def run_comprehensive_analysis(self, api_data: Dict) -> Dict[str, Any]:
        """Enhanced main analysis function with comprehensive error handling"""
        try:
            start_time = datetime.now()
            logger.info("Starting comprehensive city profiling analysis...")
            
            # Reset statistics
            self.stats = {
                'total_reviews': 0,
                'processed_reviews': 0,
                'languages_detected': defaultdict(int),
                'sentiment_distribution': defaultdict(int),
                'errors_encountered': []
            }
            
            # Validate input data
            if not self._validate_input_data(api_data):
                return {
                    "error": "Invalid input data structure",
                    "details": "Please ensure the data contains valid 'places' with reviews"
                }
            
            # Extract and process place data
            places_data = []
            all_review_texts = []
            
            for place in api_data.get("places", []):
                try:
                    place_name = place.get("displayName", {}).get("text", "Unknown Place")
                    reviews = place.get('reviews', [])
                    
                    valid_reviews = []
                    for review in reviews:
                        review_text = review.get('text', {}).get('text', '')
                        if review_text and len(review_text.strip()) >= self.config.min_review_length:
                            valid_reviews.append(review_text)
                            self.stats['total_reviews'] += 1
                    
                    if len(valid_reviews) >= self.config.min_reviews_per_place:
                        places_data.append({
                            'name': place_name,
                            'review_texts': valid_reviews,
                            'review_count': len(valid_reviews)
                        })
                        all_review_texts.extend(valid_reviews)
                        self.stats['processed_reviews'] += len(valid_reviews)
                        logger.debug(f"Processed {len(valid_reviews)} reviews for '{place_name}'")
                    else:
                        logger.debug(f"Skipping '{place_name}' - insufficient reviews ({len(valid_reviews)})")
                        
                except Exception as e:
                    logger.warning(f"Failed to process place data: {e}")
                    self.stats['errors_encountered'].append(f"Place processing error: {e}")
            
            if not all_review_texts:
                return {
                    "error": "No sufficient review content found for analysis",
                    "suggestion": "Ensure places have at least 2 reviews with 50+ characters each"
                }
            
            logger.info(f"Processing {len(places_data)} places with {len(all_review_texts)} total reviews")
            
            # Enhanced semantic analysis using intelligent discovery
            logger.info("Discovering semantic keywords and themes...")
            # Combine all review text for comprehensive analysis
            all_text = ' '.join(all_review_texts)
            semantic_keywords = self._discover_semantic_keywords(all_text)
            
            if not semantic_keywords:
                logger.warning("No semantic keywords discovered, using fallback analysis")
                city_tags = self.config.fallback_tags[:self.config.num_city_tags]
                category_scores = {}
                # Simple place categorization fallback
                place_categorization = defaultdict(list)
                for place_data in places_data:
                    place_categorization['Heritage'].append(place_data['name'])
            else:
                logger.info(f"Discovered {len(semantic_keywords)} semantic theme groups")
                
                # Categorize semantic discoveries into tourism themes
                category_scores, term_to_theme = self._categorize_semantic_discoveries_into_themes(semantic_keywords)
                
                # Generate city tags based on strongest themes
                city_tags = self._generate_city_tags_from_themes(category_scores)
                
                # Categorize places using intelligent semantic analysis
                place_categorization = self._categorize_places_by_semantic_content(places_data, semantic_keywords)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate comprehensive results
            result = {
                "city_profile": {
                    "name": self._extract_city_name(api_data, places_data),
                    "top_tags": city_tags,
                    "top_5_tags": city_tags,  # For backward compatibility
                    "category_scores": dict(category_scores),
                    "confidence_score": self._calculate_confidence(category_scores, len(all_review_texts))
                },
                "place_categorization": dict(place_categorization),
                "analysis_summary": {
                    "total_places_analyzed": len(places_data),
                    "total_reviews_processed": self.stats['processed_reviews'],
                    "processing_time_seconds": round(processing_time, 2),
                    "languages_detected": dict(self.stats['languages_detected']),
                    "sentiment_distribution": dict(self.stats['sentiment_distribution']),
                    "error_count": len(self.stats['errors_encountered'])
                }
            }
            
            if self.stats['errors_encountered']:
                result["warnings"] = {
                    "errors_encountered": self.stats['errors_encountered'][:10],  # Limit to first 10
                    "total_error_count": len(self.stats['errors_encountered'])
                }
            
            logger.info(f"Analysis completed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                "error": "Analysis failed due to unexpected error",
                "details": str(e),
                "fallback_result": {
                    "city_profile": {
                        "name": "Unknown City",
                        "top_tags": self.config.fallback_tags[:self.config.num_city_tags],
                        "confidence_score": 0.0
                    },
                    "analysis_summary": {
                        "status": "failed",
                        "error": str(e)
                    }
                }
            }

    def _extract_city_name(self, api_data: Dict, places_data: List[Dict]) -> str:
        """Extract city name from the data or use a reasonable default"""
        try:
            # Try to extract from the first place's location data
            if places_data:
                first_place = places_data[0]['name']
                # Simple heuristic to extract city name
                return "Analyzed City"  # Could be enhanced with location parsing
            return "Unknown City"
        except:
            return "Unknown City"

    def _calculate_confidence(self, category_scores: Dict[str, float], review_count: int) -> float:
        """Calculate confidence score based on data quality and consistency"""
        try:
            if not category_scores:
                return 0.0
            
            # Base confidence on review count
            review_confidence = min(1.0, review_count / 100)  # Full confidence at 100+ reviews
            
            # Category score distribution (more balanced = higher confidence)
            scores = list(category_scores.values())
            if len(scores) > 1:
                score_std = np.std(scores)
                score_mean = np.mean(scores)
                score_consistency = 1.0 - min(1.0, score_std / score_mean if score_mean > 0 else 1.0)
            else:
                score_consistency = 0.5
            
            # Combine factors
            confidence = (review_confidence * 0.6 + score_consistency * 0.4)
            return round(confidence, 3)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    # Backward compatibility method
    def run_full_analysis(self, api_data):
        """Backward compatibility wrapper for the enhanced analysis"""
        return self.run_comprehensive_analysis(api_data)


if __name__ == "__main__":
    # Example usage with the robust profiler
    print("Testing Robust City Profiler...")
    
    # Example configuration
    config = ProfilerConfig(
        num_city_tags=5,
        min_reviews_per_place=1,  # Lowered for testing
        min_review_length=30,     # Lowered for testing
        enable_sentiment_analysis=True,
        enable_language_detection=True
    )
    
    json_string = {
    "places": [
        {
            "id": "ChIJQ4srFBimvzsRJQI7KOdIlP0",
            "displayName": {
                "text": "Dudhsagar Falls",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJQ4srFBimvzsRJQI7KOdIlP0/reviews/ChdDSUhNMG9nS0VKZldwOFc1dDVhRmd3RRAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Dudhsagar Waterfalls is a breathtaking natural wonder and an absolute must visit! Cascading down from a great height, the milky white waters resemble a sea of milk hence the name. Surrounded by lush greenery and located on the Goa Karnataka border, the falls are especially stunning during the monsoon. Whether you’re trekking through the forest or riding the scenic train route, the view is unforgettable. Pro tip: book your jeep safari or guided tour in advance, especially during peak season, as access is limited. Also, check for forest department permits. A very good experience for nature lovers and adventure seekers alike!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Dudhsagar Waterfalls is a breathtaking natural wonder and an absolute must visit! Cascading down from a great height, the milky white waters resemble a sea of milk hence the name. Surrounded by lush greenery and located on the Goa Karnataka border, the falls are especially stunning during the monsoon. Whether you’re trekking through the forest or riding the scenic train route, the view is unforgettable. Pro tip: book your jeep safari or guided tour in advance, especially during peak season, as access is limited. Also, check for forest department permits. A very good experience for nature lovers and adventure seekers alike!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Mayalcito Bhalala",
                        "uri": "https://www.google.com/maps/contrib/106021385084149807258/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXJ_aEEXNZHmfiqjP8WgjJVi6wagVM_FKCs4My9DigJ6OPDtZi2=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-11T14:58:20.247982Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VKZldwOFc1dDVhRmd3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VKZldwOFc1dDVhRmd3RRAB!2m1!1s0x3bbfa618142b8b43:0xfd9448e7283b0225"
                },
                {
                    "name": "places/ChIJQ4srFBimvzsRJQI7KOdIlP0/reviews/Ci9DQUlRQUNvZENodHljRjlvT21sNlpXTnlRVXRSZUZWcGREbEdURmhSWjB4c1lYYxAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 4,
                    "text": {
                        "text": "Dudhsagar Waterfalls is a breathtaking natural wonder nestled in the Western Ghats. The sheer force and beauty of the cascading water is truly mesmerizing, especially during the monsoon season. The surrounding greenery adds to the charm, making it a perfect spot for nature lovers and photographers alike. While the journey to reach the falls can be a bit challenging, it only adds to the sense of adventure. Overall, a must-visit destination, though better facilities could enhance the experience.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Dudhsagar Waterfalls is a breathtaking natural wonder nestled in the Western Ghats. The sheer force and beauty of the cascading water is truly mesmerizing, especially during the monsoon season. The surrounding greenery adds to the charm, making it a perfect spot for nature lovers and photographers alike. While the journey to reach the falls can be a bit challenging, it only adds to the sense of adventure. Overall, a must-visit destination, though better facilities could enhance the experience.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Shreyas Bhole",
                        "uri": "https://www.google.com/maps/contrib/107176880627706441104/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjURNHX0I6VecJssDngmvNCaTLRTDCulpCcyzQZgTyiZm81mZECp=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-26T07:50:53.411683566Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT21sNlpXTnlRVXRSZUZWcGREbEdURmhSWjB4c1lYYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT21sNlpXTnlRVXRSZUZWcGREbEdURmhSWjB4c1lYYxAB!2m1!1s0x3bbfa618142b8b43:0xfd9448e7283b0225"
                },
                {
                    "name": "places/ChIJQ4srFBimvzsRJQI7KOdIlP0/reviews/ChdDSUhNMG9nS0VJQ0FnTURvNXN6SjF3RRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Back in August 2022, I had an unforgettable experience trekking to Dudhsagar Waterfall—a challenging yet thrilling 14 km hike along the railway track (now closed for public access). The journey was intense and exhausting, but the moment I saw the milky white waterfall in its full glory, all the fatigue just disappeared. It was absolutely worth every step.\n\nThe trek included walking through dark tunnels, witnessing lush greenery, and soaking in breathtaking scenic views that felt straight out of a dream. It’s one of those unique experiences that leave a lasting impression.\n\nEven though the track route is now closed, I highly recommend visiting Dudhsagar—at least once in your life—to feel that overwhelming blend of nature, adventure, and awe.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Back in August 2022, I had an unforgettable experience trekking to Dudhsagar Waterfall—a challenging yet thrilling 14 km hike along the railway track (now closed for public access). The journey was intense and exhausting, but the moment I saw the milky white waterfall in its full glory, all the fatigue just disappeared. It was absolutely worth every step.\n\nThe trek included walking through dark tunnels, witnessing lush greenery, and soaking in breathtaking scenic views that felt straight out of a dream. It’s one of those unique experiences that leave a lasting impression.\n\nEven though the track route is now closed, I highly recommend visiting Dudhsagar—at least once in your life—to feel that overwhelming blend of nature, adventure, and awe.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Uditya Raj Srivastva",
                        "uri": "https://www.google.com/maps/contrib/113809027717945818760/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXZckOx_j36xGewWzwrtNSe5dOHNQFJoc6fQ1HnhBJv03PtzK2ILw=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-24T11:48:32.754344Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURvNXN6SjF3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURvNXN6SjF3RRAB!2m1!1s0x3bbfa618142b8b43:0xfd9448e7283b0225"
                },
                {
                    "name": "places/ChIJQ4srFBimvzsRJQI7KOdIlP0/reviews/ChZDSUhNMG9nS0VJQ0FnTURnZ195V1VREAE",
                    "relativePublishTimeDescription": "4 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Dudhsagar Waterfalls is a truly mesmerizing sight! The long waterfall cascades down in stunning white streams, resembling a river of milk, which perfectly justifies its name. The shallow waters near the base allow visitors to enjoy the cool, refreshing experience up close.\n\nWhen we arrived, it wasn’t too crowded, but there was a continuous flow of visitors. One of the most fascinating sights was witnessing live fishes over a foot long swimming in the clear waters. Though the pond is shallow in some areas, certain spots are quite deep, making it mandatory to wear a life jacket for safety.\n\nA word of caution—be careful of monkeys as they tend to snatch belongings. Also, the wet surface around the falls is extremely slippery. I personally slipped, which left me with pain in my back and knee, so wearing proper footwear and walking cautiously is highly recommended.\n\nDespite the slippery terrain, the beauty of Dudhsagar Waterfalls makes it a must-visit destination. The sight of the cascading white water, the peaceful surroundings, and the overall experience make it truly unforgettable!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Dudhsagar Waterfalls is a truly mesmerizing sight! The long waterfall cascades down in stunning white streams, resembling a river of milk, which perfectly justifies its name. The shallow waters near the base allow visitors to enjoy the cool, refreshing experience up close.\n\nWhen we arrived, it wasn’t too crowded, but there was a continuous flow of visitors. One of the most fascinating sights was witnessing live fishes over a foot long swimming in the clear waters. Though the pond is shallow in some areas, certain spots are quite deep, making it mandatory to wear a life jacket for safety.\n\nA word of caution—be careful of monkeys as they tend to snatch belongings. Also, the wet surface around the falls is extremely slippery. I personally slipped, which left me with pain in my back and knee, so wearing proper footwear and walking cautiously is highly recommended.\n\nDespite the slippery terrain, the beauty of Dudhsagar Waterfalls makes it a must-visit destination. The sight of the cascading white water, the peaceful surroundings, and the overall experience make it truly unforgettable!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "VIKRANTH REDDY",
                        "uri": "https://www.google.com/maps/contrib/108355038268052057966/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjX45cChGbcdK1Gp3OPtw0mXm0t1bRcYrno9XY31qj5vuIRwdLM4LA=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-02-26T23:34:13.969837Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTURnZ195V1VREAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTURnZ195V1VREAE!2m1!1s0x3bbfa618142b8b43:0xfd9448e7283b0225"
                },
                {
                    "name": "places/ChIJQ4srFBimvzsRJQI7KOdIlP0/reviews/ChZDSUhNMG9nS0VNM0Y3cVRRNjYtUldBEAE",
                    "relativePublishTimeDescription": "4 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Dudhsagar Waterfalls is one of the most breathtaking natural wonders in Goa and a must-visit for any nature lover or adventure seeker. Tucked away deep in the Western Ghats near the Goa-Karnataka border, this majestic four-tiered waterfall drops from over 300 meters, creating a spectacular view that looks just like a \"sea of milk\" cascading down the rocks—true to its name Dudhsagar (meaning “Ocean of Milk”).\n\nThe journey to reach the falls is just as exciting as the destination. You can either take a forest jeep safari through the Bhagwan Mahavir Wildlife Sanctuary or trek along the railway track if you're up for a real adventure. The trail is surrounded by dense forests, fresh air, and occasional wildlife sightings. The monsoon season is the best time to visit, as the falls are in full force and the greenery is lush and vibrant.\n\n*Carry sufficient cash, food , drinking water. Jeep safari cost around 1500/- rs per person.\n\nOnce at the base, the sound of the gushing water and the mist in the air create a magical atmosphere. It’s a peaceful escape from Goa’s beaches and party scenes.\n\nDudhsagar is raw, powerful, and absolutely unforgettable. Be sure to carry water, wear proper shoes, and respect nature—it’s worth every bit of the journey.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Dudhsagar Waterfalls is one of the most breathtaking natural wonders in Goa and a must-visit for any nature lover or adventure seeker. Tucked away deep in the Western Ghats near the Goa-Karnataka border, this majestic four-tiered waterfall drops from over 300 meters, creating a spectacular view that looks just like a \"sea of milk\" cascading down the rocks—true to its name Dudhsagar (meaning “Ocean of Milk”).\n\nThe journey to reach the falls is just as exciting as the destination. You can either take a forest jeep safari through the Bhagwan Mahavir Wildlife Sanctuary or trek along the railway track if you're up for a real adventure. The trail is surrounded by dense forests, fresh air, and occasional wildlife sightings. The monsoon season is the best time to visit, as the falls are in full force and the greenery is lush and vibrant.\n\n*Carry sufficient cash, food , drinking water. Jeep safari cost around 1500/- rs per person.\n\nOnce at the base, the sound of the gushing water and the mist in the air create a magical atmosphere. It’s a peaceful escape from Goa’s beaches and party scenes.\n\nDudhsagar is raw, powerful, and absolutely unforgettable. Be sure to carry water, wear proper shoes, and respect nature—it’s worth every bit of the journey.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Saurav Das",
                        "uri": "https://www.google.com/maps/contrib/107317624996188504019/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWL8VXLXzLFumB6JMh-jgqe02fpb48hWmadpKc0biAk6D4jRwEN=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-06T05:25:42.873705Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VNM0Y3cVRRNjYtUldBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VNM0Y3cVRRNjYtUldBEAE!2m1!1s0x3bbfa618142b8b43:0xfd9448e7283b0225"
                }
            ]
        },
        {
            "id": "ChIJa72MxnXBvzsRHHtpszB2g6M",
            "displayName": {
                "text": "Fort Aguada",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJa72MxnXBvzsRHHtpszB2g6M/reviews/ChdDSUhNMG9nS0VMU0dvWV9kLWUtRDd3RRAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 4,
                    "text": {
                        "text": "A must-visit in Goa!\n\nFort Aguada is a beautiful historic site with stunning views of the Arabian Sea. The architecture and the old lighthouse give you a glimpse into Goa’s Portuguese past. It's well maintained, and the sunset here is absolutely breathtaking.\n\nIt’s a peaceful spot to relax, take photos, and learn a bit of history. Try visiting early morning or late afternoon to avoid the heat and crowds. Definitely worth a stop if you’re in North Goa!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "A must-visit in Goa!\n\nFort Aguada is a beautiful historic site with stunning views of the Arabian Sea. The architecture and the old lighthouse give you a glimpse into Goa’s Portuguese past. It's well maintained, and the sunset here is absolutely breathtaking.\n\nIt’s a peaceful spot to relax, take photos, and learn a bit of history. Try visiting early morning or late afternoon to avoid the heat and crowds. Definitely worth a stop if you’re in North Goa!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "pranjal saxena",
                        "uri": "https://www.google.com/maps/contrib/113363501689809880164/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjX5O9RDAF8HQztyNq07-0Gh-u0DsQMvo1AEW8ewKliyPdTOnOSxSw=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-08T16:43:36.749942Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VMU0dvWV9kLWUtRDd3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VMU0dvWV9kLWUtRDd3RRAB!2m1!1s0x3bbfc175c68cbd6b:0xa3837630b3697b1c"
                },
                {
                    "name": "places/ChIJa72MxnXBvzsRHHtpszB2g6M/reviews/Ci9DQUlRQUNvZENodHljRjlvT25weGIwODVZMEZEYjFwSFlUbHdObkY1T1Zab0xYYxAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 5,
                    "text": {
                        "text": "Fort Aguada – A Majestic Blend of History & Sea Views 🌊\n\nWhether you’re a history buff, a photography enthusiast, or simply seeking a serene sea breeze with a side of green‑blue views, Fort Aguada is an unbeatable choice. I left with photos I’ll treasure—and memories of standing where centuries of ships once paused to drink and defend. Highly recommended!\n\n⭐⭐⭐⭐⭐",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Fort Aguada – A Majestic Blend of History & Sea Views 🌊\n\nWhether you’re a history buff, a photography enthusiast, or simply seeking a serene sea breeze with a side of green‑blue views, Fort Aguada is an unbeatable choice. I left with photos I’ll treasure—and memories of standing where centuries of ships once paused to drink and defend. Highly recommended!\n\n⭐⭐⭐⭐⭐",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Neelam Takola",
                        "uri": "https://www.google.com/maps/contrib/113540526791077602399/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjX63sgmEpm2C7HxaNrB2txImS_UoJI63pXvcn4bzteJhnV1tNaY=s128-c0x00000000-cc-rp-mo-ba3"
                    },
                    "publishTime": "2025-06-26T05:51:42.832796609Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT25weGIwODVZMEZEYjFwSFlUbHdObkY1T1Zab0xYYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT25weGIwODVZMEZEYjFwSFlUbHdObkY1T1Zab0xYYxAB!2m1!1s0x3bbfc175c68cbd6b:0xa3837630b3697b1c"
                },
                {
                    "name": "places/ChIJa72MxnXBvzsRHHtpszB2g6M/reviews/ChZDSUhNMG9nS0VJQ0FnTUNJdVotWmVnEAE",
                    "relativePublishTimeDescription": "3 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Aguada Fort, Goa\n\nOverview:\nAguada Fort is one of the most famous and well-preserved forts in Goa, built by the Portuguese in 1612 to protect against Dutch and Maratha invasions. It is located near Sinquerim Beach, overlooking the Arabian Sea, and offers breathtaking views. The name \"Aguada\" comes from the Portuguese word for water, as the fort had a freshwater spring that supplied water to ships.\n\nKey Highlights:\n\nHistorical Significance: Built as a defense fort and served as a crucial water supply station for passing ships.\n\nArchitectural Marvel: Features thick walls, bastions, and a large freshwater reservoir.\n\nLighthouse: The Aguada Lighthouse, built in 1864, is one of the oldest lighthouses in Asia.\n\nPanoramic Views: Offers stunning views of the Arabian Sea and nearby beaches.\n\nPrison: A section of the fort was later used as a prison, which still exists today.\n\nLocation:\n\nSituated in North Goa, near Candolim and Sinquerim Beach.\n\nApproximately 15 km from Panaji, the capital of Goa.\n\nBest Time to Visit:\n\nOctober to March (pleasant weather, perfect for sightseeing).\n\nSunset time for mesmerizing views of the sea.\n\nEntry Fee & Timings:\n\nEntry: Free\n\nTimings: 9:30 AM – 6:00 PM\n\nAguada Fort is a must-visit for history lovers, photographers, and anyone who wants to explore Goa’s colonial past while enjoying stunning coastal views.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Aguada Fort, Goa\n\nOverview:\nAguada Fort is one of the most famous and well-preserved forts in Goa, built by the Portuguese in 1612 to protect against Dutch and Maratha invasions. It is located near Sinquerim Beach, overlooking the Arabian Sea, and offers breathtaking views. The name \"Aguada\" comes from the Portuguese word for water, as the fort had a freshwater spring that supplied water to ships.\n\nKey Highlights:\n\nHistorical Significance: Built as a defense fort and served as a crucial water supply station for passing ships.\n\nArchitectural Marvel: Features thick walls, bastions, and a large freshwater reservoir.\n\nLighthouse: The Aguada Lighthouse, built in 1864, is one of the oldest lighthouses in Asia.\n\nPanoramic Views: Offers stunning views of the Arabian Sea and nearby beaches.\n\nPrison: A section of the fort was later used as a prison, which still exists today.\n\nLocation:\n\nSituated in North Goa, near Candolim and Sinquerim Beach.\n\nApproximately 15 km from Panaji, the capital of Goa.\n\nBest Time to Visit:\n\nOctober to March (pleasant weather, perfect for sightseeing).\n\nSunset time for mesmerizing views of the sea.\n\nEntry Fee & Timings:\n\nEntry: Free\n\nTimings: 9:30 AM – 6:00 PM\n\nAguada Fort is a must-visit for history lovers, photographers, and anyone who wants to explore Goa’s colonial past while enjoying stunning coastal views.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "FINANCIAL JUNCTION",
                        "uri": "https://www.google.com/maps/contrib/112486192129421183213/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVnrI54WvyMhupFgCYP4fDa06I8WPpO8Yvy7GuFF3HQa61Pmo4YwA=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-04-03T10:07:59.127915Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNJdVotWmVnEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNJdVotWmVnEAE!2m1!1s0x3bbfc175c68cbd6b:0xa3837630b3697b1c"
                },
                {
                    "name": "places/ChIJa72MxnXBvzsRHHtpszB2g6M/reviews/ChdDSUhNMG9nS0VKdkhpYWJUc3JtTnpBRRAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "A great place to visit! The fort doesn’t have too many steps—just around 30–35—which are easy to climb, making it accessible for most visitors. Entry is just ₹30. There’s a water plant at the entrance, and once you climb up, you’re greeted with stunning views of the sea. The old Portuguese architecture adds a unique charm, making it perfect for photography—especially if you enjoy creative shots leaning against the rustic fort walls. A must-visit spot for history lovers and photo enthusiasts alike!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "A great place to visit! The fort doesn’t have too many steps—just around 30–35—which are easy to climb, making it accessible for most visitors. Entry is just ₹30. There’s a water plant at the entrance, and once you climb up, you’re greeted with stunning views of the sea. The old Portuguese architecture adds a unique charm, making it perfect for photography—especially if you enjoy creative shots leaning against the rustic fort walls. A must-visit spot for history lovers and photo enthusiasts alike!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "sudheer reddy",
                        "uri": "https://www.google.com/maps/contrib/103753537500395760255/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXAeRmrGu7w8-D4oRNz-dbGifKZI9JbfNcelSMu8wZwVt5xnsUoEA=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-07T18:43:49.345997Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VKdkhpYWJUc3JtTnpBRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VKdkhpYWJUc3JtTnpBRRAB!2m1!1s0x3bbfc175c68cbd6b:0xa3837630b3697b1c"
                },
                {
                    "name": "places/ChIJa72MxnXBvzsRHHtpszB2g6M/reviews/ChZDSUhNMG9nS0VJbUw2ZkRLanB1RVhBEAE",
                    "relativePublishTimeDescription": "4 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Fort Aguada is a beautiful spot and one of the best places to visit if you love forts. The view from the top is just stunning, and the fresh air you get to breathe there makes it even more refreshing. It’s peaceful, scenic, and perfect for spending some relaxing time while exploring history. A must-visit for anyone coming to Goa!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Fort Aguada is a beautiful spot and one of the best places to visit if you love forts. The view from the top is just stunning, and the fresh air you get to breathe there makes it even more refreshing. It’s peaceful, scenic, and perfect for spending some relaxing time while exploring history. A must-visit for anyone coming to Goa!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Sai Yashwanth P",
                        "uri": "https://www.google.com/maps/contrib/106384600525550009185/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVmZjByjyPBRJ-u3NWuNBW2LJOlGzvNm_jLUtDoLAlwIhglIdjAyg=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-04T19:50:48.651043Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJbUw2ZkRLanB1RVhBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJbUw2ZkRLanB1RVhBEAE!2m1!1s0x3bbfc175c68cbd6b:0xa3837630b3697b1c"
                }
            ]
        },
        {
            "id": "ChIJiQKbufK-vzsRuKcVn2hWgB4",
            "displayName": {
                "text": "Basilica of Bom Jesus",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJiQKbufK-vzsRuKcVn2hWgB4/reviews/Ci9DQUlRQUNvZENodHljRjlvT2sxM2EwbFBVM0Z0ZDFNMlZtaHlORVIyVTBKZldsRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 5,
                    "text": {
                        "text": "Nice place to visit.. world heritage site..worth visiting..get a glimpse of history with its architecture..if coming to goa..must visit this place\nThis ancient church gives a glimpse into the architecture and culture of those times.\nIt's a pleasure to walk along and soak in the environment.\n\nHowever the painting gallery on the first floor makes you wish that the maintenance of all those ancient paintings and sculptures is done in a better manner.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Nice place to visit.. world heritage site..worth visiting..get a glimpse of history with its architecture..if coming to goa..must visit this place\nThis ancient church gives a glimpse into the architecture and culture of those times.\nIt's a pleasure to walk along and soak in the environment.\n\nHowever the painting gallery on the first floor makes you wish that the maintenance of all those ancient paintings and sculptures is done in a better manner.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "vimal vp",
                        "uri": "https://www.google.com/maps/contrib/109791943551534961072/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWIXd34qrvhcejNR5aD3FNMvl8KftQLNhy6nWPoQlNyL7VnyQ1O=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-20T11:40:17.021449181Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2sxM2EwbFBVM0Z0ZDFNMlZtaHlORVIyVTBKZldsRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2sxM2EwbFBVM0Z0ZDFNMlZtaHlORVIyVTBKZldsRRAB!2m1!1s0x3bbfbef2b99b0289:0x1e8056689f15a7b8"
                },
                {
                    "name": "places/ChIJiQKbufK-vzsRuKcVn2hWgB4/reviews/Ci9DQUlRQUNvZENodHljRjlvT2psUWRtdDJkVlZsU1dkRFdXMDNObWg0VFhReVVuYxAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 4,
                    "text": {
                        "text": "Basilica of Bom Jesus in Goa is one of the most iconic and historically significant churches in India. Recognized as a UNESCO World Heritage Site, this 400-year-old church beautifully showcases Baroque architecture with intricate details, old-world charm, and spiritual significance. The moment you step in, the grandeur of the gilded altars and the beautifully carved wooden interiors captivate you completely.\n\nThe church holds the mortal remains of St. Francis Xavier, which is placed in a silver casket and draws pilgrims from around the world. Despite the heavy footfall, the place maintains a serene and peaceful ambiance, making it perfect for both spiritual seekers and history enthusiasts.\n\nThe outer structure, though aged and slightly weathered, adds a rustic charm that complements its historical importance. The museum inside the complex offers valuable insights into Goa’s rich religious history and colonial past. The entire area around the Basilica is well maintained, with beautiful lawns and pathways ideal for a peaceful stroll.\n\nWhether you're a history buff, an architecture lover, or someone seeking peace, the Basilica of Bom Jesus is a must-visit. It’s a place where time stands still, offering a perfect blend of spirituality, history, and art.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Basilica of Bom Jesus in Goa is one of the most iconic and historically significant churches in India. Recognized as a UNESCO World Heritage Site, this 400-year-old church beautifully showcases Baroque architecture with intricate details, old-world charm, and spiritual significance. The moment you step in, the grandeur of the gilded altars and the beautifully carved wooden interiors captivate you completely.\n\nThe church holds the mortal remains of St. Francis Xavier, which is placed in a silver casket and draws pilgrims from around the world. Despite the heavy footfall, the place maintains a serene and peaceful ambiance, making it perfect for both spiritual seekers and history enthusiasts.\n\nThe outer structure, though aged and slightly weathered, adds a rustic charm that complements its historical importance. The museum inside the complex offers valuable insights into Goa’s rich religious history and colonial past. The entire area around the Basilica is well maintained, with beautiful lawns and pathways ideal for a peaceful stroll.\n\nWhether you're a history buff, an architecture lover, or someone seeking peace, the Basilica of Bom Jesus is a must-visit. It’s a place where time stands still, offering a perfect blend of spirituality, history, and art.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "swarna gupta",
                        "uri": "https://www.google.com/maps/contrib/100361549156067196032/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXabrMR4Z_w1usJFhw8z_ir9ANwC0WdgQkKH9kDQTTGTw5LICbK=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-26T19:18:29.100939428Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2psUWRtdDJkVlZsU1dkRFdXMDNObWg0VFhReVVuYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2psUWRtdDJkVlZsU1dkRFdXMDNObWg0VFhReVVuYxAB!2m1!1s0x3bbfbef2b99b0289:0x1e8056689f15a7b8"
                },
                {
                    "name": "places/ChIJiQKbufK-vzsRuKcVn2hWgB4/reviews/Ci9DQUlRQUNvZENodHljRjlvT20xNU5VMUtZMjFUU1ZodU1USXRiMDB3TkdKdGNYYxAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 4,
                    "text": {
                        "text": "Visited the Basilica of Bom Jesus in Old Goa — truly a must-visit place! The architecture is stunning, and the surrounding area is peaceful and well-maintained. The historic charm of the building, especially with the preserved body of St. Francis Xavier, makes it a special experience. A perfect blend of heritage and serenity. Highly recommended for anyone exploring Goa!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Visited the Basilica of Bom Jesus in Old Goa — truly a must-visit place! The architecture is stunning, and the surrounding area is peaceful and well-maintained. The historic charm of the building, especially with the preserved body of St. Francis Xavier, makes it a special experience. A perfect blend of heritage and serenity. Highly recommended for anyone exploring Goa!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Sethu Mohan",
                        "uri": "https://www.google.com/maps/contrib/108446794916543714677/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXH1nO1HI2QC6ESedndWqgi2ZE9YGBOA43eAkM5PDbybX9uHhhV=s128-c0x00000000-cc-rp-mo-ba3"
                    },
                    "publishTime": "2025-06-21T05:29:29.904998670Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT20xNU5VMUtZMjFUU1ZodU1USXRiMDB3TkdKdGNYYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT20xNU5VMUtZMjFUU1ZodU1USXRiMDB3TkdKdGNYYxAB!2m1!1s0x3bbfbef2b99b0289:0x1e8056689f15a7b8"
                },
                {
                    "name": "places/ChIJiQKbufK-vzsRuKcVn2hWgB4/reviews/Ci9DQUlRQUNvZENodHljRjlvT25GdFpGVnJTVUo1Vld0S1VtUlNkV2hEY0ZGU1RGRRAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Basilica of Bom Jesus, Roman Catholic church in the abandoned city of Old Goa, Goa state, India. Built between 1594 and 1605, it is regarded as an outstanding example of Renaissance Baroque and Portuguese colonial architecture. The basilica is also known for housing the remains of the missionary St. Francis Xavier, who was based in Goa in 1541–49.\n\nAn old saint body can be found lying there that has not change for year to come.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Basilica of Bom Jesus, Roman Catholic church in the abandoned city of Old Goa, Goa state, India. Built between 1594 and 1605, it is regarded as an outstanding example of Renaissance Baroque and Portuguese colonial architecture. The basilica is also known for housing the remains of the missionary St. Francis Xavier, who was based in Goa in 1541–49.\n\nAn old saint body can be found lying there that has not change for year to come.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Aditya beekharry",
                        "uri": "https://www.google.com/maps/contrib/100865567923115457288/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXtWK45EMl7KUxqNgoE6Eh3WXDPYcrvKQGp_ZwbwSIRlYcvP8IR3A=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-11T14:34:17.162451942Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT25GdFpGVnJTVUo1Vld0S1VtUlNkV2hEY0ZGU1RGRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT25GdFpGVnJTVUo1Vld0S1VtUlNkV2hEY0ZGU1RGRRAB!2m1!1s0x3bbfbef2b99b0289:0x1e8056689f15a7b8"
                },
                {
                    "name": "places/ChIJiQKbufK-vzsRuKcVn2hWgB4/reviews/ChdDSUhNMG9nS0VJQ0FnTURvdzVqSjBnRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Definitely one of the most remarkable places to visit while you are in Old Goa. Being a World Heritage site, the place is kept in a very well preserved state. The Church also holds the last remains of St. Francis Xavier, which everyone should see for real. Weddings are hosted here in this beautiful Church, bringing an elegance touch to people's special day. The place is also surrounded by small shops from where you can get souvenirs as well.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Definitely one of the most remarkable places to visit while you are in Old Goa. Being a World Heritage site, the place is kept in a very well preserved state. The Church also holds the last remains of St. Francis Xavier, which everyone should see for real. Weddings are hosted here in this beautiful Church, bringing an elegance touch to people's special day. The place is also surrounded by small shops from where you can get souvenirs as well.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Ritam Biswas",
                        "uri": "https://www.google.com/maps/contrib/111572271647730384755/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocIt-_IRHG3HP5wmsvI8ijHoglV8_UOcHycCJKup45kLrlznmYrt=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-28T06:25:16.719479Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURvdzVqSjBnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURvdzVqSjBnRRAB!2m1!1s0x3bbfbef2b99b0289:0x1e8056689f15a7b8"
                }
            ]
        },
        {
            "id": "ChIJv9F7IjzBvzsR70r6HmyML_o",
            "displayName": {
                "text": "Candolim Beach",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJv9F7IjzBvzsR70r6HmyML_o/reviews/Ci9DQUlRQUNvZENodHljRjlvT2pKT1NUUnJjalp5VmxwQmFGZDFSVmhSYzNvMWRsRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 5,
                    "text": {
                        "text": "A very serene, clean, white sand beach. Perfect to come in early morning or evening to witness gorgeous sunsets.\n\nThe beach is beautiful. There were very few people on the beach. There are limited number of shacks. It’s a long stretch of beach. If you turn left yoj will reach Sinquerim Beach and the Sinquerim Fort (Lower Agaunda Fort) if you keep walking towards your right you will reach the more crowded Calanghute Beach.\n\nPlenty of eating, casinos, happening places and shopping options on the main street.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "A very serene, clean, white sand beach. Perfect to come in early morning or evening to witness gorgeous sunsets.\n\nThe beach is beautiful. There were very few people on the beach. There are limited number of shacks. It’s a long stretch of beach. If you turn left yoj will reach Sinquerim Beach and the Sinquerim Fort (Lower Agaunda Fort) if you keep walking towards your right you will reach the more crowded Calanghute Beach.\n\nPlenty of eating, casinos, happening places and shopping options on the main street.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Parv Kaushik",
                        "uri": "https://www.google.com/maps/contrib/108092294772982104238/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUri5j4sLarWcGD9Ayzu_D6SDNl4WL0tJub7YhyotMdOlMFyK7lQQ=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-23T07:06:51.769197585Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2pKT1NUUnJjalp5VmxwQmFGZDFSVmhSYzNvMWRsRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2pKT1NUUnJjalp5VmxwQmFGZDFSVmhSYzNvMWRsRRAB!2m1!1s0x3bbfc13c227bd1bf:0xfa2f8c6c1efa4aef"
                },
                {
                    "name": "places/ChIJv9F7IjzBvzsR70r6HmyML_o/reviews/Ci9DQUlRQUNvZENodHljRjlvT2tsVmIxRnFNbkp5ZFc4MlpIcHFUSEJrZEhOeVZFRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 4,
                    "text": {
                        "text": "It's good but dangerous during the rainy season and no one is allowed to get into water and no water sports. Very less crowded. The seashore is very nearby and deep due to which no one is allowed to enter the water.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "It's good but dangerous during the rainy season and no one is allowed to get into water and no water sports. Very less crowded. The seashore is very nearby and deep due to which no one is allowed to enter the water.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Pavan Bandari",
                        "uri": "https://www.google.com/maps/contrib/105694954332040167086/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVFtyKkyOY-vtpfDd8o73BNeyuUeIaZpg-vVvcdMWhWPli74uCb=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-22T12:43:56.396313333Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2tsVmIxRnFNbkp5ZFc4MlpIcHFUSEJrZEhOeVZFRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2tsVmIxRnFNbkp5ZFc4MlpIcHFUSEJrZEhOeVZFRRAB!2m1!1s0x3bbfc13c227bd1bf:0xfa2f8c6c1efa4aef"
                },
                {
                    "name": "places/ChIJv9F7IjzBvzsR70r6HmyML_o/reviews/ChdDSUhNMG9nS0VJZngyODJfN0t5Q3N3RRAB",
                    "relativePublishTimeDescription": "4 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Peaceful and clean shoreline. Family and couple friendly beach. Nice food. We went here during high tide so the waves were looking beautiful and sandbags were being placed in front of shacks.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Peaceful and clean shoreline. Family and couple friendly beach. Nice food. We went here during high tide so the waves were looking beautiful and sandbags were being placed in front of shacks.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Pulkit Dalawat",
                        "uri": "https://www.google.com/maps/contrib/115651011342887237029/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVwa8JOYNkbJNU0_my4n_yvIEwk43LCUmsq6E4Sn4sps_ncNhBQ=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-04T11:33:30.370350Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJZngyODJfN0t5Q3N3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJZngyODJfN0t5Q3N3RRAB!2m1!1s0x3bbfc13c227bd1bf:0xfa2f8c6c1efa4aef"
                },
                {
                    "name": "places/ChIJv9F7IjzBvzsR70r6HmyML_o/reviews/ChZDSUhNMG9nS0VJbkdpNmlEMHEzdWF3EAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 4,
                    "text": {
                        "text": "Nice beach with lots of big waves which make it fun to swim in the water. Unfortunately at some parts of the beach there is a lot of trash laying around. There are  lots of pubs - therefore i didn’t felt like it being a beach to have a nice chill day but more to party and mass tourism.\nStill a nice place to take a dive and enjoy the sun",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Nice beach with lots of big waves which make it fun to swim in the water. Unfortunately at some parts of the beach there is a lot of trash laying around. There are  lots of pubs - therefore i didn’t felt like it being a beach to have a nice chill day but more to party and mass tourism.\nStill a nice place to take a dive and enjoy the sun",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Fabi M.",
                        "uri": "https://www.google.com/maps/contrib/104979838961131485358/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUWsxMUAJi4-QaWyMaP0eXkbRDxqY-pRP-3DwPXTrSJLPCn5d6jPw=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-05-25T03:20:46.848764Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJbkdpNmlEMHEzdWF3EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJbkdpNmlEMHEzdWF3EAE!2m1!1s0x3bbfc13c227bd1bf:0xfa2f8c6c1efa4aef"
                },
                {
                    "name": "places/ChIJv9F7IjzBvzsR70r6HmyML_o/reviews/Ci9DQUlRQUNvZENodHljRjlvT2todlptZExTM000U0cxT2FERXdOWFJNVjFodlFYYxAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 3,
                    "text": {
                        "text": "Visited this beach in Goa recently (May23rd, 2025 )and was surprised by how empty it was. The shoreline was clean and peaceful — perfect if you’re looking to escape the crowds. However, most of the beach shacks and shops were closed, which took away some of the typical Goa vibe. No music, no fresh seafood, no rental services — just a quiet stretch of sand and sea.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Visited this beach in Goa recently (May23rd, 2025 )and was surprised by how empty it was. The shoreline was clean and peaceful — perfect if you’re looking to escape the crowds. However, most of the beach shacks and shops were closed, which took away some of the typical Goa vibe. No music, no fresh seafood, no rental services — just a quiet stretch of sand and sea.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Jeevan M",
                        "uri": "https://www.google.com/maps/contrib/117541003264182218615/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUoqbJhGDnZgV2NhPQKlLes6PvK0D63Bt6B0fWSRNKUjCXSJUU1=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-10T20:32:28.887294531Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2todlptZExTM000U0cxT2FERXdOWFJNVjFodlFYYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2todlptZExTM000U0cxT2FERXdOWFJNVjFodlFYYxAB!2m1!1s0x3bbfc13c227bd1bf:0xfa2f8c6c1efa4aef"
                }
            ]
        },
        {
            "id": "ChIJ952ZWo_AvzsRLoL0JZhAmKM",
            "displayName": {
                "text": "Our Lady of the Immaculate Conception Church",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJ952ZWo_AvzsRLoL0JZhAmKM/reviews/ChdDSUhNMG9nS0VMWGc5cTdxaXFDUF9BRRAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 4,
                    "text": {
                        "text": "I recently visited the Immaculate Conception Church, but it was unfortunately closed for maintenance or other work at the time. I was hoping to see the interior, but even from the outside, the church was quite beautiful. The architecture is impressive, and the peaceful surroundings add to its charm.\npoint:\n\nImmaculate Conception Church is one of the most iconic churches in Goa, known for its stunning white facade and historical significance.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "I recently visited the Immaculate Conception Church, but it was unfortunately closed for maintenance or other work at the time. I was hoping to see the interior, but even from the outside, the church was quite beautiful. The architecture is impressive, and the peaceful surroundings add to its charm.\npoint:\n\nImmaculate Conception Church is one of the most iconic churches in Goa, known for its stunning white facade and historical significance.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "pranjal saxena",
                        "uri": "https://www.google.com/maps/contrib/113363501689809880164/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjX5O9RDAF8HQztyNq07-0Gh-u0DsQMvo1AEW8ewKliyPdTOnOSxSw=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-11T18:40:48.218721Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VMWGc5cTdxaXFDUF9BRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VMWGc5cTdxaXFDUF9BRRAB!2m1!1s0x3bbfc08f5a999df7:0xa398409825f4822e"
                },
                {
                    "name": "places/ChIJ952ZWo_AvzsRLoL0JZhAmKM/reviews/ChZDSUhNMG9nS0VOZXNnZU9Oejh1Z1RREAE",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 3,
                    "text": {
                        "text": "Its a church where tourists come. Saw here the board representing what is modest dress and what is not, still people were not following what was said. The place is clean. In the front of church there is a road where people can sit on dividers(there are benches on it)",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Its a church where tourists come. Saw here the board representing what is modest dress and what is not, still people were not following what was said. The place is clean. In the front of church there is a road where people can sit on dividers(there are benches on it)",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Pulkit Dalawat",
                        "uri": "https://www.google.com/maps/contrib/115651011342887237029/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVwa8JOYNkbJNU0_my4n_yvIEwk43LCUmsq6E4Sn4sps_ncNhBQ=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-11T15:54:09.541536Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VOZXNnZU9Oejh1Z1RREAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VOZXNnZU9Oejh1Z1RREAE!2m1!1s0x3bbfc08f5a999df7:0xa398409825f4822e"
                },
                {
                    "name": "places/ChIJ952ZWo_AvzsRLoL0JZhAmKM/reviews/ChZDSUhNMG9nS0VJQ0FnTUNJdUstUEJ3EAE",
                    "relativePublishTimeDescription": "3 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Church is fantastic but closed for visitors from last 2 plus years.\nSuch a peaceful place.\nAlthpugh it is closed but You can light your candle and make a wish on the premises of the church near stairs",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Church is fantastic but closed for visitors from last 2 plus years.\nSuch a peaceful place.\nAlthpugh it is closed but You can light your candle and make a wish on the premises of the church near stairs",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Safi Jain",
                        "uri": "https://www.google.com/maps/contrib/112398238240457731112/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXfUewplI4PwozsurhdpNKd6DI2y9sJBKAzNARTk5VCeU49RWcm1Q=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-03-30T05:31:00.783339Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNJdUstUEJ3EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNJdUstUEJ3EAE!2m1!1s0x3bbfc08f5a999df7:0xa398409825f4822e"
                },
                {
                    "name": "places/ChIJ952ZWo_AvzsRLoL0JZhAmKM/reviews/ChdDSUhNMG9nS0VOUEdnUEQyOXVHNmtRRRAB",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Immaculate Conception Church is one of the most iconic churches in Goa. No trip to the city can be complete without a visit to this church. We were told that the church was closed to public because some renovation work was going on. Since it is in the middle of the city of Panjim, I wonder how they can shoot it for movies!!!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Immaculate Conception Church is one of the most iconic churches in Goa. No trip to the city can be complete without a visit to this church. We were told that the church was closed to public because some renovation work was going on. Since it is in the middle of the city of Panjim, I wonder how they can shoot it for movies!!!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Tabraz S S",
                        "uri": "https://www.google.com/maps/contrib/110866639723973560281/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVraXwuz5kiKlZeHXAECZ_9pyXzEz7QuXwGnsmYZhviBskNiu4=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-05-27T07:53:11.420638Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VOUEdnUEQyOXVHNmtRRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VOUEdnUEQyOXVHNmtRRRAB!2m1!1s0x3bbfc08f5a999df7:0xa398409825f4822e"
                },
                {
                    "name": "places/ChIJ952ZWo_AvzsRLoL0JZhAmKM/reviews/ChZDSUhNMG9nS0VMNjI2ZF94b0otbU53EAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 4,
                    "text": {
                        "text": "A beautiful and well-known church in the center of Panjim.\nIts white front and big staircase make it one of the most picture-perfect places in Goa.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "A beautiful and well-known church in the center of Panjim.\nIts white front and big staircase make it one of the most picture-perfect places in Goa.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Siraj Ahmed",
                        "uri": "https://www.google.com/maps/contrib/113629468535278655186/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWSPGp6kKIgxiT9MQeQFEC8g2ARwZB-LsbHg8brzPb5Zm3uuJx1FA=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-05-15T21:22:23.526493Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VMNjI2ZF94b0otbU53EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VMNjI2ZF94b0otbU53EAE!2m1!1s0x3bbfc08f5a999df7:0xa398409825f4822e"
                }
            ]
        },
        {
            "id": "ChIJN7gFSwNPvjsRikm58eoZL_U",
            "displayName": {
                "text": "Cola Beach",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJN7gFSwNPvjsRikm58eoZL_U/reviews/ChdDSUhNMG9nS0VJQ0FnTURvNk12N3pBRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Cola Beach is a hidden gem in South Goa, offering a serene and unspoiled environment.\n\nThe beach is nestled between hills, with a freshwater lagoon on one side and the Arabian Sea on the other. The golden sands and swaying coconut palms create a picturesque backdrop, perfect for relaxation and meditation. It’s a peaceful retreat away from the bustling tourist spots.\n\nReaching Cola Beach is an adventure in itself. The last stretch involves navigating a 1.5 km off-road track, which can be challenging for regular vehicles. After parking, there’s a trek of about 800 meters to reach the beach. The approach road is quite rough, so it’s advisable to use a sturdy vehicle or it’s better to trek either opt for a Jeep or an Ecco vehicle, which charges ₹100 per person one-way for the 2 km trip. Despite the effort, the destination is worth it. The beach is clean, less crowded, and offers activities like kayaking in the backwaters. It’s an ideal place for those looking to disconnect and immerse themselves in nature.\n\nThere’s no mobile network connectivity, but the Wi-Fi is decent, You will get option of kayaking on charge of ₹ 800 for 2 persons for 45 mins.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Cola Beach is a hidden gem in South Goa, offering a serene and unspoiled environment.\n\nThe beach is nestled between hills, with a freshwater lagoon on one side and the Arabian Sea on the other. The golden sands and swaying coconut palms create a picturesque backdrop, perfect for relaxation and meditation. It’s a peaceful retreat away from the bustling tourist spots.\n\nReaching Cola Beach is an adventure in itself. The last stretch involves navigating a 1.5 km off-road track, which can be challenging for regular vehicles. After parking, there’s a trek of about 800 meters to reach the beach. The approach road is quite rough, so it’s advisable to use a sturdy vehicle or it’s better to trek either opt for a Jeep or an Ecco vehicle, which charges ₹100 per person one-way for the 2 km trip. Despite the effort, the destination is worth it. The beach is clean, less crowded, and offers activities like kayaking in the backwaters. It’s an ideal place for those looking to disconnect and immerse themselves in nature.\n\nThere’s no mobile network connectivity, but the Wi-Fi is decent, You will get option of kayaking on charge of ₹ 800 for 2 persons for 45 mins.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Ganesh Checkz",
                        "uri": "https://www.google.com/maps/contrib/101094661409274437054/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWjf82XaSCkYL7Jy8I_g8GdOZJytPUFKmr9mA12hFCydoJLwoUW=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-21T17:36:50.277363Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURvNk12N3pBRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURvNk12N3pBRRAB!2m1!1s0x3bbe4f034b05b837:0xf52f19eaf1b9498a"
                },
                {
                    "name": "places/ChIJN7gFSwNPvjsRikm58eoZL_U/reviews/Ci9DQUlRQUNvZENodHljRjlvT21KcFNEUkhiRGhCTjNONVN6QnZhWEV6TldsdWJrRRAB",
                    "relativePublishTimeDescription": "2 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Hidden Gem Beach Experience 🌊\n\nA small yet peaceful beach – truly a hidden gem! You can enjoy kayaking for 45 minutes at just ₹400 for two people. There’s also a beach shack right on the shore – a perfect place to eat and chill. The service might be a bit slow, but the stunning sea view and relaxing vibe make up for it.\n\nIf you're staying in North Goa, definitely plan an entire day for this spot, as it's a bit far and travel takes time. Totally worth it for a serene and rejuvenating experience!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Hidden Gem Beach Experience 🌊\n\nA small yet peaceful beach – truly a hidden gem! You can enjoy kayaking for 45 minutes at just ₹400 for two people. There’s also a beach shack right on the shore – a perfect place to eat and chill. The service might be a bit slow, but the stunning sea view and relaxing vibe make up for it.\n\nIf you're staying in North Goa, definitely plan an entire day for this spot, as it's a bit far and travel takes time. Totally worth it for a serene and rejuvenating experience!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Suraj Parab",
                        "uri": "https://www.google.com/maps/contrib/113682124677935269884/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVbiX0MJRJh5hHfuIM5vo3bePMdu7t_2qQv3clqmFrlObjVEnr5zg=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-14T05:42:17.764084485Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT21KcFNEUkhiRGhCTjNONVN6QnZhWEV6TldsdWJrRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT21KcFNEUkhiRGhCTjNONVN6QnZhWEV6TldsdWJrRRAB!2m1!1s0x3bbe4f034b05b837:0xf52f19eaf1b9498a"
                },
                {
                    "name": "places/ChIJN7gFSwNPvjsRikm58eoZL_U/reviews/ChZDSUhNMG9nS0VQT2g3dVNOb2FUV1FBEAE",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Cola Beach is best known for its serene backwater kayaking experience. The beach is surrounded by lush greenery and features exceptionally clean, crystal-clear waters.\n\nKayaking here is perfect even for beginners, as the water is only about 3–4 feet deep—making it both safe and enjoyable. Swimming in the pristine water is a refreshing experience in itself.\n\nWhile the kayaking route isn’t very long, it offers stunning views of nature and vibrant greenery, making it a paradise for nature lovers.\n\nThe final stretch to reach Cola Beach involves a 2–3 km off-road drive, but it’s easily manageable for both two-wheelers and four-wheelers.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Cola Beach is best known for its serene backwater kayaking experience. The beach is surrounded by lush greenery and features exceptionally clean, crystal-clear waters.\n\nKayaking here is perfect even for beginners, as the water is only about 3–4 feet deep—making it both safe and enjoyable. Swimming in the pristine water is a refreshing experience in itself.\n\nWhile the kayaking route isn’t very long, it offers stunning views of nature and vibrant greenery, making it a paradise for nature lovers.\n\nThe final stretch to reach Cola Beach involves a 2–3 km off-road drive, but it’s easily manageable for both two-wheelers and four-wheelers.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Ketanbhai Desai",
                        "uri": "https://www.google.com/maps/contrib/113724897207672818526/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWu0onLN4DfO2egPZ5ciglz6k-EqolTi3pxMHA_3kI_lDDd1aBD=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-07T15:11:20.990722Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VQT2g3dVNOb2FUV1FBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VQT2g3dVNOb2FUV1FBEAE!2m1!1s0x3bbe4f034b05b837:0xf52f19eaf1b9498a"
                },
                {
                    "name": "places/ChIJN7gFSwNPvjsRikm58eoZL_U/reviews/ChdDSUhNMG9nS0VJQ0FnTURvcmJTQTdnRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "The place is very peaceful and clean. You can just sit and relax watching the waves, and the sunset here looks amazing. The sand is soft and the whole vibe is really calming.\n\nThe best part is you can also go kayaking! It’s a lot of fun. The water is calm, and they give proper life jackets and guide you nicely, so even if it’s your first time, you’ll enjoy it. The colorful kayaks and all the greenery around make it feel even more special.\n\nThere are also a few nice beach shacks where you can chill and eat after kayaking.\nPerfect place for a peaceful trip or some fun adventure. I had a really good time here and would definitely suggest it to anyone visiting",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "The place is very peaceful and clean. You can just sit and relax watching the waves, and the sunset here looks amazing. The sand is soft and the whole vibe is really calming.\n\nThe best part is you can also go kayaking! It’s a lot of fun. The water is calm, and they give proper life jackets and guide you nicely, so even if it’s your first time, you’ll enjoy it. The colorful kayaks and all the greenery around make it feel even more special.\n\nThere are also a few nice beach shacks where you can chill and eat after kayaking.\nPerfect place for a peaceful trip or some fun adventure. I had a really good time here and would definitely suggest it to anyone visiting",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Shubh Borkar",
                        "uri": "https://www.google.com/maps/contrib/105234235603219529213/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXLA308tzSelvvHCoFFHI2TmGeHwmxZN80953nXLVnCFvvzHjiWqg=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-27T15:15:08.230177Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURvcmJTQTdnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURvcmJTQTdnRRAB!2m1!1s0x3bbe4f034b05b837:0xf52f19eaf1b9498a"
                },
                {
                    "name": "places/ChIJN7gFSwNPvjsRikm58eoZL_U/reviews/ChdDSUhNMG9nS0VJeUQ5OGlXam9xdnh3RRAB",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 3,
                    "text": {
                        "text": "Cola Beach offers stunning views and a peaceful, less-crowded vibe, which is great if you're looking to relax and take in the scenery. The combination of the lagoon and the sea is unique and definitely photo-worthy.\n\nHowever, it's more of a \"look but don't touch\" kind of spot. The water wasn’t safe or inviting enough to step into during my visit — the waves were strong, and there were no lifeguards around. Also, access to the beach is a bit tricky, with rough roads leading in.\n\nOverall, a nice place for a quiet stroll or to admire nature, but not ideal if you’re hoping to swim or spend time in the water.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Cola Beach offers stunning views and a peaceful, less-crowded vibe, which is great if you're looking to relax and take in the scenery. The combination of the lagoon and the sea is unique and definitely photo-worthy.\n\nHowever, it's more of a \"look but don't touch\" kind of spot. The water wasn’t safe or inviting enough to step into during my visit — the waves were strong, and there were no lifeguards around. Also, access to the beach is a bit tricky, with rough roads leading in.\n\nOverall, a nice place for a quiet stroll or to admire nature, but not ideal if you’re hoping to swim or spend time in the water.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Lakshmi G",
                        "uri": "https://www.google.com/maps/contrib/107540103532111947902/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUfk_MVDDjLo7RZqw3x2BjstKeS00kfophbTjyP8aumAOeApnw=s128-c0x00000000-cc-rp-mo"
                    },
                    "publishTime": "2025-05-19T03:08:01.238838Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJeUQ5OGlXam9xdnh3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJeUQ5OGlXam9xdnh3RRAB!2m1!1s0x3bbe4f034b05b837:0xf52f19eaf1b9498a"
                }
            ]
        },
        {
            "id": "ChIJCxPb6YvAvzsR4M4pLK1AJV4",
            "displayName": {
                "text": "Big Daddy Casino",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJCxPb6YvAvzsR4M4pLK1AJV4/reviews/Ci9DQUlRQUNvZENodHljRjlvT2xSbk0ycGtlbGhZYmpBelFsWk9aMWR5V2padU1YYxAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Big Daddy Casino is one of the better casinos in Goa. Yes, it’s on the expensive side, but you truly get what you pay for—great entertainment, a clean and hygienic environment, and professional service.\n\nOne thing I really appreciated was the dedicated kids’ room on the ground floor, which is a thoughtful touch for families. They also take your food order before you enter the main casino area, which keeps things organized.\n\nThank you, Big Daddy Casino, for a great experience!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Big Daddy Casino is one of the better casinos in Goa. Yes, it’s on the expensive side, but you truly get what you pay for—great entertainment, a clean and hygienic environment, and professional service.\n\nOne thing I really appreciated was the dedicated kids’ room on the ground floor, which is a thoughtful touch for families. They also take your food order before you enter the main casino area, which keeps things organized.\n\nThank you, Big Daddy Casino, for a great experience!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Daxesh Patel",
                        "uri": "https://www.google.com/maps/contrib/105897260567195201090/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocJD3LyBPmgOd1EN5rf9PppMNhSu9xfoWEbkH2C50rl9Rano1A=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-11T17:52:09.553278622Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2xSbk0ycGtlbGhZYmpBelFsWk9aMWR5V2padU1YYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2xSbk0ycGtlbGhZYmpBelFsWk9aMWR5V2padU1YYxAB!2m1!1s0x3bbfc08be9db130b:0x5e2540ad2c29cee0"
                },
                {
                    "name": "places/ChIJCxPb6YvAvzsR4M4pLK1AJV4/reviews/ChZDSUhNMG9nS0VJQ0FnTUNnZ3ByS0t3EAE",
                    "relativePublishTimeDescription": "4 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Big Daddy Casino boasts a luxurious and well-maintained environment. The decor is stylish, with modern gaming floors, a dedicated VIP lounge, and comfortable seating arrangements. The cruise-style setup enhances the overall experience, providing a mix of entertainment, dining, and gaming in one place.\n\nBig Daddy Casino is not just about gaming; it also offers live entertainment, including DJ nights, dance performances, and music acts. The hospitality is commendable, with courteous staff and prompt service. The casino also has an in-house restaurant and bar serving a variety of cuisines, including Indian, Continental, and Chinese dishes.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Big Daddy Casino boasts a luxurious and well-maintained environment. The decor is stylish, with modern gaming floors, a dedicated VIP lounge, and comfortable seating arrangements. The cruise-style setup enhances the overall experience, providing a mix of entertainment, dining, and gaming in one place.\n\nBig Daddy Casino is not just about gaming; it also offers live entertainment, including DJ nights, dance performances, and music acts. The hospitality is commendable, with courteous staff and prompt service. The casino also has an in-house restaurant and bar serving a variety of cuisines, including Indian, Continental, and Chinese dishes.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Mavenick Sharma",
                        "uri": "https://www.google.com/maps/contrib/114691111114141295481/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjX5TA_M47TjshQ6HRdEXEAJKC_OB4B0zV3LNbj2G4v3U1B2GCMs=s128-c0x00000000-cc-rp-mo-ba3"
                    },
                    "publishTime": "2025-02-15T07:54:36.940345Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNnZ3ByS0t3EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNnZ3ByS0t3EAE!2m1!1s0x3bbfc08be9db130b:0x5e2540ad2c29cee0"
                },
                {
                    "name": "places/ChIJCxPb6YvAvzsR4M4pLK1AJV4/reviews/ChZDSUhNMG9nS0VJQ0FnTUNZNG95M1hBEAE",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 4,
                    "text": {
                        "text": "Good Casino\n\nNeed to improve on the service and food quality\n\nMost costlier but No Valet Parking ! Regular entry 3500₹ with 1000₹ one time cash with no access to sky deck. Performances are limited with more wait time and less actual performances.\n\nThey need to improve the parking service, you can’t do shuttle everytime which is around 1.3kms away.\n\nTable service lags and is insufficient\nJust a one time service",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Good Casino\n\nNeed to improve on the service and food quality\n\nMost costlier but No Valet Parking ! Regular entry 3500₹ with 1000₹ one time cash with no access to sky deck. Performances are limited with more wait time and less actual performances.\n\nThey need to improve the parking service, you can’t do shuttle everytime which is around 1.3kms away.\n\nTable service lags and is insufficient\nJust a one time service",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Pankaj Hogade",
                        "uri": "https://www.google.com/maps/contrib/118152259325562840274/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWISe6J-xOTDWmg-T2IsNHBAsXYxHKRgpMwXe_sMrti8dHSCaifwA=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-05-02T21:57:46.853848Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNZNG95M1hBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNZNG95M1hBEAE!2m1!1s0x3bbfc08be9db130b:0x5e2540ad2c29cee0"
                },
                {
                    "name": "places/ChIJCxPb6YvAvzsR4M4pLK1AJV4/reviews/ChdDSUhNMG9nS0VJQ0FnTURvdE9HNF9nRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 1,
                    "text": {
                        "text": "Definitely not worth the money. The cruise is outdated, the toilets are in terrible condition, and the overall maintenance is poor. If you're playing solo, be cautious—they seem to target individuals to drain your wallet fast. Within 10 minutes, you're practically forced out of the game. Compared to Deltin Royale, Big Daddy falls way short. The only thing that keeps people around is the pole dancing on every floor, not the gaming experience. I would choose Deltin Royale over Big Daddy any day.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Definitely not worth the money. The cruise is outdated, the toilets are in terrible condition, and the overall maintenance is poor. If you're playing solo, be cautious—they seem to target individuals to drain your wallet fast. Within 10 minutes, you're practically forced out of the game. Compared to Deltin Royale, Big Daddy falls way short. The only thing that keeps people around is the pole dancing on every floor, not the gaming experience. I would choose Deltin Royale over Big Daddy any day.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Danamjaya reddy",
                        "uri": "https://www.google.com/maps/contrib/106899984373151536757/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUNBIDlxa9zzsiPCKtmoMMCnPNTf92a2_738UybjYe-regNPPf7Fw=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-22T08:27:00.627590Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURvdE9HNF9nRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURvdE9HNF9nRRAB!2m1!1s0x3bbfc08be9db130b:0x5e2540ad2c29cee0"
                },
                {
                    "name": "places/ChIJCxPb6YvAvzsR4M4pLK1AJV4/reviews/ChdDSUhNMG9nS0VJQ0FnTURJMTl2Yjd3RRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 4,
                    "text": {
                        "text": "Goa is best known for its casinos. Big Daddy is one of them. Entry tickets started from rupees 3500 to 6000 per person at night time and 2000 rupees at day time. In this price boat ride in mandovi river to reach casino is included. You can enjoy unlimited food and hard drinks in this price. You can also get 1000 rupees worth coins to play casino games.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Goa is best known for its casinos. Big Daddy is one of them. Entry tickets started from rupees 3500 to 6000 per person at night time and 2000 rupees at day time. In this price boat ride in mandovi river to reach casino is included. You can enjoy unlimited food and hard drinks in this price. You can also get 1000 rupees worth coins to play casino games.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "multani mohsin",
                        "uri": "https://www.google.com/maps/contrib/101895614988586760537/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocJnmnVOjQUFAxyJP_DUzBDr7N4oFV0EQcMwvQSwXDULyp-Mlg=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-13T16:59:36.588907Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURJMTl2Yjd3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURJMTl2Yjd3RRAB!2m1!1s0x3bbfc08be9db130b:0x5e2540ad2c29cee0"
                }
            ]
        },
        {
            "id": "ChIJy1w3MFFJvjsRW9cTbAchLdw",
            "displayName": {
                "text": "Cabo de Rama Fort",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJy1w3MFFJvjsRW9cTbAchLdw/reviews/Ci9DQUlRQUNvZENodHljRjlvT2xCTmVXdEJjR0pwVlZsRFFsOHRhRlF4WVhKVVFuYxAB",
                    "relativePublishTimeDescription": "in the last week",
                    "rating": 4,
                    "text": {
                        "text": "Fort has nothing much to see inside, but the location of the fort gives a very good vantage point to view the Arabian sea and pebble beach from above. Overall it is a good place to visit, the best time to visit is November to April.\n\nPlease be careful of the rocks that protrude from the pavement while walking inside the fort. You may get injured if any mis-step is taken! ⚠️",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Fort has nothing much to see inside, but the location of the fort gives a very good vantage point to view the Arabian sea and pebble beach from above. Overall it is a good place to visit, the best time to visit is November to April.\n\nPlease be careful of the rocks that protrude from the pavement while walking inside the fort. You may get injured if any mis-step is taken! ⚠️",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Arindam Debnath",
                        "uri": "https://www.google.com/maps/contrib/117276694148402736488/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWUX9yP8PudpzAeHsZ7XpAhjb0DwNMRuJqNam5nhu1fLO2ddXs2=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-27T12:46:09.027215997Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2xCTmVXdEJjR0pwVlZsRFFsOHRhRlF4WVhKVVFuYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2xCTmVXdEJjR0pwVlZsRFFsOHRhRlF4WVhKVVFuYxAB!2m1!1s0x3bbe495130375ccb:0xdc2d21076c13d75b"
                },
                {
                    "name": "places/ChIJy1w3MFFJvjsRW9cTbAchLdw/reviews/ChdDSUhNMG9nS0VJQ0FnTURJcDRYN2xBRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Stunning views of the Arabian sea. Nice place to spend quiet hours and click photos. Includes a beautiful church and pebble beach too if you want to experience the crashing sea waves up close. No entry fee. Not much of the fort left anymore.\n\nMust visit to see nature's beauty. Please do not litter here. Kindly take your picnic trash with you and dispose it off responsibly.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Stunning views of the Arabian sea. Nice place to spend quiet hours and click photos. Includes a beautiful church and pebble beach too if you want to experience the crashing sea waves up close. No entry fee. Not much of the fort left anymore.\n\nMust visit to see nature's beauty. Please do not litter here. Kindly take your picnic trash with you and dispose it off responsibly.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Pooja Singh",
                        "uri": "https://www.google.com/maps/contrib/101721298363950160671/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocIwmTgLSVNO6LsuU7rvJJ_qecHgqIyA6yjhpe60prKdszlo-e6L=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-04-13T13:30:49.781129Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURJcDRYN2xBRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURJcDRYN2xBRRAB!2m1!1s0x3bbe495130375ccb:0xdc2d21076c13d75b"
                },
                {
                    "name": "places/ChIJy1w3MFFJvjsRW9cTbAchLdw/reviews/ChdDSUhNMG9nS0VJQ0FnTURJb3NXMmhRRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "The much Famous Cabo De Rama Fort is located in the Cancona region.\nA very famous point of visit in South Goa region, it offers stunning sea + Sunset views, is a truly paradise offering for its visitors, that too without paying any cost.\nAlso it hosts the hidden gem of South Goa, Pebble Beach inside it. Also if you love exploring, walk along its boundary wall and you'll get to see beautiful sea & cliff views.\nAlso adhere to the timings mentioned on Google, as it opens for fixed hours.\nPS:- Make sure to visit the Fort & Pebble Beach for stunning views.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "The much Famous Cabo De Rama Fort is located in the Cancona region.\nA very famous point of visit in South Goa region, it offers stunning sea + Sunset views, is a truly paradise offering for its visitors, that too without paying any cost.\nAlso it hosts the hidden gem of South Goa, Pebble Beach inside it. Also if you love exploring, walk along its boundary wall and you'll get to see beautiful sea & cliff views.\nAlso adhere to the timings mentioned on Google, as it opens for fixed hours.\nPS:- Make sure to visit the Fort & Pebble Beach for stunning views.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Vaibhav Jain",
                        "uri": "https://www.google.com/maps/contrib/117299102484604813950/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocKaN4qxMak3r0WG4EmIenHz175wNG55Xv-OTzDEMpZgJ5BulQ=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-04-08T10:48:08.576167Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURJb3NXMmhRRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURJb3NXMmhRRRAB!2m1!1s0x3bbe495130375ccb:0xdc2d21076c13d75b"
                },
                {
                    "name": "places/ChIJy1w3MFFJvjsRW9cTbAchLdw/reviews/Ci9DQUlRQUNvZENodHljRjlvT25GRU1FRlNXa0UyTjNaek9WRTBURlZwYWtNNFMwRRAB",
                    "relativePublishTimeDescription": "2 weeks ago",
                    "rating": 4,
                    "text": {
                        "text": "Definitely something that should be on your checklist when you're in Goa! The view is just amazing and the pebbles beach is one of the most beautiful beaches I've seen on this trip!! Definitely something to visit during the monsoon season!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Definitely something that should be on your checklist when you're in Goa! The view is just amazing and the pebbles beach is one of the most beautiful beaches I've seen on this trip!! Definitely something to visit during the monsoon season!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "gayatri",
                        "uri": "https://www.google.com/maps/contrib/116724898577027312779/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWvkDDzc1y7cqPsI3ZszUKiO9SZLw8_5As1MOY0TirZr0zT1O7v5Q=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-17T12:58:49.783183082Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT25GRU1FRlNXa0UyTjNaek9WRTBURlZwYWtNNFMwRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT25GRU1FRlNXa0UyTjNaek9WRTBURlZwYWtNNFMwRRAB!2m1!1s0x3bbe495130375ccb:0xdc2d21076c13d75b"
                },
                {
                    "name": "places/ChIJy1w3MFFJvjsRW9cTbAchLdw/reviews/ChZDSUhNMG9nS0VJQ0FnTUNJbHNDWWZnEAE",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 4,
                    "text": {
                        "text": "Drove a long way to reach the fort and the views from the top was worth it. Road is being widened and hopefully that will attract tourists. For new drivers the curves of the hill will test their skill. To see the sunset reach by 5 pm. No entry after that. Nothing spectacular about the fort though but the  beach is worth visiting. It's a long way down though but would recommend to visit Cola /Khola beach nearby",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Drove a long way to reach the fort and the views from the top was worth it. Road is being widened and hopefully that will attract tourists. For new drivers the curves of the hill will test their skill. To see the sunset reach by 5 pm. No entry after that. Nothing spectacular about the fort though but the  beach is worth visiting. It's a long way down though but would recommend to visit Cola /Khola beach nearby",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "DR GEETIN MURMU",
                        "uri": "https://www.google.com/maps/contrib/103256654668214051805/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjU95LDlNUaYPjGLRwRHCZii0JtYferUuSBUSnkz9yRLZyn-rAd1vw=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-15T02:51:14.221661Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNJbHNDWWZnEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNJbHNDWWZnEAE!2m1!1s0x3bbe495130375ccb:0xdc2d21076c13d75b"
                }
            ]
        },
        {
            "id": "ChIJEfYjUmPBvzsRZ7HsUoMFXIk",
            "displayName": {
                "text": "Fontainhas",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJEfYjUmPBvzsRZ7HsUoMFXIk/reviews/ChdDSUhNMG9nS0VLckQzZGpMdjVpT3lnRRAB",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "I really enjoyed visiting Fontainhas, it felt like stepping into a completely different world. The colorful Portuguese style houses, narrow lanes, and charming old world vibe make it a truly unique experience in Goa.\n\nWalking through the area gives you a sense of history and culture that's both peaceful and refreshing. Perfect for photography, a relaxed stroll, or simply soaking in the heritage atmosphere.\n\nDefinitely worth a visit if you want to experience a different side of Goa beyond the beaches.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "I really enjoyed visiting Fontainhas, it felt like stepping into a completely different world. The colorful Portuguese style houses, narrow lanes, and charming old world vibe make it a truly unique experience in Goa.\n\nWalking through the area gives you a sense of history and culture that's both peaceful and refreshing. Perfect for photography, a relaxed stroll, or simply soaking in the heritage atmosphere.\n\nDefinitely worth a visit if you want to experience a different side of Goa beyond the beaches.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Avinash Varma Kalidindi",
                        "uri": "https://www.google.com/maps/contrib/103342528676246764163/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWqjVfh0Nqs_OfFGnaA_uioeGJtuMEq1LRI_zmskwjfmQ6ragpoKA=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-05-31T13:44:54.442204Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VLckQzZGpMdjVpT3lnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VLckQzZGpMdjVpT3lnRRAB!2m1!1s0x3bbfc1635223f611:0x895c058352ecb167"
                },
                {
                    "name": "places/ChIJEfYjUmPBvzsRZ7HsUoMFXIk/reviews/ChZDSUhNMG9nS0VKNnFqWWZ0LXVXUkdBEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "We visited Fontainhas, and the area was absolutely beautiful. It might have been the off-season, but the lack of crowds made the experience even more peaceful and enjoyable. The charming Portuguese-style buildings are stunning and give the area a unique character. Since it’s a residential neighborhood, it feels calm and authentic. We managed to get some great pictures too. Definitely a must-visit spot in Goa!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "We visited Fontainhas, and the area was absolutely beautiful. It might have been the off-season, but the lack of crowds made the experience even more peaceful and enjoyable. The charming Portuguese-style buildings are stunning and give the area a unique character. Since it’s a residential neighborhood, it feels calm and authentic. We managed to get some great pictures too. Definitely a must-visit spot in Goa!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Vyshnavi L R",
                        "uri": "https://www.google.com/maps/contrib/116644127724785758486/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWn1VAC-1xFqquwmuvpMNnQIrkfBAc5h3-fBLH5VV4mLIP9HNj4QA=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-05-30T14:49:29.786595Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VKNnFqWWZ0LXVXUkdBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VKNnFqWWZ0LXVXUkdBEAE!2m1!1s0x3bbfc1635223f611:0x895c058352ecb167"
                },
                {
                    "name": "places/ChIJEfYjUmPBvzsRZ7HsUoMFXIk/reviews/Ci9DQUlRQUNvZENodHljRjlvT25SUlNUSk1SWFp5UWpneVVWcE1NekJsWjFKMldWRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 4,
                    "text": {
                        "text": "☀️Fontainhas is a historic Place in Panaji-Goa, known for its well preserved Portuguese architecture and vibrant atmosphere.\n\n☀️Fontainhas is usually very crowded on weekends.\n\n☀️Avoid bringing your car in, instead trying walking(best actually).",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "☀️Fontainhas is a historic Place in Panaji-Goa, known for its well preserved Portuguese architecture and vibrant atmosphere.\n\n☀️Fontainhas is usually very crowded on weekends.\n\n☀️Avoid bringing your car in, instead trying walking(best actually).",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Om Mayekar",
                        "uri": "https://www.google.com/maps/contrib/108034619003960097495/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXLrNL8ok0Py2Aj0-g2B3--la_Jv1nPd1Zbpue9Gz4G3Tiwmp_N=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-25T06:42:03.777828596Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT25SUlNUSk1SWFp5UWpneVVWcE1NekJsWjFKMldWRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT25SUlNUSk1SWFp5UWpneVVWcE1NekJsWjFKMldWRRAB!2m1!1s0x3bbfc1635223f611:0x895c058352ecb167"
                },
                {
                    "name": "places/ChIJEfYjUmPBvzsRZ7HsUoMFXIk/reviews/ChdDSUhNMG9nS0VQalM5X0xfanJLVWt3RRAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Fontanas has an amazing colourful architecture which depicts the true style of Portuguese. You won’t feel you are in India. You will feel like you are in Portugal. The place has a very good and positive vibe. I went to this place at dusk, and it was truly amazing when you could feel the golden colour of sun rays falling on these buildings, making them look amazing.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Fontanas has an amazing colourful architecture which depicts the true style of Portuguese. You won’t feel you are in India. You will feel like you are in Portugal. The place has a very good and positive vibe. I went to this place at dusk, and it was truly amazing when you could feel the golden colour of sun rays falling on these buildings, making them look amazing.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Satyajit Das",
                        "uri": "https://www.google.com/maps/contrib/103368364077840519516/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWKikbasohnrf40jRpicMDkPozqErnVBYoT_KeNJKLCI-vcSsIj=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-12T13:33:25.205969Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VQalM5X0xfanJLVWt3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VQalM5X0xfanJLVWt3RRAB!2m1!1s0x3bbfc1635223f611:0x895c058352ecb167"
                },
                {
                    "name": "places/ChIJEfYjUmPBvzsRZ7HsUoMFXIk/reviews/ChZDSUhNMG9nS0VLcXJ2YU9WeWZyZlB3EAE",
                    "relativePublishTimeDescription": "4 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Fontainhas is like stepping into a different world—peaceful, colorful, and full of old-world charm. The Portuguese-style houses, bright walls, and narrow lanes feel like you're walking through a living postcard. Every corner has its own story.\n\nIt’s perfect for a relaxed stroll, taking photos, or just soaking in the calm vibe away from the usual beach crowd. Don’t miss the art galleries, local bakeries, and small cafés tucked inside the heritage homes.\n\nIf you want to experience Goa beyond the beaches and parties, Fontainhas is a must-visit.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Fontainhas is like stepping into a different world—peaceful, colorful, and full of old-world charm. The Portuguese-style houses, bright walls, and narrow lanes feel like you're walking through a living postcard. Every corner has its own story.\n\nIt’s perfect for a relaxed stroll, taking photos, or just soaking in the calm vibe away from the usual beach crowd. Don’t miss the art galleries, local bakeries, and small cafés tucked inside the heritage homes.\n\nIf you want to experience Goa beyond the beaches and parties, Fontainhas is a must-visit.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Saurav Das",
                        "uri": "https://www.google.com/maps/contrib/107317624996188504019/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWL8VXLXzLFumB6JMh-jgqe02fpb48hWmadpKc0biAk6D4jRwEN=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-06T05:02:32.426811Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VLcXJ2YU9WeWZyZlB3EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VLcXJ2YU9WeWZyZlB3EAE!2m1!1s0x3bbfc1635223f611:0x895c058352ecb167"
                }
            ]
        },
        {
            "id": "ChIJKTeIBk4IvzsRekHQksfEKkk",
            "displayName": {
                "text": "Kadamba Shri Mahadeva Temple, Tambdisurla",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJKTeIBk4IvzsRekHQksfEKkk/reviews/ChZDSUhNMG9nS0VLdTE2cXF5MXRqSlNBEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "This is a fascinating destination for nature lovers, history lovers and the adventurers. As also those who are religious(primarily hindus) since they worship God in this temple. This monument has been declared to be of national importance. The journey to this location situated in the Mollem wildlife Sanctuary was breathtakingly beautiful with lush forests on both sides of the road. The road which connnects from the main road to the location was well maintained and was a pleasure to drive on. We started off from Mapusa and proceeded to Valpoi via Bicholim,  proceeeded to Honda and took a right turn at Cotorem untill we reached a junction where we took a left turn. From here the temple is about 15 km. The road is straight and simple. The road is being widened at certain areas. If you continue going right for another 7 kms you ll reach Bondla Park and Zoo. It took us about 90 minutes or so to reach this destination. A charge of Rs 10 per person is charged by the Panchayat at the Mollem Sanctuary checkpost to take care of the Garbage as per their receipt. On the way there were ample opportunities to click beautiful pics with mother Nature. Upon arrival you are greeted with the locals selling fresh flowers to offer to the temple diety for 30 Rs a bunch. (Wrapped by a large leaf) an eco-sensitive move. There are a couple of eateries which serve you lime soda, softdrinks etc. Along with mirchi pav, kaapa, maggi noodles etc. There is a small bridge built over the stream and after a brief walk (100m) leads to the main temple arena. Chilled and hygienic drinking water is available. There is toilet facility. Car parking is ample but on weekends it could get a little crowded. In one corner of the area there this beautiful tree with Pajakta flowers(parijat) also known as night jasmine. It made a pretty sight. There is provision to light incense sticks. The structure of the temple is made up of Basalt rock and has multiple carvings which has been detailed by the ASI(Goa circle) in its information board at the entrance. This place would be worth visiting during rains however no swimming is permitted which can lead to fatalities. A warning board is seen to that effect. We were lucky to see small animals such as the cute monkeys who eat from your hand, monitor lizards and mongooses. we also saw a school, a health centre and a car workshop in this area. Looks like things are developing in this area. All in all this temple is a masterpiece of a carving dated to 12th or 13th sanctuary. While in Goa, put this destination on your travel list!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "This is a fascinating destination for nature lovers, history lovers and the adventurers. As also those who are religious(primarily hindus) since they worship God in this temple. This monument has been declared to be of national importance. The journey to this location situated in the Mollem wildlife Sanctuary was breathtakingly beautiful with lush forests on both sides of the road. The road which connnects from the main road to the location was well maintained and was a pleasure to drive on. We started off from Mapusa and proceeded to Valpoi via Bicholim,  proceeeded to Honda and took a right turn at Cotorem untill we reached a junction where we took a left turn. From here the temple is about 15 km. The road is straight and simple. The road is being widened at certain areas. If you continue going right for another 7 kms you ll reach Bondla Park and Zoo. It took us about 90 minutes or so to reach this destination. A charge of Rs 10 per person is charged by the Panchayat at the Mollem Sanctuary checkpost to take care of the Garbage as per their receipt. On the way there were ample opportunities to click beautiful pics with mother Nature. Upon arrival you are greeted with the locals selling fresh flowers to offer to the temple diety for 30 Rs a bunch. (Wrapped by a large leaf) an eco-sensitive move. There are a couple of eateries which serve you lime soda, softdrinks etc. Along with mirchi pav, kaapa, maggi noodles etc. There is a small bridge built over the stream and after a brief walk (100m) leads to the main temple arena. Chilled and hygienic drinking water is available. There is toilet facility. Car parking is ample but on weekends it could get a little crowded. In one corner of the area there this beautiful tree with Pajakta flowers(parijat) also known as night jasmine. It made a pretty sight. There is provision to light incense sticks. The structure of the temple is made up of Basalt rock and has multiple carvings which has been detailed by the ASI(Goa circle) in its information board at the entrance. This place would be worth visiting during rains however no swimming is permitted which can lead to fatalities. A warning board is seen to that effect. We were lucky to see small animals such as the cute monkeys who eat from your hand, monitor lizards and mongooses. we also saw a school, a health centre and a car workshop in this area. Looks like things are developing in this area. All in all this temple is a masterpiece of a carving dated to 12th or 13th sanctuary. While in Goa, put this destination on your travel list!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Julius Fds",
                        "uri": "https://www.google.com/maps/contrib/111741683115529699498/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUqbsCVWv_hD4LH4fRiV1n46tkpRYBwENXAsVAL_QkNz0iV29sA=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-05-16T16:23:46.207529Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VLdTE2cXF5MXRqSlNBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VLdTE2cXF5MXRqSlNBEAE!2m1!1s0x3bbf084e06883729:0x492ac4c792d0417a"
                },
                {
                    "name": "places/ChIJKTeIBk4IvzsRekHQksfEKkk/reviews/ChZDSUhNMG9nS0VQakltWTNQcE92QUdnEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Such a serene place this is. The vibe here is so calming and the environment feels perfect. Its a beautiful place and a must visit if you are a person who likes exploring Goa’s culture and heritage and want to witness a glorious site. And yes there are mini waterfalls around of which you can catch a glimpse but getting into water is not allowed.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Such a serene place this is. The vibe here is so calming and the environment feels perfect. Its a beautiful place and a must visit if you are a person who likes exploring Goa’s culture and heritage and want to witness a glorious site. And yes there are mini waterfalls around of which you can catch a glimpse but getting into water is not allowed.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Dipishka Harmalkar",
                        "uri": "https://www.google.com/maps/contrib/111927780546712362441/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocIMXOLaGfyfxnuKuRo2ZbgEKcvQgb16hWmgRwO1u_ZsXqKU7vEb=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-01T17:46:11.119233Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VQakltWTNQcE92QUdnEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VQakltWTNQcE92QUdnEAE!2m1!1s0x3bbf084e06883729:0x492ac4c792d0417a"
                },
                {
                    "name": "places/ChIJKTeIBk4IvzsRekHQksfEKkk/reviews/ChdDSUhNMG9nS0VJQ0FnTURvMWR6SnhBRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "An ancient temple right in between a dense forest area.\nIt is located quite a distance from many attractions in Goa so it is missed by many.\nVery nice place for great Insta stories and reels.\nAs it is not explored fully many folks miss this gem.\nIt has a very calm atmosphere and some monkeys around the place.\nThe route can be tricky to get there as you will loose your mobile network at many places.\nThey have placed boards to help out.\nThe road is very well done and feels like a good ride.\nThere are a few bridges close by as well which are great places for reels.\nThe place in itself is about 2 hours ride from many beaches of Goa.\nThe place has a few small shops close by and it costs about 10-30 Rs per person to enter via the forest area.\nAs there is no network carry cash.\nThe place can be in your list if you are on to visit hidden gems and want to do a good reel or shorts.\nEven if you are not into the reels or shorts and want to see some place different from\nthe routine then must visit this place.\nThe close by attraction is the Dudhsagar falls. Best to get here by the bike or taxi.\nAnd plan to get there by noon as by evening they close the entry to the place as it’s located in the forest areas.\nSo if you want to visit plan get there by 3 and plan for 2 hours journey by road",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "An ancient temple right in between a dense forest area.\nIt is located quite a distance from many attractions in Goa so it is missed by many.\nVery nice place for great Insta stories and reels.\nAs it is not explored fully many folks miss this gem.\nIt has a very calm atmosphere and some monkeys around the place.\nThe route can be tricky to get there as you will loose your mobile network at many places.\nThey have placed boards to help out.\nThe road is very well done and feels like a good ride.\nThere are a few bridges close by as well which are great places for reels.\nThe place in itself is about 2 hours ride from many beaches of Goa.\nThe place has a few small shops close by and it costs about 10-30 Rs per person to enter via the forest area.\nAs there is no network carry cash.\nThe place can be in your list if you are on to visit hidden gems and want to do a good reel or shorts.\nEven if you are not into the reels or shorts and want to see some place different from\nthe routine then must visit this place.\nThe close by attraction is the Dudhsagar falls. Best to get here by the bike or taxi.\nAnd plan to get there by noon as by evening they close the entry to the place as it’s located in the forest areas.\nSo if you want to visit plan get there by 3 and plan for 2 hours journey by road",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Nagarajan T V",
                        "uri": "https://www.google.com/maps/contrib/113821775492617164549/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWRj6OlkXF_aUsMuKyRDJ_EXcLinmg1lU7yMoPMCFbbcJzo2nVJxQ=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-04-27T06:13:37.635118Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURvMWR6SnhBRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURvMWR6SnhBRRAB!2m1!1s0x3bbf084e06883729:0x492ac4c792d0417a"
                },
                {
                    "name": "places/ChIJKTeIBk4IvzsRekHQksfEKkk/reviews/ChZDSUhNMG9nS0VJQ0FnSURfOXQyU1NBEAE",
                    "relativePublishTimeDescription": "5 months ago",
                    "rating": 5,
                    "text": {
                        "text": "I'm still in awe of my visit to this ancient Shiv temple in Goa! Although it's a bit of a trek from the city, the breathtaking views and serene atmosphere make it an absolute must-visit.\n\nAs you step inside, you'll instantly feel a deep sense of peace and connection. Be sure to download the road map before heading out, as the roads can get a bit tricky.\n\nA hidden gem along the way is a small, family-run eatery about 20 minutes before the temple. Their homemade kokam water is an absolute delight!\n\nOn our return journey, we were blessed with a rare sighting of a cobra snake crossing our path. A gentle reminder to drive safely and respectfully through nature.\n\nI'm thrilled to share my experience, and I highly recommend adding this temple to your Goa itinerary!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "I'm still in awe of my visit to this ancient Shiv temple in Goa! Although it's a bit of a trek from the city, the breathtaking views and serene atmosphere make it an absolute must-visit.\n\nAs you step inside, you'll instantly feel a deep sense of peace and connection. Be sure to download the road map before heading out, as the roads can get a bit tricky.\n\nA hidden gem along the way is a small, family-run eatery about 20 minutes before the temple. Their homemade kokam water is an absolute delight!\n\nOn our return journey, we were blessed with a rare sighting of a cobra snake crossing our path. A gentle reminder to drive safely and respectfully through nature.\n\nI'm thrilled to share my experience, and I highly recommend adding this temple to your Goa itinerary!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Chinchu Nair",
                        "uri": "https://www.google.com/maps/contrib/113369824509711324348/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVV1APg52PLL9x6iocwGcExSJ45n3nRsDaziXh-pUb9WoFSnjnQ=s128-c0x00000000-cc-rp-mo-ba2"
                    },
                    "publishTime": "2025-01-24T05:15:18.908374Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnSURfOXQyU1NBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnSURfOXQyU1NBEAE!2m1!1s0x3bbf084e06883729:0x492ac4c792d0417a"
                },
                {
                    "name": "places/ChIJKTeIBk4IvzsRekHQksfEKkk/reviews/ChZDSUhNMG9nS0VMR2tsUGIxaDYyOEZREAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Got an opportunity to visit the special place in Goa. The place is fascinating destination for nature lovers, history lovers and the adventurers. Goa is not just for beaches it also has very unique places like this one. If visiting Goa then please take the time and do visit the temple surely you love the peace, beauty, greenery and views. From Baga or Calangute it will take around 2 to 2.5 hours to reach.\nAlso please make sure to full scooty tanks as there are very less petrol pumps that to on highway once you left the highway and entered the local road its just beautiful views with lush greenery. The road is straight and simple. The road is being widened at certain areas.\nThe structure of the temple is made up of Basalt rock and has multiple carvings which has been detailed by the ASI(Goa circle) in its information board at the entrance. This place would be worth visiting during rains however no swimming is permitted which can lead to fatalities. A warning board is seen to that effect. We were lucky to see small animals such as the cute monkeys who eat from your hand.\nAll in all this temple is a masterpiece of a carving dated to 12th or 13th sanctuary.\n\nParking - Ample amount of parking is there.\nTimings - Try to visit the temple before 05:30 PM and don’t stay after sunset as the place is very quiet and you barely see the vehicle until you touch the highway.\nFood - Yes the snacks are available\noutside the temple",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Got an opportunity to visit the special place in Goa. The place is fascinating destination for nature lovers, history lovers and the adventurers. Goa is not just for beaches it also has very unique places like this one. If visiting Goa then please take the time and do visit the temple surely you love the peace, beauty, greenery and views. From Baga or Calangute it will take around 2 to 2.5 hours to reach.\nAlso please make sure to full scooty tanks as there are very less petrol pumps that to on highway once you left the highway and entered the local road its just beautiful views with lush greenery. The road is straight and simple. The road is being widened at certain areas.\nThe structure of the temple is made up of Basalt rock and has multiple carvings which has been detailed by the ASI(Goa circle) in its information board at the entrance. This place would be worth visiting during rains however no swimming is permitted which can lead to fatalities. A warning board is seen to that effect. We were lucky to see small animals such as the cute monkeys who eat from your hand.\nAll in all this temple is a masterpiece of a carving dated to 12th or 13th sanctuary.\n\nParking - Ample amount of parking is there.\nTimings - Try to visit the temple before 05:30 PM and don’t stay after sunset as the place is very quiet and you barely see the vehicle until you touch the highway.\nFood - Yes the snacks are available\noutside the temple",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Sheshank Jaiswal",
                        "uri": "https://www.google.com/maps/contrib/110541659422844032827/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjV5U5o5Ko-GfXN4PboAoswkC5Im4OBQ3J-PfleNTjZOgJzX7pO5=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-05-26T10:32:06.627095Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VMR2tsUGIxaDYyOEZREAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VMR2tsUGIxaDYyOEZREAE!2m1!1s0x3bbf084e06883729:0x492ac4c792d0417a"
                }
            ]
        },
        {
            "id": "ChIJf96RIQ7qvzsRZ6zRbAGjnyo",
            "displayName": {
                "text": "Fat Fish",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJf96RIQ7qvzsRZ6zRbAGjnyo/reviews/ChZDSUhNMG9nS0VJQ0FnTURva3BiQ1FBEAE",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 4,
                    "text": {
                        "text": "One of the best and must visit place for sea food lovers in Goa. It’s a huge place spread in two buildings on both sides of the road. The ambience is good. We ordered king fish tawa fry, butter garlic prawns, spaghetti, chicken biryani along with few drinks. Both the fish and prawns dishes were really good. King fish was thinly sliced and then fried on tawa to perfection. I didn’t like spaghetti and the chicken biryani was average. The seafood was so good that we went there again the next day and ordered king fish tawa fry once again along with butter garlic mussels, goan style chili prawns, prawns tawa fry, dal kichidi, and chicken fried rice. All the dishes were really good. Finally we ended the meal with Goa special dessert Sera Durra.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "One of the best and must visit place for sea food lovers in Goa. It’s a huge place spread in two buildings on both sides of the road. The ambience is good. We ordered king fish tawa fry, butter garlic prawns, spaghetti, chicken biryani along with few drinks. Both the fish and prawns dishes were really good. King fish was thinly sliced and then fried on tawa to perfection. I didn’t like spaghetti and the chicken biryani was average. The seafood was so good that we went there again the next day and ordered king fish tawa fry once again along with butter garlic mussels, goan style chili prawns, prawns tawa fry, dal kichidi, and chicken fried rice. All the dishes were really good. Finally we ended the meal with Goa special dessert Sera Durra.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "nagarjuna varma",
                        "uri": "https://www.google.com/maps/contrib/102215721985573959092/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVRaJbkXvai1x1or5HRdmYI337B-x4e7tp-ru-qA6e_K1gMWpC4=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-23T10:07:18.580486Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTURva3BiQ1FBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTURva3BiQ1FBEAE!2m1!1s0x3bbfea0e2191de7f:0x2a9fa3016cd1ac67"
                },
                {
                    "name": "places/ChIJf96RIQ7qvzsRZ6zRbAGjnyo/reviews/Ci9DQUlRQUNvZENodHljRjlvT21wWVZGUmhXRXhWV2tZNWFrMVVSRkU1ZVZwSFlVRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 5,
                    "text": {
                        "text": "Had an absolute blast at Fat Fish Restauran🌴👌 The ambience was lively and vibrant, with a beautiful open-air setup that added to the overall dining experience. 🌊\n\nThe food was phenomenal - I ordered the Butter Garlic Prawns and Fish Tawa Fry, and both were incredibly delicious! 🍴👌 The prawns were succulent and flavorful, while the fish was cooked to perfection. The portions were generous too! 🤩\n\nService was top-notch, with friendly and attentive staff who ensured our glasses were always full. 🙌\n\nValue-wise, it was a bit on the higher side, but considering the quality and quantity of food, it was worth every penny! 💸👌\n\nAll in all, Fat Fish is a must-visit for seafood lovers in Goa.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Had an absolute blast at Fat Fish Restauran🌴👌 The ambience was lively and vibrant, with a beautiful open-air setup that added to the overall dining experience. 🌊\n\nThe food was phenomenal - I ordered the Butter Garlic Prawns and Fish Tawa Fry, and both were incredibly delicious! 🍴👌 The prawns were succulent and flavorful, while the fish was cooked to perfection. The portions were generous too! 🤩\n\nService was top-notch, with friendly and attentive staff who ensured our glasses were always full. 🙌\n\nValue-wise, it was a bit on the higher side, but considering the quality and quantity of food, it was worth every penny! 💸👌\n\nAll in all, Fat Fish is a must-visit for seafood lovers in Goa.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Batul Antariya",
                        "uri": "https://www.google.com/maps/contrib/101653895179716172158/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXE-iVLZwWvccoJd7X3b5KorP7L-Wr9WV2xieQEK44kSPcdSetP=s128-c0x00000000-cc-rp-mo-ba3"
                    },
                    "publishTime": "2025-06-27T04:54:27.274281751Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT21wWVZGUmhXRXhWV2tZNWFrMVVSRkU1ZVZwSFlVRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT21wWVZGUmhXRXhWV2tZNWFrMVVSRkU1ZVZwSFlVRRAB!2m1!1s0x3bbfea0e2191de7f:0x2a9fa3016cd1ac67"
                },
                {
                    "name": "places/ChIJf96RIQ7qvzsRZ6zRbAGjnyo/reviews/ChdDSUhNMG9nS0VJQ0FnTURveXB2MTRnRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "We dined at Fat Fish day after day and it was truly an incredible experience from start to finish. The food was absolutely fresh and bursting with flavor — every dish we tried was delicious and clearly made with quality ingredients.\n\nThe atmosphere was relaxed yet vibrant, making it a perfect spot for both a casual meal or a night out. The staff were friendly and welcoming amazing service, adding to the overall great vibe of the place.  On top of all that, the prices were very fair for the quality you get — excellent value for money.\n\nHighly recommend Fat Fish if you're looking for great food in a great setting. We’ll definitely be coming back!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "We dined at Fat Fish day after day and it was truly an incredible experience from start to finish. The food was absolutely fresh and bursting with flavor — every dish we tried was delicious and clearly made with quality ingredients.\n\nThe atmosphere was relaxed yet vibrant, making it a perfect spot for both a casual meal or a night out. The staff were friendly and welcoming amazing service, adding to the overall great vibe of the place.  On top of all that, the prices were very fair for the quality you get — excellent value for money.\n\nHighly recommend Fat Fish if you're looking for great food in a great setting. We’ll definitely be coming back!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "tanya stefanov",
                        "uri": "https://www.google.com/maps/contrib/105812210198168602473/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWyCiXv5FcxH8KxvI0jPlFkn9k5ChVWGD_oEc82-YFbnFHChVCB=s128-c0x00000000-cc-rp-mo"
                    },
                    "publishTime": "2025-04-23T18:23:06.376782Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURveXB2MTRnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURveXB2MTRnRRAB!2m1!1s0x3bbfea0e2191de7f:0x2a9fa3016cd1ac67"
                },
                {
                    "name": "places/ChIJf96RIQ7qvzsRZ6zRbAGjnyo/reviews/ChdDSUhNMG9nS0VJQ0FnTUNvNjR5S19nRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 4,
                    "text": {
                        "text": "This restaurant can be pretty busy with lots of tourists flocking in to try the delicious food that have on offer, but they have a large dining area with lots of tables. The staff is helpful and the ambience is casual with western music playing on speakers. The prices are on the higher side.\n\nI tried the Chonak Rawa Fry, and while the crispy exterior was a highlight, the dish fell short of expectations. The fish was cooked to perfection, with a satisfying crunch from the rawa coating. Compared to the rich, buttery taste of bhetki fish, the Chonak Rawa Fry was somewhat lacking.\n\nThe butter garlic prawn is delicious and soon became our favourite choice for the day. The butter chicken was on the blander side and could have tasted better. Overall, the food quality is good and fresh.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "This restaurant can be pretty busy with lots of tourists flocking in to try the delicious food that have on offer, but they have a large dining area with lots of tables. The staff is helpful and the ambience is casual with western music playing on speakers. The prices are on the higher side.\n\nI tried the Chonak Rawa Fry, and while the crispy exterior was a highlight, the dish fell short of expectations. The fish was cooked to perfection, with a satisfying crunch from the rawa coating. Compared to the rich, buttery taste of bhetki fish, the Chonak Rawa Fry was somewhat lacking.\n\nThe butter garlic prawn is delicious and soon became our favourite choice for the day. The butter chicken was on the blander side and could have tasted better. Overall, the food quality is good and fresh.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "debjyoti das",
                        "uri": "https://www.google.com/maps/contrib/105043103286967925328/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUb1AxMpU1q6HEG2eAtlslNpYmK2g9TyyjFeM398V4I8yq9y1mYVA=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-04-20T02:07:35.337378Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTUNvNjR5S19nRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTUNvNjR5S19nRRAB!2m1!1s0x3bbfea0e2191de7f:0x2a9fa3016cd1ac67"
                },
                {
                    "name": "places/ChIJf96RIQ7qvzsRZ6zRbAGjnyo/reviews/Ci9DQUlRQUNvZENodHljRjlvT2poTmF6RlVXbXhMWWxwTFNFdDVNVzkyYjJjMmFHYxAB",
                    "relativePublishTimeDescription": "2 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "A must visit place for seafood lovers who would love to try different types of cuisines at the same time.\nAmbience: It is a huge place with ample seating but the tables are placed quite close to each other to accommodate huge crowds. There is ample parking space too. The music is loud and so is the crowd. So you barely get to have a proper conversation.\nFood : we ordered stuffed crab, fish n chips, beef lasagna, tawa fried prawns, pork vindalu, poi bread ,cheese naan and brownie with icecream.The food was delicious and outstanding. We loved everything on our plate .\nThe service was slow but could not blame them as the orders kept pouring and it was worth the wait!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "A must visit place for seafood lovers who would love to try different types of cuisines at the same time.\nAmbience: It is a huge place with ample seating but the tables are placed quite close to each other to accommodate huge crowds. There is ample parking space too. The music is loud and so is the crowd. So you barely get to have a proper conversation.\nFood : we ordered stuffed crab, fish n chips, beef lasagna, tawa fried prawns, pork vindalu, poi bread ,cheese naan and brownie with icecream.The food was delicious and outstanding. We loved everything on our plate .\nThe service was slow but could not blame them as the orders kept pouring and it was worth the wait!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "niki Joe",
                        "uri": "https://www.google.com/maps/contrib/113541316603962740617/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWF8GUiiCHcDU_XabkvLGceJrhq7W-sRoqF50Ig08IABexttDznZg=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-18T17:22:57.176170127Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2poTmF6RlVXbXhMWWxwTFNFdDVNVzkyYjJjMmFHYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2poTmF6RlVXbXhMWWxwTFNFdDVNVzkyYjJjMmFHYxAB!2m1!1s0x3bbfea0e2191de7f:0x2a9fa3016cd1ac67"
                }
            ]
        },
        {
            "id": "ChIJtW8AjS7HvzsRnnAZoefYMAg",
            "displayName": {
                "text": "Dona Paula View Point",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJtW8AjS7HvzsRnnAZoefYMAg/reviews/Ci9DQUlRQUNvZENodHljRjlvT2s1RGNDMUpZMUpZVURkck9VSTNiekpVUlhGNWJXYxAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 4,
                    "text": {
                        "text": "Dona Paula Jetty View Point is one the places worth visiting when you are in Goa. There is entry ticket for adults - ₹50 and kids - ₹25 . There are parking areas nearby hence its convenient. Various movie scenes are shot at this place. There are various view points at Dona Paula and each one of them are worth visiting. The sea is infinite and beautiful. Would definitely recommend this place.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Dona Paula Jetty View Point is one the places worth visiting when you are in Goa. There is entry ticket for adults - ₹50 and kids - ₹25 . There are parking areas nearby hence its convenient. Various movie scenes are shot at this place. There are various view points at Dona Paula and each one of them are worth visiting. The sea is infinite and beautiful. Would definitely recommend this place.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Sofiya Kapasi",
                        "uri": "https://www.google.com/maps/contrib/113325401690521682308/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVHC-SDDPneOPU_SIPd3qBkVbDF6l7fN4PhycncDdBML4SkSYBFNg=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-26T14:36:15.460828866Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2s1RGNDMUpZMUpZVURkck9VSTNiekpVUlhGNWJXYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2s1RGNDMUpZMUpZVURkck9VSTNiekpVUlhGNWJXYxAB!2m1!1s0x3bbfc72e8d006fb5:0x830d8e7a119709e"
                },
                {
                    "name": "places/ChIJtW8AjS7HvzsRnnAZoefYMAg/reviews/ChZDSUhNMG9nS0VJQ0FnTUNJaFB6X2Z3EAE",
                    "relativePublishTimeDescription": "3 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Such a romantic place..\n\nGood spot for sunset view. But it is crowded. There is a climb of about 70 steps to reach the top. There are viewing spots at each level. Seems to be privately managed (ticket cost ₹50 per adult\n\nIt's a great spot to enjoy panoramic sights. Its peaceful surroundings make it an ideal place for relaxation and sightseeing.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Such a romantic place..\n\nGood spot for sunset view. But it is crowded. There is a climb of about 70 steps to reach the top. There are viewing spots at each level. Seems to be privately managed (ticket cost ₹50 per adult\n\nIt's a great spot to enjoy panoramic sights. Its peaceful surroundings make it an ideal place for relaxation and sightseeing.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Safi Jain",
                        "uri": "https://www.google.com/maps/contrib/112398238240457731112/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXfUewplI4PwozsurhdpNKd6DI2y9sJBKAzNARTk5VCeU49RWcm1Q=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-03-30T07:58:31.012689Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNJaFB6X2Z3EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNJaFB6X2Z3EAE!2m1!1s0x3bbfc72e8d006fb5:0x830d8e7a119709e"
                },
                {
                    "name": "places/ChIJtW8AjS7HvzsRnnAZoefYMAg/reviews/ChZDSUhNMG9nS0VLTzc4NWJUcGZQZEtBEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "In my recent visit to Goa , I was staying very close to Dona Paula Point. I went there early morning at around 6 30 am. Breathtaking view,  the silence of the waves and whispers of the wind. Completely loved it. Felt like the whole world is mine. Did my mediation there for sometime,  the experience was heavenly.  To add to this amazing experience was the beautiful rainbow.  Collected amazing memories",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "In my recent visit to Goa , I was staying very close to Dona Paula Point. I went there early morning at around 6 30 am. Breathtaking view,  the silence of the waves and whispers of the wind. Completely loved it. Felt like the whole world is mine. Did my mediation there for sometime,  the experience was heavenly.  To add to this amazing experience was the beautiful rainbow.  Collected amazing memories",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Shalini Sankar",
                        "uri": "https://www.google.com/maps/contrib/110833918563023643696/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocIOPdSFNNYVHWzkDE7TpwIc8LTWKXEzoSDXFKUqPMz0ze-O3tQL=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-05-27T05:43:44.960794Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VLTzc4NWJUcGZQZEtBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VLTzc4NWJUcGZQZEtBEAE!2m1!1s0x3bbfc72e8d006fb5:0x830d8e7a119709e"
                },
                {
                    "name": "places/ChIJtW8AjS7HvzsRnnAZoefYMAg/reviews/ChZDSUhNMG9nS0VJQ0FnTUNnOUo3T2NnEAE",
                    "relativePublishTimeDescription": "4 months ago",
                    "rating": 4,
                    "text": {
                        "text": "Beautiful Sunset Viewpoint\n\n👍Dona Paula Viewpoint is a great place to visit if you love sea views and sunsets 🌇.The view of the ocean 🌊 is amazing and the sunset looks beautiful from here. It's a peaceful spot to relax and enjoy 🤗 the scenery.\n\n👎The only downside is the ticket price, which is 💶 ₹50 per person. It feels a bit unnecessary, but apart from that, everything is good.\n\n🌞The place is well-maintained and the sunset experience is totally worth it.\n\nIf you're in Goa⛱️, it's a nice place to visit in the evening.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Beautiful Sunset Viewpoint\n\n👍Dona Paula Viewpoint is a great place to visit if you love sea views and sunsets 🌇.The view of the ocean 🌊 is amazing and the sunset looks beautiful from here. It's a peaceful spot to relax and enjoy 🤗 the scenery.\n\n👎The only downside is the ticket price, which is 💶 ₹50 per person. It feels a bit unnecessary, but apart from that, everything is good.\n\n🌞The place is well-maintained and the sunset experience is totally worth it.\n\nIf you're in Goa⛱️, it's a nice place to visit in the evening.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Puneet",
                        "uri": "https://www.google.com/maps/contrib/111145525687065736680/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUKV1Mn0pktliNDg_mwAktPYjMSdpUndBgvKM1uGarX-mdvvEJeyg=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-02-14T16:35:07.565252Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNnOUo3T2NnEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNnOUo3T2NnEAE!2m1!1s0x3bbfc72e8d006fb5:0x830d8e7a119709e"
                },
                {
                    "name": "places/ChIJtW8AjS7HvzsRnnAZoefYMAg/reviews/Ci9DQUlRQUNvZENodHljRjlvT2xjeE1tWmtkMlpHVkdSNWFURlVVamhZUkVoVFpVRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 5,
                    "text": {
                        "text": "Really very nice place to spend some quality time. The views are really good and you can enjoy Sunset here.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Really very nice place to spend some quality time. The views are really good and you can enjoy Sunset here.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Hardial Singh",
                        "uri": "https://www.google.com/maps/contrib/103620291944839179582/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXxHBtVAJtoEvAnsuDflacnjDP3VstbyZpevmyvtfJl0H0T2R-zug=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-21T15:45:10.923410729Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2xjeE1tWmtkMlpHVkdSNWFURlVVamhZUkVoVFpVRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2xjeE1tWmtkMlpHVkdSNWFURlVVamhZUkVoVFpVRRAB!2m1!1s0x3bbfc72e8d006fb5:0x830d8e7a119709e"
                }
            ]
        },
        {
            "id": "ChIJd1UvU43AvzsRbZivinLpPlA",
            "displayName": {
                "text": "Deltin Royale",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJd1UvU43AvzsRbZivinLpPlA/reviews/ChZDSUhNMG9nS0VJQ0FnTUNvNTRINU93EAE",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "The Ultimate Gaming & Entertainment Experience – Deltin Royale Casino, Goa\n\nIf you’re looking for the best casino experience in Goa, Deltin Royale is where your search ends. It’s not just a casino — it’s a full-fledged entertainment destination. With a wide range of games to choose from, a cozy and well-designed ambiance, and extremely cooperative staff, everything here feels top-notch and seamless.\n\nThe live entertainment shows are a real highlight, keeping the energy alive and the guests engaged throughout the evening. One of the most impressive aspects is their Kids Zone — thoughtfully designed so parents can enjoy their time without worry. Our kids had such a great time, they didn’t want to leave — and frankly, neither did we!\n\nThe Sky Bar is another gem, hosting lively parties that go on till late, offering a fantastic vibe with beautiful views. Food at Deltin Royale is another win — flavorful, with a wide variety of options. Just a tip: try to eat before peak closing hours to avoid the crowd.\n\nKids are welcome in the restaurant and entertainment areas, and those 12 years and older can even access the Sky Bar on the top floor, which is a nice touch for families. The money transactions are smooth and hassle-free, making the entire experience even more relaxed.\n\nWhether you’re visiting Goa for the first time or the tenth, Deltin Royale is an experience you’ll want to return to — again and again. Highly recommended!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "The Ultimate Gaming & Entertainment Experience – Deltin Royale Casino, Goa\n\nIf you’re looking for the best casino experience in Goa, Deltin Royale is where your search ends. It’s not just a casino — it’s a full-fledged entertainment destination. With a wide range of games to choose from, a cozy and well-designed ambiance, and extremely cooperative staff, everything here feels top-notch and seamless.\n\nThe live entertainment shows are a real highlight, keeping the energy alive and the guests engaged throughout the evening. One of the most impressive aspects is their Kids Zone — thoughtfully designed so parents can enjoy their time without worry. Our kids had such a great time, they didn’t want to leave — and frankly, neither did we!\n\nThe Sky Bar is another gem, hosting lively parties that go on till late, offering a fantastic vibe with beautiful views. Food at Deltin Royale is another win — flavorful, with a wide variety of options. Just a tip: try to eat before peak closing hours to avoid the crowd.\n\nKids are welcome in the restaurant and entertainment areas, and those 12 years and older can even access the Sky Bar on the top floor, which is a nice touch for families. The money transactions are smooth and hassle-free, making the entire experience even more relaxed.\n\nWhether you’re visiting Goa for the first time or the tenth, Deltin Royale is an experience you’ll want to return to — again and again. Highly recommended!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Kuldip Deshmukh",
                        "uri": "https://www.google.com/maps/contrib/117734146077705346098/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVoB-dimc69lWaTtYCTeY9E6Z9CLWoyS9DLGr63NCkyBLIe87IX=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-20T13:17:31.209940Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNvNTRINU93EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNvNTRINU93EAE!2m1!1s0x3bbfc08d532f5577:0x503ee9728aaf986d"
                },
                {
                    "name": "places/ChIJd1UvU43AvzsRbZivinLpPlA/reviews/ChZDSUhNMG9nS0VOX2tuTENfa296Q2FREAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Loved the experience, good for great cocktails. Amazing place to chill with friends, I was not sure whether to go to a casino or not. My friends forced me to go with them and I joined them. It was amazing! No regrets. Had a lot of fun. But SPEND RESPONSIBLY!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Loved the experience, good for great cocktails. Amazing place to chill with friends, I was not sure whether to go to a casino or not. My friends forced me to go with them and I joined them. It was amazing! No regrets. Had a lot of fun. But SPEND RESPONSIBLY!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Shivaranjan S D",
                        "uri": "https://www.google.com/maps/contrib/114938162932831962287/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWLA2A3vgZ-ItvjEpSHieNgsCAkj7rsg-KzPiJGTX43VBlOJWUhRw=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-05-13T09:34:22.285638Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VOX2tuTENfa296Q2FREAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VOX2tuTENfa296Q2FREAE!2m1!1s0x3bbfc08d532f5577:0x503ee9728aaf986d"
                },
                {
                    "name": "places/ChIJd1UvU43AvzsRbZivinLpPlA/reviews/ChdDSUhNMG9nS0VJQ0FnTUNJeWQ2TGlnRRAB",
                    "relativePublishTimeDescription": "3 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Amazing cruise experience! They have lots of fun games like poker, blackjack, roulette and other slot machines. Multi storey facilities available and a good crowd. Highly recommended for someone who wants to have an out of ordinary experience.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Amazing cruise experience! They have lots of fun games like poker, blackjack, roulette and other slot machines. Multi storey facilities available and a good crowd. Highly recommended for someone who wants to have an out of ordinary experience.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Aniruddhan PN",
                        "uri": "https://www.google.com/maps/contrib/117124688562695234451/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVdOLu0uno0_2Y_Skke1LGB9ezhavM4G01tHXkDK96zI8iq7v_B=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-03T03:29:44.812764Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTUNJeWQ2TGlnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTUNJeWQ2TGlnRRAB!2m1!1s0x3bbfc08d532f5577:0x503ee9728aaf986d"
                },
                {
                    "name": "places/ChIJd1UvU43AvzsRbZivinLpPlA/reviews/ChZDSUhNMG9nS0VJQ0FnTURReFotdVJREAE",
                    "relativePublishTimeDescription": "3 months ago",
                    "rating": 3,
                    "text": {
                        "text": "I recently visited Deltin Royale Casino in Goa with my family, and since it was our first casino experience (aside from what we’ve seen in Vegas movies), our expectations were quite high. The reception and boarding process were impressive, with a luxurious feel that set the stage for an exciting evening.\n\nHowever, once we stepped inside the ship, the experience started to feel underwhelming. The interiors felt more like a low-budget movie set rather than the grand, opulent casino we had imagined. While the gaming area was engaging, the rest of the ship— including the deck with a stage and dining setup—felt quite average. The entertainment and food were not up to the mark, lacking the premium quality one would expect from a high-end casino.\n\nOverall, I didn’t find the experience worth the price. It’s decent for a one-time visit if you’ve never been to a casino before, but if you’re expecting a true Vegas-style experience, you might be disappointed.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "I recently visited Deltin Royale Casino in Goa with my family, and since it was our first casino experience (aside from what we’ve seen in Vegas movies), our expectations were quite high. The reception and boarding process were impressive, with a luxurious feel that set the stage for an exciting evening.\n\nHowever, once we stepped inside the ship, the experience started to feel underwhelming. The interiors felt more like a low-budget movie set rather than the grand, opulent casino we had imagined. While the gaming area was engaging, the rest of the ship— including the deck with a stage and dining setup—felt quite average. The entertainment and food were not up to the mark, lacking the premium quality one would expect from a high-end casino.\n\nOverall, I didn’t find the experience worth the price. It’s decent for a one-time visit if you’ve never been to a casino before, but if you’re expecting a true Vegas-style experience, you might be disappointed.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Bijoy Bhaskar K",
                        "uri": "https://www.google.com/maps/contrib/110124328706885309818/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjW7uWi_VMMdOuV6SdrQFiktGVFr_NCpIllN8qgDbuhDHA0NIEo2=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-03-12T12:44:02.472715Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTURReFotdVJREAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTURReFotdVJREAE!2m1!1s0x3bbfc08d532f5577:0x503ee9728aaf986d"
                },
                {
                    "name": "places/ChIJd1UvU43AvzsRbZivinLpPlA/reviews/ChZDSUhNMG9nS0VQcnF5cXFfLVBESVZBEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 2,
                    "text": {
                        "text": "Deltin Royale has a great atmosphere with lively dance performances and an overall vibrant vibe. However, if you're a vegetarian and mainly going for the casino experience, I would recommend thinking twice. The food options for vegetarians are quite limited, and from a gaming perspective, I didn’t find it worth the value as I couldn’t recover even close to the amount I spent. Still, it’s an interesting place to visit for the ambience and entertainment.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Deltin Royale has a great atmosphere with lively dance performances and an overall vibrant vibe. However, if you're a vegetarian and mainly going for the casino experience, I would recommend thinking twice. The food options for vegetarians are quite limited, and from a gaming perspective, I didn’t find it worth the value as I couldn’t recover even close to the amount I spent. Still, it’s an interesting place to visit for the ambience and entertainment.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Nitesh Solanki",
                        "uri": "https://www.google.com/maps/contrib/111867605682700839250/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUdkFykl14W4CKmf1yi63JPZyvixpqpB4sA9C2QCCR67GPNUvSj=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-01T08:31:00.916056Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VQcnF5cXFfLVBESVZBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VQcnF5cXFfLVBESVZBEAE!2m1!1s0x3bbfc08d532f5577:0x503ee9728aaf986d"
                }
            ]
        },
        {
            "id": "ChIJWUM-d5rrvzsRpCPgq2akWDE",
            "displayName": {
                "text": "Hammerzz Nightclub",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJWUM-d5rrvzsRpCPgq2akWDE/reviews/Ci9DQUlRQUNvZENodHljRjlvT25VeVFYbzVVSFF4VlZCYVkza3plRkV3YUV0U2QwRRAB",
                    "relativePublishTimeDescription": "in the last week",
                    "rating": 5,
                    "text": {
                        "text": "Hammerzz Nightclub in Baga, Goa,has luxurious ambiance and electrifying atmosphere.\n* Vibe and Atmosphere: the club as having an amazing, incredible, and electrifying vibe with a luxurious and opulent ambiance.\n* Music and DJs: The club is praised for its great music, excellent DJ performances, and top-notch sound system.\n* Interiors and Facilities: stylish interiors, spacious dance floor, flashy light displays, and VIP lounges. and rooftop bar.\n* Service: friendly staff and good management.\n* Food and Drinks: The club offers delicious cocktails and tasty pizzas, with a wide range of drinks available.\n\nOverall, Hammerzz Nightclub is considered a must-visit for party lovers in Goa seeking a high-energy, luxurious clubbing experience.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Hammerzz Nightclub in Baga, Goa,has luxurious ambiance and electrifying atmosphere.\n* Vibe and Atmosphere: the club as having an amazing, incredible, and electrifying vibe with a luxurious and opulent ambiance.\n* Music and DJs: The club is praised for its great music, excellent DJ performances, and top-notch sound system.\n* Interiors and Facilities: stylish interiors, spacious dance floor, flashy light displays, and VIP lounges. and rooftop bar.\n* Service: friendly staff and good management.\n* Food and Drinks: The club offers delicious cocktails and tasty pizzas, with a wide range of drinks available.\n\nOverall, Hammerzz Nightclub is considered a must-visit for party lovers in Goa seeking a high-energy, luxurious clubbing experience.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Madhav Chauhan",
                        "uri": "https://www.google.com/maps/contrib/102063976578729326364/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjX7BD3Cvl8Q9X33m0kULXesetoHJs2vYdbOCFrfTzEZdxdM8n1b=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-29T07:15:22.831129710Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT25VeVFYbzVVSFF4VlZCYVkza3plRkV3YUV0U2QwRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT25VeVFYbzVVSFF4VlZCYVkza3plRkV3YUV0U2QwRRAB!2m1!1s0x3bbfeb9a773e4359:0x3158a466abe023a4"
                },
                {
                    "name": "places/ChIJWUM-d5rrvzsRpCPgq2akWDE/reviews/ChZDSUhNMG9nS0VJQ0FnTUR3Mk5fUkZREAE",
                    "relativePublishTimeDescription": "3 months ago",
                    "rating": 5,
                    "text": {
                        "text": "I came to Hammerzz to DJ. It was literally one of the best experiences ever. Especially from my perspective. The staff from the door all the way to the DJ booth were incredible and awesome personalities! The layout of the venue was spectacular, with multiple floors. The atmosphere of the customers was on another level. The kind of people who go here are really well behaved and educated people. Completely luxury and high class in every regard. From my perspective, the party goers were all free to dance and express themselves in a fulfilling way. I truly believe this is one of the best clubs in the world. The MUSIC in this club is awesome! everyone sings a long, the visuals, the lighting, is crazy! All the DJ's play the most popular songs from start to finish!  So if you're looking to have an amazing time, trust me! go to hammerzz!!!!\nAlso, the food + shisha was incredible. Its amazing that they have so much highend things happening in one facility!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "I came to Hammerzz to DJ. It was literally one of the best experiences ever. Especially from my perspective. The staff from the door all the way to the DJ booth were incredible and awesome personalities! The layout of the venue was spectacular, with multiple floors. The atmosphere of the customers was on another level. The kind of people who go here are really well behaved and educated people. Completely luxury and high class in every regard. From my perspective, the party goers were all free to dance and express themselves in a fulfilling way. I truly believe this is one of the best clubs in the world. The MUSIC in this club is awesome! everyone sings a long, the visuals, the lighting, is crazy! All the DJ's play the most popular songs from start to finish!  So if you're looking to have an amazing time, trust me! go to hammerzz!!!!\nAlso, the food + shisha was incredible. Its amazing that they have so much highend things happening in one facility!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Mike Bond",
                        "uri": "https://www.google.com/maps/contrib/105408779553307209255/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocJ9fkFEmx7BhnPDum64HufNjOxQEizKKGTlhkFXAe_0CR04Bg=s128-c0x00000000-cc-rp-mo"
                    },
                    "publishTime": "2025-03-23T09:21:00.505815Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUR3Mk5fUkZREAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUR3Mk5fUkZREAE!2m1!1s0x3bbfeb9a773e4359:0x3158a466abe023a4"
                },
                {
                    "name": "places/ChIJWUM-d5rrvzsRpCPgq2akWDE/reviews/ChdDSUhNMG9nS0VLcUdwYnp4Z1B6TTNRRRAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 4,
                    "text": {
                        "text": "Hammerzz is a great party place which is exactly what we were looking for. The music, the dj, dj Melroy is really good. For a party lover this could be a great experience. However please note that there are cover charges. 2 girls were charged Rs 1000 each. They give you a loaded card which can be swiped to buy drinks and food. The crowd could have been better. People do not understand the concept of personal space.\nAnyway drinks are priced quite high, for example a Budweiser can was for Rs 600 a pop, Their signature cocktails start at 1500. So be mindful of that. Do not reach before 10pm. That's when the party really picks up.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Hammerzz is a great party place which is exactly what we were looking for. The music, the dj, dj Melroy is really good. For a party lover this could be a great experience. However please note that there are cover charges. 2 girls were charged Rs 1000 each. They give you a loaded card which can be swiped to buy drinks and food. The crowd could have been better. People do not understand the concept of personal space.\nAnyway drinks are priced quite high, for example a Budweiser can was for Rs 600 a pop, Their signature cocktails start at 1500. So be mindful of that. Do not reach before 10pm. That's when the party really picks up.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Dilpreet Kaur Bedi",
                        "uri": "https://www.google.com/maps/contrib/107119300223454906758/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVyeLubm6dKo2Bo2APIMpFvI9rPnYVxJF0ZMXD7oAX52pJM8yAH=s128-c0x00000000-cc-rp-mo-ba3"
                    },
                    "publishTime": "2025-06-12T03:44:48.101686Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VLcUdwYnp4Z1B6TTNRRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VLcUdwYnp4Z1B6TTNRRRAB!2m1!1s0x3bbfeb9a773e4359:0x3158a466abe023a4"
                },
                {
                    "name": "places/ChIJWUM-d5rrvzsRpCPgq2akWDE/reviews/ChdDSUhNMG9nS0VJQ0FnTUNBd0p6YzBnRRAB",
                    "relativePublishTimeDescription": "5 months ago",
                    "rating": 1,
                    "text": {
                        "text": "Highly overrated place. The unpleasant experience starts from the entry with very rude guards and the reception which takes entry fees as per whims and fancies.\nThe service inside is better but the dance floor is not good. Dj and the music selection is good.\nBut one thing that kiils the entire vibe is the place is very highly overpriced. 500 ml water bottle alone costs 150 rs and to top it all the menu and prices are not revealed untill you pay cover charge and enter.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Highly overrated place. The unpleasant experience starts from the entry with very rude guards and the reception which takes entry fees as per whims and fancies.\nThe service inside is better but the dance floor is not good. Dj and the music selection is good.\nBut one thing that kiils the entire vibe is the place is very highly overpriced. 500 ml water bottle alone costs 150 rs and to top it all the menu and prices are not revealed untill you pay cover charge and enter.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Pankaj",
                        "uri": "https://www.google.com/maps/contrib/102526702859698279953/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXszVxcTI-yRCtr2A5IUNwWGNbSCNFsoOApJeBPbgyliqUxljeq=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-01-28T18:27:03.589232Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTUNBd0p6YzBnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTUNBd0p6YzBnRRAB!2m1!1s0x3bbfeb9a773e4359:0x3158a466abe023a4"
                },
                {
                    "name": "places/ChIJWUM-d5rrvzsRpCPgq2akWDE/reviews/ChZDSUhNMG9nS0VJQ0FnTUNnOWJqTEh3EAE",
                    "relativePublishTimeDescription": "4 months ago",
                    "rating": 2,
                    "text": {
                        "text": "My visit to Hammerzz last month (January 2025) was extremely disappointing, especially considering the ₹5,000 cover charge for my partner and me.  The dance floor was so overcrowded that dancing was impossible – even moving was a struggle.  While they initially appeared strict about enforcing the no-minors policy, I later saw the same individuals inside, making the initial checks seem like a mere formality.  The exorbitant ₹30,000 table charge is also ridiculous.\n\nThe food was mediocre at best, and the DJ's performance was equally underwhelming.  He abruptly switched songs, barely letting a few lines play, which completely disrupted any flow and made it difficult to enjoy the music.  The constant influx of people created a suffocating atmosphere, particularly on a Saturday (I can't speak for other nights).  Anyone with breathing difficulties should absolutely avoid this club, especially on weekends.\n\nThe only redeeming quality was the lounge area featuring live music.  I thoroughly enjoyed Sudhir's performance and was able to dance freely there.\n\nWhile the bartenders were friendly and efficient, and the service was generally good, it couldn't compensate for the negative aspects.  The drinks were also overpriced.  Hammerzz is vastly overhyped, especially on Instagram.  There are far better clubs in Goa with ample space to dance and enjoy a night out.  This was my first and definitely my last visit.  Overall, I wouldn't recommend Hammerzz to anyone.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "My visit to Hammerzz last month (January 2025) was extremely disappointing, especially considering the ₹5,000 cover charge for my partner and me.  The dance floor was so overcrowded that dancing was impossible – even moving was a struggle.  While they initially appeared strict about enforcing the no-minors policy, I later saw the same individuals inside, making the initial checks seem like a mere formality.  The exorbitant ₹30,000 table charge is also ridiculous.\n\nThe food was mediocre at best, and the DJ's performance was equally underwhelming.  He abruptly switched songs, barely letting a few lines play, which completely disrupted any flow and made it difficult to enjoy the music.  The constant influx of people created a suffocating atmosphere, particularly on a Saturday (I can't speak for other nights).  Anyone with breathing difficulties should absolutely avoid this club, especially on weekends.\n\nThe only redeeming quality was the lounge area featuring live music.  I thoroughly enjoyed Sudhir's performance and was able to dance freely there.\n\nWhile the bartenders were friendly and efficient, and the service was generally good, it couldn't compensate for the negative aspects.  The drinks were also overpriced.  Hammerzz is vastly overhyped, especially on Instagram.  There are far better clubs in Goa with ample space to dance and enjoy a night out.  This was my first and definitely my last visit.  Overall, I wouldn't recommend Hammerzz to anyone.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Pritesh Gajjan",
                        "uri": "https://www.google.com/maps/contrib/106865682579749528869/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXx3SFKekgXB1yOjX8i9lmRL6ZhWRiVOuAzsLAaVX6991G9p1_G=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-02-19T17:20:53.546033Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNnOWJqTEh3EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNnOWJqTEh3EAE!2m1!1s0x3bbfeb9a773e4359:0x3158a466abe023a4"
                }
            ],
            "priceRange": {
                "startPrice": {
                    "currencyCode": "INR",
                    "units": "2000"
                }
            }
        },
        {
            "id": "ChIJAQAAALy6vzsR9rv-XA9g7a8",
            "displayName": {
                "text": "Shree Mangesh temple",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJAQAAALy6vzsR9rv-XA9g7a8/reviews/ChdDSUhNMG9nS0VJQ0FnTUR3bE5iS2p3RRAB",
                    "relativePublishTimeDescription": "3 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Shree Mangueshi Temple is a serene and spiritually uplifting place that’s definitely worth a visit. Although it’s quite a long drive from North Goa, the peaceful atmosphere and beautiful architecture make the journey absolutely worthwhile. The temple is well-maintained, clean, and surrounded by scenic greenery, offering a calm escape from the hustle and bustle. The intricate carvings and traditional Goan temple style add to its charm. Visitors are welcomed with warmth, and the overall vibe is very soothing. A must-visit for anyone seeking a quiet, spiritual experience while exploring Goa’s rich cultural heritage.\nPlease follow the dress code.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Shree Mangueshi Temple is a serene and spiritually uplifting place that’s definitely worth a visit. Although it’s quite a long drive from North Goa, the peaceful atmosphere and beautiful architecture make the journey absolutely worthwhile. The temple is well-maintained, clean, and surrounded by scenic greenery, offering a calm escape from the hustle and bustle. The intricate carvings and traditional Goan temple style add to its charm. Visitors are welcomed with warmth, and the overall vibe is very soothing. A must-visit for anyone seeking a quiet, spiritual experience while exploring Goa’s rich cultural heritage.\nPlease follow the dress code.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Balavignesh Murali",
                        "uri": "https://www.google.com/maps/contrib/107058636134226737308/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUvji6KDhZVqrpWmVqeBLctR1Iss0-AzLF7_OvLPQz2irYP1h0=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-03-23T19:36:54.660408Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTUR3bE5iS2p3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTUR3bE5iS2p3RRAB!2m1!1s0x3bbfbabc00000001:0xafed600f5cfebbf6"
                },
                {
                    "name": "places/ChIJAQAAALy6vzsR9rv-XA9g7a8/reviews/Ci9DQUlRQUNvZENodHljRjlvT2pZd2RtcHNiazExYUhoRGRVWXhZbEYwZVZCc1YxRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 5,
                    "text": {
                        "text": "Nestled in the serene heart of Goa, **Mangesh Temple is pure peace in every sense**. From the gentle chimes of bells to the timeless architecture echoing devotion, the entire experience feels like a quiet retreat for the soul. The moment you step in, there's a calming stillness—as if the world pauses to let you breathe.\n\nWhether you're seeking spiritual solace or just a moment of mindful silence, this temple offers a beautiful blend of culture and tranquility. *Peace, peace, peace* isn’t just a thought here—it’s a feeling you carry with you long after you leave.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Nestled in the serene heart of Goa, **Mangesh Temple is pure peace in every sense**. From the gentle chimes of bells to the timeless architecture echoing devotion, the entire experience feels like a quiet retreat for the soul. The moment you step in, there's a calming stillness—as if the world pauses to let you breathe.\n\nWhether you're seeking spiritual solace or just a moment of mindful silence, this temple offers a beautiful blend of culture and tranquility. *Peace, peace, peace* isn’t just a thought here—it’s a feeling you carry with you long after you leave.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Rahul Bangia",
                        "uri": "https://www.google.com/maps/contrib/103688257815520303248/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjW9bzXffZjdxIjyIhDTSYRKqbpUyiouoT86cEINLWe7PrCwh7mWcA=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-21T18:36:24.237441449Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2pZd2RtcHNiazExYUhoRGRVWXhZbEYwZVZCc1YxRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2pZd2RtcHNiazExYUhoRGRVWXhZbEYwZVZCc1YxRRAB!2m1!1s0x3bbfbabc00000001:0xafed600f5cfebbf6"
                },
                {
                    "name": "places/ChIJAQAAALy6vzsR9rv-XA9g7a8/reviews/ChZDSUhNMG9nS0VJQ0FnTUNvMS1Ya0RREAE",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 4,
                    "text": {
                        "text": "“माघे ऊभा मंगेश, पूढ़े ऊभा मंगेश”\nAmidst the beautiful Goan ghats and green terrains lie an absolute serene temple of Lord Shiva. Beautiful premises with strict dress code. Kindly come here properly dressed. Aura in the air is divine and you’ll feel the connect instantly. Stunning architecture is a mix of different cultures and history of our country. You’ll find less crowds on weekdays specially in afternoon period. A must visit place if you want to witness Goan history.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "“माघे ऊभा मंगेश, पूढ़े ऊभा मंगेश”\nAmidst the beautiful Goan ghats and green terrains lie an absolute serene temple of Lord Shiva. Beautiful premises with strict dress code. Kindly come here properly dressed. Aura in the air is divine and you’ll feel the connect instantly. Stunning architecture is a mix of different cultures and history of our country. You’ll find less crowds on weekdays specially in afternoon period. A must visit place if you want to witness Goan history.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Dr. Kartik Patil",
                        "uri": "https://www.google.com/maps/contrib/105658958006420830414/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWt-fpb-1pb-nY5rusBRD4H4YOYGFjjfP4dIfdE6TTR1fiJ7RY=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-04-20T15:39:12.071360Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNvMS1Ya0RREAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNvMS1Ya0RREAE!2m1!1s0x3bbfbabc00000001:0xafed600f5cfebbf6"
                },
                {
                    "name": "places/ChIJAQAAALy6vzsR9rv-XA9g7a8/reviews/ChdDSUhNMG9nS0VJQ0FnTURvdGEzbzdnRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Great temple for Lord Shiva. Tucked in the ghats and has a few shops and restaurants along the way to the temple.\nIt’s and old temple with great architecture.\nThe temple has a strict dress code to be followed.\nSo please be aware of that .\nAnd they do provide stoles for women toand traditional clothing for men to ensure that the  dress code is adhered to. They charge a nominal fee for the dress they give on rent.\nThat fee has to be paid in cash.\nThey have a paid car parking area and they take cash only.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Great temple for Lord Shiva. Tucked in the ghats and has a few shops and restaurants along the way to the temple.\nIt’s and old temple with great architecture.\nThe temple has a strict dress code to be followed.\nSo please be aware of that .\nAnd they do provide stoles for women toand traditional clothing for men to ensure that the  dress code is adhered to. They charge a nominal fee for the dress they give on rent.\nThat fee has to be paid in cash.\nThey have a paid car parking area and they take cash only.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Nagarajan T V",
                        "uri": "https://www.google.com/maps/contrib/113821775492617164549/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWRj6OlkXF_aUsMuKyRDJ_EXcLinmg1lU7yMoPMCFbbcJzo2nVJxQ=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-04-27T09:08:34.737569Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURvdGEzbzdnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURvdGEzbzdnRRAB!2m1!1s0x3bbfbabc00000001:0xafed600f5cfebbf6"
                },
                {
                    "name": "places/ChIJAQAAALy6vzsR9rv-XA9g7a8/reviews/Ci9DQUlRQUNvZENodHljRjlvT2pKd2Eyb3hhekI1TjNsM055MW1Rek5FU21JNGNWRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 5,
                    "text": {
                        "text": "Shri Manguesh Temple in Priol‑Ponda, Goa, is a serene and deeply spiritual sanctuary dedicated to Lord Manguesh—an embodiment of Mahadev (Lord Shiva) revered as the ultimate power and cosmic force. This sacred site is famed for inspiring inner peace, calm, and introspection.\n\nAs an incarnation of Lord Shiva—the cosmic destroyer and creator—the deity Manguesh represents supreme spiritual power and wisdom.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Shri Manguesh Temple in Priol‑Ponda, Goa, is a serene and deeply spiritual sanctuary dedicated to Lord Manguesh—an embodiment of Mahadev (Lord Shiva) revered as the ultimate power and cosmic force. This sacred site is famed for inspiring inner peace, calm, and introspection.\n\nAs an incarnation of Lord Shiva—the cosmic destroyer and creator—the deity Manguesh represents supreme spiritual power and wisdom.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Kishor Deotare",
                        "uri": "https://www.google.com/maps/contrib/101705151372610015682/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXBFApxCqUEbSowl3aTJ50M5sQyAmACi8d4_hR7EWsXFiNXrN05=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-21T06:32:24.440418407Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2pKd2Eyb3hhekI1TjNsM055MW1Rek5FU21JNGNWRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2pKd2Eyb3hhekI1TjNsM055MW1Rek5FU21JNGNWRRAB!2m1!1s0x3bbfbabc00000001:0xafed600f5cfebbf6"
                }
            ]
        },
        {
            "id": "ChIJgQ9KpXTrvzsRccrngLKJ-bM",
            "displayName": {
                "text": "Calangute Beach, Goa",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJgQ9KpXTrvzsRccrngLKJ-bM/reviews/Ci9DQUlRQUNvZENodHljRjlvT214VVVsTjVSamR2ZDNFMFEweHpiVkI0YjBGWlJHYxAB",
                    "relativePublishTimeDescription": "2 weeks ago",
                    "rating": 4,
                    "text": {
                        "text": "We have visited Calangute Beach on 14-15 June weekend, it was very good experience, beach is very clean and neat.\nAs of now weather is slightly raining so there were tides ...\nBut overall experience on beach was very good and we will get all the swimming related things ( cloths, sport activities.. etc ) there only.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "We have visited Calangute Beach on 14-15 June weekend, it was very good experience, beach is very clean and neat.\nAs of now weather is slightly raining so there were tides ...\nBut overall experience on beach was very good and we will get all the swimming related things ( cloths, sport activities.. etc ) there only.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Rushikesh Kota",
                        "uri": "https://www.google.com/maps/contrib/113354592147486761999/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjX1Nv0w1yI4akDmSbf8I9GIoE6BOgnePIRgoQC9OEJd20UoLQnAgg=s128-c0x00000000-cc-rp-mo-ba2"
                    },
                    "publishTime": "2025-06-17T04:09:40.397266685Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT214VVVsTjVSamR2ZDNFMFEweHpiVkI0YjBGWlJHYxAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT214VVVsTjVSamR2ZDNFMFEweHpiVkI0YjBGWlJHYxAB!2m1!1s0x3bbfeb74a54a0f81:0xb3f989b280e7ca71"
                },
                {
                    "name": "places/ChIJgQ9KpXTrvzsRccrngLKJ-bM/reviews/ChZDSUhNMG9nS0VPYVVrYVdSLVpXMWJBEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "If want to enjoy Beach time in Evening or Night must Visit Calangute Beach, nearby shopping Options available, parking available, Beautiful and Energetic Vibe!\nCalangute Beach is the heart of North Goa! Loved the vibrant energy — from water sports to local shacks serving fresh seafood. Sunset was magical. Perfect place to chill with friends, sip a beer, and enjoy the Goan vibe. Highly recommend visiting during the weekdays to avoid crowds.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "If want to enjoy Beach time in Evening or Night must Visit Calangute Beach, nearby shopping Options available, parking available, Beautiful and Energetic Vibe!\nCalangute Beach is the heart of North Goa! Loved the vibrant energy — from water sports to local shacks serving fresh seafood. Sunset was magical. Perfect place to chill with friends, sip a beer, and enjoy the Goan vibe. Highly recommend visiting during the weekdays to avoid crowds.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Mahesh Patel",
                        "uri": "https://www.google.com/maps/contrib/113035659463371987002/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWMHxZGpmvbSImOpPTTM0OW0hS_f_pdNuRd7jdUTpQ8CXgX9hO0=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-01T17:53:59.814690Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VPYVVrYVdSLVpXMWJBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VPYVVrYVdSLVpXMWJBEAE!2m1!1s0x3bbfeb74a54a0f81:0xb3f989b280e7ca71"
                },
                {
                    "name": "places/ChIJgQ9KpXTrvzsRccrngLKJ-bM/reviews/ChdDSUhNMG9nS0VJQ0FnTURvLWViS3N3RRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "Very nice clean and clear oné of Goa beach. Fully enjoy on this place naice atmosphere.also  the guards very cooperative very time they intimate all the visitors. And one thing is that rates of too many things lower than other citys  also there is a shower room and changing room per person 30 rs unlimited water",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Very nice clean and clear oné of Goa beach. Fully enjoy on this place naice atmosphere.also  the guards very cooperative very time they intimate all the visitors. And one thing is that rates of too many things lower than other citys  also there is a shower room and changing room per person 30 rs unlimited water",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Vishal Kamble",
                        "uri": "https://www.google.com/maps/contrib/100061523182516587053/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocIzvZrq2sPnJuNDdXlgswRasVIi8xFvnkiK4RKewzTPD2Nh4A=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-04-26T18:03:27.740724Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURvLWViS3N3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURvLWViS3N3RRAB!2m1!1s0x3bbfeb74a54a0f81:0xb3f989b280e7ca71"
                },
                {
                    "name": "places/ChIJgQ9KpXTrvzsRccrngLKJ-bM/reviews/ChZDSUhNMG9nS0VJQ0FnTUNZNlphMVdREAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 4,
                    "text": {
                        "text": "Calangute Beach, often referred to as the \"Queen of Beaches\", is one of the most popular and vibrant beaches in North Goa. Located around 15 km from Panaji, it attracts thousands of domestic and international tourists every year for its golden sands, energetic vibe, and buzzing atmosphere.\n\nFood - Food is costly at the beach; you can compare before you sit.\n\nNearby Attractions\nBaga Beach: Just north of Calangute, Baga is famous for nightlife, Tito’s Lane, and vibrant beach clubs.\n\nAnjuna & Candolim: Other nearby beaches with different vibes – Anjuna for flea markets and hippie culture, Candolim for a quieter escape.\n\nLocal Churches & Culture: Explore Portuguese-era churches, local Goan architecture, and nearby markets for souvenirs and handicrafts.\n\nLively Atmosphere: Calangute is known for its energetic and youthful crowd. It's perfect for travelers who enjoy music, beach parties, and socializing with people from around the world.\n\nWide Sandy Shoreline: The beach is long and spacious, great for sunbathing, beach games, or simply relaxing by the sea.\n\nAdventure & Water Sports: Calangute is a hub for water sports like parasailing, water skiing, jet-skiing, banana boat rides, and windsurfing. It's ideal for thrill-seekers looking for beachside excitement.\n\nBeach Shacks and Cafes: Enjoy fresh seafood, Goan curry, cocktails, and chilled beers at one of the many beach shacks.\n\nNightlife: Calangute lights up at night with beach parties, music, and dance. Popular clubs and lounges are nearby, especially in Baga.\n\nHow to Reach Calangute Beach\nBy Air: The nearest airport is Dabolim Airport (GOI), about 40 km away.\n\nBy Train: Nearest railway station is Thivim, around 19 km from Calangute.\n\nBy Road: Well-connected by road from Panaji, Mapusa, and other parts of Goa. Rental bikes, taxis, and buses are commonly used.\n\nBest Time to Visit\nNovember to February is the best time to visit Calangute Beach. The weather is pleasant, ideal for water sports, sunbathing, and beach parties.\n\nAvoid the monsoon season (June to September) if you're looking for water activities, though the lush greenery during this time is a different kind of beauty.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Calangute Beach, often referred to as the \"Queen of Beaches\", is one of the most popular and vibrant beaches in North Goa. Located around 15 km from Panaji, it attracts thousands of domestic and international tourists every year for its golden sands, energetic vibe, and buzzing atmosphere.\n\nFood - Food is costly at the beach; you can compare before you sit.\n\nNearby Attractions\nBaga Beach: Just north of Calangute, Baga is famous for nightlife, Tito’s Lane, and vibrant beach clubs.\n\nAnjuna & Candolim: Other nearby beaches with different vibes – Anjuna for flea markets and hippie culture, Candolim for a quieter escape.\n\nLocal Churches & Culture: Explore Portuguese-era churches, local Goan architecture, and nearby markets for souvenirs and handicrafts.\n\nLively Atmosphere: Calangute is known for its energetic and youthful crowd. It's perfect for travelers who enjoy music, beach parties, and socializing with people from around the world.\n\nWide Sandy Shoreline: The beach is long and spacious, great for sunbathing, beach games, or simply relaxing by the sea.\n\nAdventure & Water Sports: Calangute is a hub for water sports like parasailing, water skiing, jet-skiing, banana boat rides, and windsurfing. It's ideal for thrill-seekers looking for beachside excitement.\n\nBeach Shacks and Cafes: Enjoy fresh seafood, Goan curry, cocktails, and chilled beers at one of the many beach shacks.\n\nNightlife: Calangute lights up at night with beach parties, music, and dance. Popular clubs and lounges are nearby, especially in Baga.\n\nHow to Reach Calangute Beach\nBy Air: The nearest airport is Dabolim Airport (GOI), about 40 km away.\n\nBy Train: Nearest railway station is Thivim, around 19 km from Calangute.\n\nBy Road: Well-connected by road from Panaji, Mapusa, and other parts of Goa. Rental bikes, taxis, and buses are commonly used.\n\nBest Time to Visit\nNovember to February is the best time to visit Calangute Beach. The weather is pleasant, ideal for water sports, sunbathing, and beach parties.\n\nAvoid the monsoon season (June to September) if you're looking for water activities, though the lush greenery during this time is a different kind of beauty.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Siddharth Salve",
                        "uri": "https://www.google.com/maps/contrib/102590202952414580038/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUE4WDDQoqr86fdDaX0lFkcura2Z55-LaApVjwh8tgiWr2Fx41O=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-05-05T10:20:27.380752Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNZNlphMVdREAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNZNlphMVdREAE!2m1!1s0x3bbfeb74a54a0f81:0xb3f989b280e7ca71"
                },
                {
                    "name": "places/ChIJgQ9KpXTrvzsRccrngLKJ-bM/reviews/Ci9DQUlRQUNvZENodHljRjlvT201blRrODBhVnB6TlVSMVZFMDFheTFaVVZoaGRsRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 5,
                    "text": {
                        "text": "Loved the beach..  just people don't have any civic sense.. and litter here and there..!! Must visit for sunset view... It's.. lovely....",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Loved the beach..  just people don't have any civic sense.. and litter here and there..!! Must visit for sunset view... It's.. lovely....",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "himanshu maheshwari",
                        "uri": "https://www.google.com/maps/contrib/110966851517197177333/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWkk9Q3jJK_RodBTyWEiMkHoyepIY0wrtmf99bxtgC71bqqkB0=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-06-26T18:32:50.342121389Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT201blRrODBhVnB6TlVSMVZFMDFheTFaVVZoaGRsRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT201blRrODBhVnB6TlVSMVZFMDFheTFaVVZoaGRsRRAB!2m1!1s0x3bbfeb74a54a0f81:0xb3f989b280e7ca71"
                }
            ]
        },
        {
            "id": "ChIJdzWLsxzqvzsRF6bC5-mKXSQ",
            "displayName": {
                "text": "Club Titos",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJdzWLsxzqvzsRF6bC5-mKXSQ/reviews/ChdDSUhNMG9nS0VJQ0FnTURJazdlZzdnRRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "My recent visit to Club Tito’s in Goa was nothing short of spectacular! Located on the vibrant Tito’s Lane near Baga Beach, it’s hands-down the best happening place in Goa. My friends and I had an absolute blast, dancing the night away to pulsating Bollywood and EDM beats spun by talented DJs. The energy was electric, with dazzling lights and a lively crowd that kept the vibes soaring. The drinks were top-notch, and the food, especially the Goan-inspired bites, was delicious. The open-air dance floor and beachside vibes made it an unforgettable experience. I highly recommend everyone visiting Goa to hit up Club Tito’s at least once—it’s a must-have experience that captures the heart of Goa’s nightlife! Five stars for an epic night",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "My recent visit to Club Tito’s in Goa was nothing short of spectacular! Located on the vibrant Tito’s Lane near Baga Beach, it’s hands-down the best happening place in Goa. My friends and I had an absolute blast, dancing the night away to pulsating Bollywood and EDM beats spun by talented DJs. The energy was electric, with dazzling lights and a lively crowd that kept the vibes soaring. The drinks were top-notch, and the food, especially the Goan-inspired bites, was delicious. The open-air dance floor and beachside vibes made it an unforgettable experience. I highly recommend everyone visiting Goa to hit up Club Tito’s at least once—it’s a must-have experience that captures the heart of Goa’s nightlife! Five stars for an epic night",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "HarshKumar Ahir",
                        "uri": "https://www.google.com/maps/contrib/107016486890689499805/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVfl4Z8wBT0BRCY9k-Uc4e3j6Zb-p-RkLbUpqgXIvGH6vR7m4Ui=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2025-04-17T05:32:42.553030Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTURJazdlZzdnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTURJazdlZzdnRRAB!2m1!1s0x3bbfea1cb38b3577:0x245d8ae9e7c2a617"
                },
                {
                    "name": "places/ChIJdzWLsxzqvzsRF6bC5-mKXSQ/reviews/ChdDSUhNMG9nS0VMaklrOEgydGJDVGlRRRAB",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Great time at Titos! We had come to Baga beach just to visit one of the clubs on the Titos lane and it didn't disappoint. We are a family of 4 with 2 teenage kids and it was suitable for the kids too who enjoyed some of the acrobatics displays. Couple entry is 1500 which will be redeemed against drinks. Cards are given at the entry and for each order the waiters will swipe the cards and provide a bill which marks the remaining value. All well managed. We went in past 9pm on a Friday first week of May and it was not crowded. Action started picking up past 10pm with DJ, in house performers and the dance floor (which was not that big). Good entertainment for few hours and we left well past midnight. The streets were still bustling with lot energy. Although we were staying the night at Aaristo inn at walking distance, I saw a fairly large paid parking lot right opposite the club. So it should not be a problem if you have to drive here.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Great time at Titos! We had come to Baga beach just to visit one of the clubs on the Titos lane and it didn't disappoint. We are a family of 4 with 2 teenage kids and it was suitable for the kids too who enjoyed some of the acrobatics displays. Couple entry is 1500 which will be redeemed against drinks. Cards are given at the entry and for each order the waiters will swipe the cards and provide a bill which marks the remaining value. All well managed. We went in past 9pm on a Friday first week of May and it was not crowded. Action started picking up past 10pm with DJ, in house performers and the dance floor (which was not that big). Good entertainment for few hours and we left well past midnight. The streets were still bustling with lot energy. Although we were staying the night at Aaristo inn at walking distance, I saw a fairly large paid parking lot right opposite the club. So it should not be a problem if you have to drive here.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "sgtachat",
                        "uri": "https://www.google.com/maps/contrib/108637248988056365563/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocJxj9imAJBHs2wOwqX7zfhti2jfORUlzvB3qyJGlE-55CLZUw=s128-c0x00000000-cc-rp-mo"
                    },
                    "publishTime": "2025-05-11T13:07:56.869732Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VMaklrOEgydGJDVGlRRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VMaklrOEgydGJDVGlRRRAB!2m1!1s0x3bbfea1cb38b3577:0x245d8ae9e7c2a617"
                },
                {
                    "name": "places/ChIJdzWLsxzqvzsRF6bC5-mKXSQ/reviews/ChdDSUhNMG9nS0VOemd2N3kyanNyd3dnRRAB",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Club Titos is very nice. Full enjoyment.\nDrinking is costly. The price is too high. Entry fees are High for boys only. A couple entries are free.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Club Titos is very nice. Full enjoyment.\nDrinking is costly. The price is too high. Entry fees are High for boys only. A couple entries are free.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Keyur Kuvadiya",
                        "uri": "https://www.google.com/maps/contrib/101672367406361765449/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocLYl-wnJ-LT8URgflGzjOqKdc1afItLaXViWx0coXXPcaFrkA=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-05-27T13:27:05.862553Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VOemd2N3kyanNyd3dnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VOemd2N3kyanNyd3dnRRAB!2m1!1s0x3bbfea1cb38b3577:0x245d8ae9e7c2a617"
                },
                {
                    "name": "places/ChIJdzWLsxzqvzsRF6bC5-mKXSQ/reviews/Ci9DQUlRQUNvZENodHljRjlvT2xaYVdWWlViSE5ZU1dwb1VVOWlVVzFOV2pKVk1rRRAB",
                    "relativePublishTimeDescription": "a week ago",
                    "rating": 5,
                    "text": {
                        "text": "First time visiting a club so it was a totally new experience for me\nMusic and DJ was managed by DJ Kriss and Bhumicka Singh\nIt had a entry fee around 2k\nWith complimentary drinks for expensive entry tickets\nCouple entry had a discount\nMusic and Sound was great\nBeats were upto the mark\nLightning and surrounding was perfect\nThey have good security too for safety\nIf you are visiting by car be ready to pay 500rs for parking",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "First time visiting a club so it was a totally new experience for me\nMusic and DJ was managed by DJ Kriss and Bhumicka Singh\nIt had a entry fee around 2k\nWith complimentary drinks for expensive entry tickets\nCouple entry had a discount\nMusic and Sound was great\nBeats were upto the mark\nLightning and surrounding was perfect\nThey have good security too for safety\nIf you are visiting by car be ready to pay 500rs for parking",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Akashsingh Sisodiya",
                        "uri": "https://www.google.com/maps/contrib/112616596914076396963/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjV6gTheOsGlrJJ7l2p_ZdaoShKE3k9aM0gmfiAWUSM7FyfkjACHzA=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-25T11:18:50.406719735Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2xaYVdWWlViSE5ZU1dwb1VVOWlVVzFOV2pKVk1rRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2xaYVdWWlViSE5ZU1dwb1VVOWlVVzFOV2pKVk1rRRAB!2m1!1s0x3bbfea1cb38b3577:0x245d8ae9e7c2a617"
                },
                {
                    "name": "places/ChIJdzWLsxzqvzsRF6bC5-mKXSQ/reviews/ChZDSUhNMG9nS0VJQ0FnTUNBX3RiVVNnEAE",
                    "relativePublishTimeDescription": "5 months ago",
                    "rating": 2,
                    "text": {
                        "text": "Review of Titos Club, Goa – Stag Entry Experience.\n\nIf you're planning a night out at Titos Club Goa as a group of stags, be prepared for overpriced entry fees and poor crowd management. We were a group of 10 guys, and the entry charge was ridiculous—they forced us to buy a ₹20,000 drink coupon, which was only for drinks.\n\nHidden Costs & Pricing\n\nFood is separate: To order food, you need another ₹10,000 coupon. So, in total, you’re looking at ₹30,000+ just for basic food and drinks.\n\nDrink prices are sky-high: Bottles range from ₹6,000 to ₹13,000. So, unless you’re prepared to spend a fortune, you’ll feel ripped off.\n\nTerrible Crowd Management & Insulting Behavior\n\nThe worst part of the night was the disrespectful behavior of the DJ Bhumika Singh. She poured liquor on my friend's face—completely unprovoked and rude! What kind of behavior is this from a professional?\n\nTo make it worse, my friend is not a random drunk tourist—he is a well-settled businessman from the UK, yet this so-called DJ treated him with absolute disrespect. The crowd management was terrible, and no one from the staff even intervened. Is this how they treat their guests?\n\nAmbiance & Experience\n\nStage Shows: There were some performances, but nothing too special. I have uploaded a few videos for reference.\n\nCrowd: The place only gets active after 11 PM, so if you go early, expect a dull atmosphere.\n\nSpace Issues: Due to ongoing renovations, the club feels cramped and lacks proper movement space.\n\nParking & Accessibility\n\nIf you’re coming by car, there is paid parking available about 100 meters up and down from Titos.\n\nFinal Verdict – Overhyped & Disrespectful Treatment\n\n⚠️ BEWARE OF THE PRICING TRAP—the mandatory drink and food coupons make this place one of the most expensive and misleading clubs in Goa for stags. But beyond the cost, the lack of basic hospitality and respect ruins the experience.\n\nThe rude behavior of DJ Bhumika Singh, terrible crowd management, and unjustified high pricing make this a club to avoid. Spend your money elsewhere, where you’ll be treated with respect!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Review of Titos Club, Goa – Stag Entry Experience.\n\nIf you're planning a night out at Titos Club Goa as a group of stags, be prepared for overpriced entry fees and poor crowd management. We were a group of 10 guys, and the entry charge was ridiculous—they forced us to buy a ₹20,000 drink coupon, which was only for drinks.\n\nHidden Costs & Pricing\n\nFood is separate: To order food, you need another ₹10,000 coupon. So, in total, you’re looking at ₹30,000+ just for basic food and drinks.\n\nDrink prices are sky-high: Bottles range from ₹6,000 to ₹13,000. So, unless you’re prepared to spend a fortune, you’ll feel ripped off.\n\nTerrible Crowd Management & Insulting Behavior\n\nThe worst part of the night was the disrespectful behavior of the DJ Bhumika Singh. She poured liquor on my friend's face—completely unprovoked and rude! What kind of behavior is this from a professional?\n\nTo make it worse, my friend is not a random drunk tourist—he is a well-settled businessman from the UK, yet this so-called DJ treated him with absolute disrespect. The crowd management was terrible, and no one from the staff even intervened. Is this how they treat their guests?\n\nAmbiance & Experience\n\nStage Shows: There were some performances, but nothing too special. I have uploaded a few videos for reference.\n\nCrowd: The place only gets active after 11 PM, so if you go early, expect a dull atmosphere.\n\nSpace Issues: Due to ongoing renovations, the club feels cramped and lacks proper movement space.\n\nParking & Accessibility\n\nIf you’re coming by car, there is paid parking available about 100 meters up and down from Titos.\n\nFinal Verdict – Overhyped & Disrespectful Treatment\n\n⚠️ BEWARE OF THE PRICING TRAP—the mandatory drink and food coupons make this place one of the most expensive and misleading clubs in Goa for stags. But beyond the cost, the lack of basic hospitality and respect ruins the experience.\n\nThe rude behavior of DJ Bhumika Singh, terrible crowd management, and unjustified high pricing make this a club to avoid. Spend your money elsewhere, where you’ll be treated with respect!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Vineeth S",
                        "uri": "https://www.google.com/maps/contrib/117362380296667235577/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjVdJVC9HpjQwO46UjQaggxn8eZ7ctX7imkMjx6hCOF0pQer5BCMlg=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-02-01T09:09:42.100990Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUNBX3RiVVNnEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUNBX3RiVVNnEAE!2m1!1s0x3bbfea1cb38b3577:0x245d8ae9e7c2a617"
                }
            ],
            "priceRange": {
                "startPrice": {
                    "currencyCode": "INR",
                    "units": "2000"
                }
            }
        },
        {
            "id": "ChIJVUbAVEy9vzsRS6uKBv4zNmI",
            "displayName": {
                "text": "Harvalem Waterfalls",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJVUbAVEy9vzsRS6uKBv4zNmI/reviews/ChdDSUhNMG9nS0VQUDFtTXpFdzg3anVBRRAB",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Although it's a small waterfall still it's good for taking pictures and also for people who haven't seen any waterfall yet. Bonus point is the landscape you get to see on your way to waterfall. I went on a rainyday and really enjoyed the view as well as waterfall, my decision of visiting it today when it was raining was perfect. For people who love nature are going to enjoy all the way to waterfall",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Although it's a small waterfall still it's good for taking pictures and also for people who haven't seen any waterfall yet. Bonus point is the landscape you get to see on your way to waterfall. I went on a rainyday and really enjoyed the view as well as waterfall, my decision of visiting it today when it was raining was perfect. For people who love nature are going to enjoy all the way to waterfall",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Priya Hemrom",
                        "uri": "https://www.google.com/maps/contrib/113917121607254735420/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjW7xxs_9lrRyYHC0lakAiW6Y2o2Er1xTUDPsDPuVu47fRszDTel=s128-c0x00000000-cc-rp-mo"
                    },
                    "publishTime": "2025-05-28T18:30:04.906040Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VQUDFtTXpFdzg3anVBRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VQUDFtTXpFdzg3anVBRRAB!2m1!1s0x3bbfbd4c54c04655:0x623633fe068aab4b"
                },
                {
                    "name": "places/ChIJVUbAVEy9vzsRS6uKBv4zNmI/reviews/ChZDSUhNMG9nS0VJQ0FnTUN3aHZDUmZBEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "This waterfall is close beautiful rudreshwar temple. The atmosphere is quite serene and peaceful.\nBest time to visit would be morning, evening in rainy season.\n\nNote- entering to water is not allowed.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "This waterfall is close beautiful rudreshwar temple. The atmosphere is quite serene and peaceful.\nBest time to visit would be morning, evening in rainy season.\n\nNote- entering to water is not allowed.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Shruti Poojary",
                        "uri": "https://www.google.com/maps/contrib/105475543976286884977/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjU37V2nxCd1LUNu7C6yIv937uMR2BpUb5aTHw_sPdZGdT3CyB5F=s128-c0x00000000-cc-rp-mo-ba3"
                    },
                    "publishTime": "2025-06-03T10:33:37.989795Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VJQ0FnTUN3aHZDUmZBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VJQ0FnTUN3aHZDUmZBEAE!2m1!1s0x3bbfbd4c54c04655:0x623633fe068aab4b"
                },
                {
                    "name": "places/ChIJVUbAVEy9vzsRS6uKBv4zNmI/reviews/Ci9DQUlRQUNvZENodHljRjlvT2xKWE5tVjVRa2cwYkROSE1UWllWWFJKU0dGcmJVRRAB",
                    "relativePublishTimeDescription": "in the last week",
                    "rating": 5,
                    "text": {
                        "text": "It's amazing Waterfalls, No needs to track to get here, There are crocodiles in the water, so swimming is not allowed.♥️☺️",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "It's amazing Waterfalls, No needs to track to get here, There are crocodiles in the water, so swimming is not allowed.♥️☺️",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Manab Mondal",
                        "uri": "https://www.google.com/maps/contrib/114475715928926735604/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjX7gbwVRGQM-jG0T7it-Tjcav951031IeNW22GJBt0rfTz50qOb=s128-c0x00000000-cc-rp-mo"
                    },
                    "publishTime": "2025-06-27T18:28:25.198180575Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2xKWE5tVjVRa2cwYkROSE1UWllWWFJKU0dGcmJVRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2xKWE5tVjVRa2cwYkROSE1UWllWWFJKU0dGcmJVRRAB!2m1!1s0x3bbfbd4c54c04655:0x623633fe068aab4b"
                },
                {
                    "name": "places/ChIJVUbAVEy9vzsRS6uKBv4zNmI/reviews/ChdDSUhNMG9nS0VJQ0FnSUNfOXRMZXN3RRAB",
                    "relativePublishTimeDescription": "5 months ago",
                    "rating": 4,
                    "text": {
                        "text": "As we missed Dudsagar falls due to distance during our Goa Trip, we thought of at least visiting this Falls.\n\nYes, we made a wise decision, there's a nearby temple and ancient caves. A must visit place.\n\nThere are small shops selling, Flowers for temples, and Sugarcane juice and Lime Soda and seasonal fruits. Try those also.\n\nBut please note, you cannot take a bath here.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "As we missed Dudsagar falls due to distance during our Goa Trip, we thought of at least visiting this Falls.\n\nYes, we made a wise decision, there's a nearby temple and ancient caves. A must visit place.\n\nThere are small shops selling, Flowers for temples, and Sugarcane juice and Lime Soda and seasonal fruits. Try those also.\n\nBut please note, you cannot take a bath here.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Jayaraj S",
                        "uri": "https://www.google.com/maps/contrib/107626765950013044560/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUqD1uYWpSxuzjZZ6M_aueL1mK0ykxRv7-WTjfDW-xAbxnMa0gCdw=s128-c0x00000000-cc-rp-mo-ba7"
                    },
                    "publishTime": "2025-01-16T03:31:42.808759Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnSUNfOXRMZXN3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnSUNfOXRMZXN3RRAB!2m1!1s0x3bbfbd4c54c04655:0x623633fe068aab4b"
                },
                {
                    "name": "places/ChIJVUbAVEy9vzsRS6uKBv4zNmI/reviews/ChdDSUhNMG9nS0VJQ0FnSUNQLWF5RXRnRRAB",
                    "relativePublishTimeDescription": "7 months ago",
                    "rating": 4,
                    "text": {
                        "text": "Truly a hidden gem in Goa. It's better to visit during the monsoon or after monsoon.\nThe gorgeous waterfall is a must visit place in Goa.\nParking isn't available over there, so you need to park your vehicle at your own risk.\nYou need to walk beside of a house and there you could found out steps to get down.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Truly a hidden gem in Goa. It's better to visit during the monsoon or after monsoon.\nThe gorgeous waterfall is a must visit place in Goa.\nParking isn't available over there, so you need to park your vehicle at your own risk.\nYou need to walk beside of a house and there you could found out steps to get down.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Vyshnu Jayakumar",
                        "uri": "https://www.google.com/maps/contrib/102915578649046472390/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWJ4Igbfh_-BCTqXy8btobLfo3Ro584k2f2k9plFJhdoWunjVOl=s128-c0x00000000-cc-rp-mo-ba5"
                    },
                    "publishTime": "2024-11-26T05:30:43.555624Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnSUNQLWF5RXRnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnSUNQLWF5RXRnRRAB!2m1!1s0x3bbfbd4c54c04655:0x623633fe068aab4b"
                }
            ]
        },
        {
            "id": "ChIJlYXf6YXAvzsR-byEcRSW9fA",
            "displayName": {
                "text": "Joseph Bar",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJlYXf6YXAvzsR-byEcRSW9fA/reviews/ChdDSUhNMG9nS0VJQ0FnTUNvM2FfRHJ3RRAB",
                    "relativePublishTimeDescription": "2 months ago",
                    "rating": 5,
                    "text": {
                        "text": "It's a good place to hangout, the best part about visiting this place is the surroundings. It's located in the vicinity of a very famous place in old goa named as \"Fontainhas\".\n\nIt can be quite problematic to find a place to sit as it has not so much space for many guests. But still it's worth the wait for your turn.\n\nThese days it has gained so much popularity on Instagram as \"the oldest bar of goa.\"\nThis place is purely a retro style which you must visit if you are in old goa.\nIn short \"Good - Vibes.\"",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "It's a good place to hangout, the best part about visiting this place is the surroundings. It's located in the vicinity of a very famous place in old goa named as \"Fontainhas\".\n\nIt can be quite problematic to find a place to sit as it has not so much space for many guests. But still it's worth the wait for your turn.\n\nThese days it has gained so much popularity on Instagram as \"the oldest bar of goa.\"\nThis place is purely a retro style which you must visit if you are in old goa.\nIn short \"Good - Vibes.\"",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Saurav Kumar",
                        "uri": "https://www.google.com/maps/contrib/111215402829960828674/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUEKLtasIY_y8XJ-zqDHT-qcv9lF1PZqXTvG4EMMdOHBwDrSF4=s128-c0x00000000-cc-rp-mo-ba3"
                    },
                    "publishTime": "2025-04-19T08:44:11.476708Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VJQ0FnTUNvM2FfRHJ3RRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VJQ0FnTUNvM2FfRHJ3RRAB!2m1!1s0x3bbfc085e9df8595:0xf0f596147184bcf9"
                },
                {
                    "name": "places/ChIJlYXf6YXAvzsR-byEcRSW9fA/reviews/ChZDSUhNMG9nS0VQeVE2ZnZla29ucUZnEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Must try fenis. Lots of beef and pork dishes which you will like depending on taste. One of our dishes was not available and we waited a lot of time for the same, but we anyways enjoyed sitting and sipping with some good music. Loved the seating and vibe",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Must try fenis. Lots of beef and pork dishes which you will like depending on taste. One of our dishes was not available and we waited a lot of time for the same, but we anyways enjoyed sitting and sipping with some good music. Loved the seating and vibe",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Rohit Aggarwal",
                        "uri": "https://www.google.com/maps/contrib/103520229699920099820/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjUcSVy-fSfMpPzh4z_yg1jrEOYOFfyvHX5whDjwN9Kg4shkyzcp=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-03T15:16:28.844826Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VQeVE2ZnZla29ucUZnEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VQeVE2ZnZla29ucUZnEAE!2m1!1s0x3bbfc085e9df8595:0xf0f596147184bcf9"
                },
                {
                    "name": "places/ChIJlYXf6YXAvzsR-byEcRSW9fA/reviews/ChZDSUhNMG9nS0VPNzh1Wmljek5hZ0ZnEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "If there (EVER) was a 5 STAR BAR (for me) it’s Joseph Bar. Just the sort of place you’d want to reunite with your old friends or maybe go with your partner who likes (and doesn’t mind) an old vintage tiny setup. The people serving at the Joseph are fabulous. How I wish it could stay open beyond 11-11.30PM.\n\nPS: if you’re someone who likes a more posh vibe, this ain’t the right place for you. Don’t forget to try the Cafaer Chicken (it’s got a homemade taste to it)",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "If there (EVER) was a 5 STAR BAR (for me) it’s Joseph Bar. Just the sort of place you’d want to reunite with your old friends or maybe go with your partner who likes (and doesn’t mind) an old vintage tiny setup. The people serving at the Joseph are fabulous. How I wish it could stay open beyond 11-11.30PM.\n\nPS: if you’re someone who likes a more posh vibe, this ain’t the right place for you. Don’t forget to try the Cafaer Chicken (it’s got a homemade taste to it)",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Sidhant Madan",
                        "uri": "https://www.google.com/maps/contrib/112184151996304202306/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjV-nbJuSCUk0xofUON7lbnFVcO6cLDtGZL6uVlMTA2XhKMAPcwb=s128-c0x00000000-cc-rp-mo-ba3"
                    },
                    "publishTime": "2025-05-22T19:47:37.943243Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VPNzh1Wmljek5hZ0ZnEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VPNzh1Wmljek5hZ0ZnEAE!2m1!1s0x3bbfc085e9df8595:0xf0f596147184bcf9"
                },
                {
                    "name": "places/ChIJlYXf6YXAvzsR-byEcRSW9fA/reviews/ChZDSUhNMG9nS0VOZVB3OEMtdmJ2RlF3EAE",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "As always we enjoyed the visit to Joseph’s bar again . A very old restaurant with a 100 years old look , slightly dark inside , walls give an old look as people have written their messages on the wall . Off lately they have tied up with community kitchen and now you get good Goan food also at a very reasonable price .",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "As always we enjoyed the visit to Joseph’s bar again . A very old restaurant with a 100 years old look , slightly dark inside , walls give an old look as people have written their messages on the wall . Off lately they have tied up with community kitchen and now you get good Goan food also at a very reasonable price .",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "ANIL Warokar",
                        "uri": "https://www.google.com/maps/contrib/104375179632366766274/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocL8oVAufko38OXCv5zALHpHoHAuzmDtO4i4K8bBYvmI4OX6eg=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-06-10T08:16:47.033610Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VOZVB3OEMtdmJ2RlF3EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VOZVB3OEMtdmJ2RlF3EAE!2m1!1s0x3bbfc085e9df8595:0xf0f596147184bcf9"
                },
                {
                    "name": "places/ChIJlYXf6YXAvzsR-byEcRSW9fA/reviews/ChZDSUhNMG9nS0VONlY2Skc0LXJyTllBEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "A very famous place in Panjim and no longer a hidden gem as everyone wants to visit here.\nThe place was packed on a Sunday night and I was surprised as most of the adjoining bars and restaurants were at max 50% occupied.\nThe food menu is very small but decent , so if you plan tobhave more than a couple of drinks ( like we had ) then it's better you eat something before coming here.\nThe drinks ate v reasonably priced and we ended up having quite a few on their menu.\nOwner is a very good person and staff is also very prompt in their service.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "A very famous place in Panjim and no longer a hidden gem as everyone wants to visit here.\nThe place was packed on a Sunday night and I was surprised as most of the adjoining bars and restaurants were at max 50% occupied.\nThe food menu is very small but decent , so if you plan tobhave more than a couple of drinks ( like we had ) then it's better you eat something before coming here.\nThe drinks ate v reasonably priced and we ended up having quite a few on their menu.\nOwner is a very good person and staff is also very prompt in their service.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Prakul Verma",
                        "uri": "https://www.google.com/maps/contrib/109326301041388901982/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXAOh2o3ihXLr7AYxnK4KJ7_yMFKVbfy0CzvUccsdoa4oTgAtU=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-02T04:48:44.416876Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VONlY2Skc0LXJyTllBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VONlY2Skc0LXJyTllBEAE!2m1!1s0x3bbfc085e9df8595:0xf0f596147184bcf9"
                }
            ],
            "priceRange": {
                "startPrice": {
                    "currencyCode": "INR",
                    "units": "200"
                },
                "endPrice": {
                    "currencyCode": "INR",
                    "units": "600"
                }
            }
        },
        {
            "id": "ChIJXVO8PxvqvzsRta823KYFYvE",
            "displayName": {
                "text": "Snow Park, Goa",
                "languageCode": "en"
            },
            "reviews": [
                {
                    "name": "places/ChIJXVO8PxvqvzsRta823KYFYvE/reviews/ChZDSUhNMG9nS0VLR0VfWnI4Mk1DeWJ3EAE",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 5,
                    "text": {
                        "text": "Really wanted to experience Snow of any kind? Visit this chilly place. A fun, snowy experience with activities like ice slides, sledging, and snowball fights, as well as a DJ playing Bollywood music. Its a family-friendly atmosphere, the friendly staff, and the affordable ticket price which includes all necessary thermal wear. Shoutout to the Photographers working at that cold temperature taking best photos at every positions. The music is a little loud and it can get a little crowded. Other than that, Its awesome. There's also Warm Tea and some food stall after you exit the snow park and you can also purchase the physical photos clicked inside at a very cheap rate. Must Visit!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Really wanted to experience Snow of any kind? Visit this chilly place. A fun, snowy experience with activities like ice slides, sledging, and snowball fights, as well as a DJ playing Bollywood music. Its a family-friendly atmosphere, the friendly staff, and the affordable ticket price which includes all necessary thermal wear. Shoutout to the Photographers working at that cold temperature taking best photos at every positions. The music is a little loud and it can get a little crowded. Other than that, Its awesome. There's also Warm Tea and some food stall after you exit the snow park and you can also purchase the physical photos clicked inside at a very cheap rate. Must Visit!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Srinath Bhandarkar",
                        "uri": "https://www.google.com/maps/contrib/100825776850581888345/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjXHp929Z0bceo6KNxQ-7FomU75W9x97nh471KHwZTalH-a0jLcl=s128-c0x00000000-cc-rp-mo-ba3"
                    },
                    "publishTime": "2025-06-12T13:01:59.647958Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VLR0VfWnI4Mk1DeWJ3EAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VLR0VfWnI4Mk1DeWJ3EAE!2m1!1s0x3bbfea1b3fbc535d:0xf16205a6dc36afb5"
                },
                {
                    "name": "places/ChIJXVO8PxvqvzsRta823KYFYvE/reviews/ChdDSUhNMG9nS0VOS1YxY2VVazgyZnhRRRAB",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 4,
                    "text": {
                        "text": "The price of the ticket and the rides available do not coordinate, the name Park is named, but it is not too big, it is very small and the inner environment is very cool, people should be tolerated so that they can play longer, so that children can play longer, and the maximum number of children should be provided to the children. Once you can visit, but you cannot wait for more than half an hour ... the facilities for the fistula to heat up, ie the facilities for shaking ....",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "The price of the ticket and the rides available do not coordinate, the name Park is named, but it is not too big, it is very small and the inner environment is very cool, people should be tolerated so that they can play longer, so that children can play longer, and the maximum number of children should be provided to the children. Once you can visit, but you cannot wait for more than half an hour ... the facilities for the fistula to heat up, ie the facilities for shaking ....",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Kanifnath Suryawanshi (Kanif Sir)",
                        "uri": "https://www.google.com/maps/contrib/105549843956072598598/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjU9rPh0rmZ3v3-yTPWWS6l3hPEGG4_hkVCvgCTYAQlScbEfhqkJ=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-05-21T14:14:28.586123Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VOS1YxY2VVazgyZnhRRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VOS1YxY2VVazgyZnhRRRAB!2m1!1s0x3bbfea1b3fbc535d:0xf16205a6dc36afb5"
                },
                {
                    "name": "places/ChIJXVO8PxvqvzsRta823KYFYvE/reviews/ChdDSUhNMG9nS0VKS2MzZVhacnQzcmhnRRAB",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Snow Park Goa is located near the famous Calangute Beach, just a lane behind Tito's Club. The park operates from 11:00 AM to 7:00 PM. Weekends can be a bit crowded, but it's still manageable. The entry fee is ₹500 per person for everyone above 3 years of age or over 3 feet tall. Please note that children below 3 years and individuals with heart ailments are not allowed.\n\nThe entry fee includes gum boots, jackets, gloves, and caps — all necessary for enjoying the chilly environment inside the Snow Park.\n\nInside, you can have a lot of fun with friends and family. The disco lights and music create a lively party atmosphere. You can also get memorable photos clicked — either with your own phone or by using the professional photographer available there. They charge only ₹50 per photo, and you'll receive both a printed copy and the digital version on WhatsApp.\n\nAttractions include slides, an igloo, a throne, a climbing wall, and snowmen — all made of real ice. There's also an ice bar where you can enjoy a drink in a unique setting.\n\nYou can easily spend 1 to 1.5 hours here having fun. If you're nearby, it's definitely worth a visit — kids will especially love it!\n\nOutside the snow area, there's a café and bar offering reasonably priced snacks, tea, and coffee.\n\nOne thing to keep in mind is that parking can be a challenge during peak times. If the main parking is full, you may need to use the nearby paid parking. However, the good news is that they reimburse half of the parking cost (₹100).\n\nOverall, it's a great place to spend quality time with loved ones and make some beautiful memories.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Snow Park Goa is located near the famous Calangute Beach, just a lane behind Tito's Club. The park operates from 11:00 AM to 7:00 PM. Weekends can be a bit crowded, but it's still manageable. The entry fee is ₹500 per person for everyone above 3 years of age or over 3 feet tall. Please note that children below 3 years and individuals with heart ailments are not allowed.\n\nThe entry fee includes gum boots, jackets, gloves, and caps — all necessary for enjoying the chilly environment inside the Snow Park.\n\nInside, you can have a lot of fun with friends and family. The disco lights and music create a lively party atmosphere. You can also get memorable photos clicked — either with your own phone or by using the professional photographer available there. They charge only ₹50 per photo, and you'll receive both a printed copy and the digital version on WhatsApp.\n\nAttractions include slides, an igloo, a throne, a climbing wall, and snowmen — all made of real ice. There's also an ice bar where you can enjoy a drink in a unique setting.\n\nYou can easily spend 1 to 1.5 hours here having fun. If you're nearby, it's definitely worth a visit — kids will especially love it!\n\nOutside the snow area, there's a café and bar offering reasonably priced snacks, tea, and coffee.\n\nOne thing to keep in mind is that parking can be a challenge during peak times. If the main parking is full, you may need to use the nearby paid parking. However, the good news is that they reimburse half of the parking cost (₹100).\n\nOverall, it's a great place to spend quality time with loved ones and make some beautiful memories.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Abhishek Srivastava",
                        "uri": "https://www.google.com/maps/contrib/109179783993286524944/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWrcE4W8wzPIxN55ehcsvErfHDL1ZniL9B5s_G9BIZOdKaGcJnZWA=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-05-17T12:33:37.655885Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChdDSUhNMG9nS0VKS2MzZVhacnQzcmhnRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChdDSUhNMG9nS0VKS2MzZVhacnQzcmhnRRAB!2m1!1s0x3bbfea1b3fbc535d:0xf16205a6dc36afb5"
                },
                {
                    "name": "places/ChIJXVO8PxvqvzsRta823KYFYvE/reviews/ChZDSUhNMG9nS0VNaVo2dFdJb3R1QUlBEAE",
                    "relativePublishTimeDescription": "a month ago",
                    "rating": 5,
                    "text": {
                        "text": "Had a great time at Snow Park today!\nThe place was super fun and well-maintained. From snow slides to the artificial snowfall, everything was exciting and definitely worth the visit. The staff were friendly and made sure everyone was safe and having a good time. It’s a great spot to chill (literally 😄) and enjoy something different, especially if you’re looking to escape the heat. Would totally recommend it for families, friends, or even a solo fun day. Definitely planning to go again!",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Had a great time at Snow Park today!\nThe place was super fun and well-maintained. From snow slides to the artificial snowfall, everything was exciting and definitely worth the visit. The staff were friendly and made sure everyone was safe and having a good time. It’s a great spot to chill (literally 😄) and enjoy something different, especially if you’re looking to escape the heat. Would totally recommend it for families, friends, or even a solo fun day. Definitely planning to go again!",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "Munir Sayed",
                        "uri": "https://www.google.com/maps/contrib/103165764444800625154/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a/ACg8ocJX-Zq854tpHtPudnLBIijBQvRtgYpsOwzQ4nOhUwB4_sLS2Q=s128-c0x00000000-cc-rp-mo-ba4"
                    },
                    "publishTime": "2025-05-14T14:49:06.303754Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=ChZDSUhNMG9nS0VNaVo2dFdJb3R1QUlBEAE&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sChZDSUhNMG9nS0VNaVo2dFdJb3R1QUlBEAE!2m1!1s0x3bbfea1b3fbc535d:0xf16205a6dc36afb5"
                },
                {
                    "name": "places/ChIJXVO8PxvqvzsRta823KYFYvE/reviews/Ci9DQUlRQUNvZENodHljRjlvT2psMVZGcHNaM1ZRZFZKQk9GZHJXbm96Tm1Gb1RVRRAB",
                    "relativePublishTimeDescription": "3 weeks ago",
                    "rating": 1,
                    "text": {
                        "text": "Extremely disappointing experience at Snow Park Goa. The ticket price is way too high for what you actually get inside — definitely not value for money. The space is shockingly small, more like stepping into a supply chain cold storage room than an actual amusement facility. It feels cramped, with barely anything exciting or fun to explore.\n\nThere are no special snow props or interactive setups that you’d expect in a snow-themed attraction. The entire setup feels very basic and unimaginative — a few random ice blocks and some snow scattered around, with very little effort in ambiance or entertainment. If you're expecting a magical winter experience or even a family-friendly play zone, this is not it.\n\nThe \"activities\" are too limited, and honestly, it seemed like a place put together with minimum investment just to cash in on tourists. It's frustrating when you spend a significant amount, especially with kids, and walk out feeling completely underwhelmed.\n\nWould not recommend this place to anyone — whether you're a local or a tourist. There are far better things to do and places to see in Goa. Snow Park Goa seriously needs a revamp or a price cut — preferably both.",
                        "languageCode": "en"
                    },
                    "originalText": {
                        "text": "Extremely disappointing experience at Snow Park Goa. The ticket price is way too high for what you actually get inside — definitely not value for money. The space is shockingly small, more like stepping into a supply chain cold storage room than an actual amusement facility. It feels cramped, with barely anything exciting or fun to explore.\n\nThere are no special snow props or interactive setups that you’d expect in a snow-themed attraction. The entire setup feels very basic and unimaginative — a few random ice blocks and some snow scattered around, with very little effort in ambiance or entertainment. If you're expecting a magical winter experience or even a family-friendly play zone, this is not it.\n\nThe \"activities\" are too limited, and honestly, it seemed like a place put together with minimum investment just to cash in on tourists. It's frustrating when you spend a significant amount, especially with kids, and walk out feeling completely underwhelmed.\n\nWould not recommend this place to anyone — whether you're a local or a tourist. There are far better things to do and places to see in Goa. Snow Park Goa seriously needs a revamp or a price cut — preferably both.",
                        "languageCode": "en"
                    },
                    "authorAttribution": {
                        "displayName": "krrazie",
                        "uri": "https://www.google.com/maps/contrib/115449923601038629469/reviews",
                        "photoUri": "https://lh3.googleusercontent.com/a-/ALV-UjWotP9yfo2bTWxfWOFwDlOSe2yxrQnmC7D1tFS8Ink8b2MAGWKA9Q=s128-c0x00000000-cc-rp-mo-ba6"
                    },
                    "publishTime": "2025-06-06T09:15:07.937250809Z",
                    "flagContentUri": "https://www.google.com/local/review/rap/report?postId=Ci9DQUlRQUNvZENodHljRjlvT2psMVZGcHNaM1ZRZFZKQk9GZHJXbm96Tm1Gb1RVRRAB&d=17924085&t=1",
                    "googleMapsUri": "https://www.google.com/maps/reviews/data=!4m6!14m5!1m4!2m3!1sCi9DQUlRQUNvZENodHljRjlvT2psMVZGcHNaM1ZRZFZKQk9GZHJXbm96Tm1Gb1RVRRAB!2m1!1s0x3bbfea1b3fbc535d:0xf16205a6dc36afb5"
                }
            ]
        }
    ]
}
    
    try:
        

        # Initialize the robust profiler with configuration
        profiler = RobustCityProfiler(config)
        
        # Run the comprehensive analysis
        final_profile = profiler.run_comprehensive_analysis(json_string)

        if final_profile and "error" not in final_profile:
            print("\n" + "="*80)
            print("      ROBUST CITY PROFILING ANALYSIS RESULTS      ")
            print("="*80)
            
            city_profile = final_profile["city_profile"]
            place_categorization = final_profile["place_categorization"]
            analysis_summary = final_profile["analysis_summary"]
            
            print(f"\n🏙️ City Profile for {city_profile['name']}:")
            print(f"🎯 Top Tags: {', '.join(city_profile['top_tags'])}")
            print(f"📊 Confidence Score: {city_profile['confidence_score']}")
            
            if city_profile.get('category_scores'):
                print(f"\n📈 Category Scores:")
                for category, score in sorted(city_profile['category_scores'].items(), 
                                            key=lambda x: x[1], reverse=True):
                    print(f"   • {category.title()}: {score:.3f}")
            
            print(f"\n🗂️ Place Categorization:")
            for category, places in place_categorization.items():
                if places:  # Only show categories with places
                    print(f"\n  📍 {category}:")
                    for place in places:
                        print(f"     • {place}")
            
            print(f"\n📋 Analysis Summary:")
            print(f"   • Total Places: {analysis_summary['total_places_analyzed']}")
            print(f"   • Reviews Processed: {analysis_summary['total_reviews_processed']}")
            print(f"   • Processing Time: {analysis_summary['processing_time_seconds']}s")
            
            if analysis_summary.get('sentiment_distribution'):
                print(f"   • Sentiment Distribution: {dict(analysis_summary['sentiment_distribution'])}")
            
            if analysis_summary.get('languages_detected'):
                print(f"   • Languages Detected: {dict(analysis_summary['languages_detected'])}")
            
            if final_profile.get("warnings"):
                print(f"\n⚠️ Warnings: {final_profile['warnings']['total_error_count']} issues encountered")
                
            print("\n" + "="*80)
            
        else:
            print("\n❌ Analysis failed:")
            print(f"Error: {final_profile.get('error', 'Unknown error')}")
            if 'details' in final_profile:
                print(f"Details: {final_profile['details']}")
            
    except json.JSONDecodeError as e:
        print(f"Error: The provided JSON string is invalid. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()