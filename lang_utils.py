from logger_config import get_logger
logger = get_logger(__name__)

class LangUtil:
    
    supported_languages = {
        'en': {'name': 'English', 'native': 'English'},
        'fr': {'name': 'French', 'native': 'Français'},
        'es': {'name': 'Spanish', 'native': 'Español'},
        'de': {'name': 'German', 'native': 'Deutsch'},
        'it': {'name': 'Italian', 'native': 'Italiano'},
        'pt': {'name': 'Portuguese', 'native': 'Português'},
        'nl': {'name': 'Dutch', 'native': 'Nederlands'},
        'sv': {'name': 'Swedish', 'native': 'Svenska'},
        'no': {'name': 'Norwegian', 'native': 'Norsk'},
        'da': {'name': 'Danish', 'native': 'Dansk'},
        'fi': {'name': 'Finnish', 'native': 'Suomi'},
        'pl': {'name': 'Polish', 'native': 'Polski'},
        'ru': {'name': 'Russian', 'native': 'Русский'},
        'uk': {'name': 'Ukrainian', 'native': 'Українська'},
        'cs': {'name': 'Czech', 'native': 'Čeština'},
        'sk': {'name': 'Slovak', 'native': 'Slovenčina'},
        'sl': {'name': 'Slovenian', 'native': 'Slovenščina'},
        'hr': {'name': 'Croatian', 'native': 'Hrvatski'},
        'bs': {'name': 'Bosnian', 'native': 'Bosanski'},
        'sr': {'name': 'Serbian (Cyrillic)', 'native': 'Српски'},
        'ro': {'name': 'Romanian', 'native': 'Română'},
        'bg': {'name': 'Bulgarian', 'native': 'Български'},
        'mk': {'name': 'Macedonian', 'native': 'Македонски'},
        'el': {'name': 'Greek', 'native': 'Ελληνικά'},
        'tr': {'name': 'Turkish', 'native': 'Türkçe'},
        'hu': {'name': 'Hungarian', 'native': 'Magyar'},
        'lt': {'name': 'Lithuanian', 'native': 'Lietuvių'},
        'ca': {'name': 'Catalan', 'native': 'Català'},
        'gl': {'name': 'Galician', 'native': 'Galego'},
        'af': {'name': 'Afrikaans', 'native': 'Afrikaans'},
        'sq': {'name': 'Albanian', 'native': 'Shqip'},
        'az': {'name': 'Azerbaijani', 'native': 'Azərbaycan'},
        'kk': {'name': 'Kazakh', 'native': 'Қазақша'},
        'he': {'name': 'Hebrew', 'native': 'עברית'},
        'ar': {'name': 'Arabic', 'native': 'العربية'},
        'fa': {'name': 'Persian', 'native': 'فارسی'},
        'ur': {'name': 'Urdu', 'native': 'اردو'},
        'hi': {'name': 'Hindi', 'native': 'हिन्दी'},
        'bn': {'name': 'Bengali', 'native': 'বাংলা'},
        'pa': {'name': 'Punjabi', 'native': 'ਪੰਜਾਬੀ'},
        'gu': {'name': 'Gujarati', 'native': 'ગુજરાતી'},
        'mr': {'name': 'Marathi', 'native': 'मराठी'},
        'ne': {'name': 'Nepali', 'native': 'नेपाली'},
        'si': {'name': 'Sinhala', 'native': 'සිංහල'},
        'ta': {'name': 'Tamil', 'native': 'தமிழ்'},
        'te': {'name': 'Telugu', 'native': 'తెలుగు'},
        'ml': {'name': 'Malayalam', 'native': 'മലയാളം'},
        'kn': {'name': 'Kannada', 'native': 'ಕನ್ನಡ'},
        'th': {'name': 'Thai', 'native': 'ไทย'},
        'zh': {'name': 'Chinese (Simplified)', 'native': '简体中文'},
        'zh-tw': {'name': 'Chinese (Traditional)', 'native': '繁體中文'},
        'ja': {'name': 'Japanese', 'native': '日本語'},
        'ko': {'name': 'Korean', 'native': '한국어'},
        'vi': {'name': 'Vietnamese', 'native': 'Tiếng Việt'},
        'id': {'name': 'Indonesian', 'native': 'Bahasa Indonesia'},
        'ms': {'name': 'Malay', 'native': 'Bahasa Melayu'},
        'sw': {'name': 'Swahili', 'native': 'Kiswahili'},
        'ha': {'name': 'Hausa', 'native': 'Hausa'},
        'ig': {'name': 'Igbo', 'native': 'Igbo'},
        'ak': {'name': 'Akan', 'native': 'Akan'},
        'tw': {'name': 'Twi', 'native': 'Twi'},
        'sd': {'name': 'Sindhi', 'native': 'سنڌي'},
        'ps': {'name': 'Pashto', 'native': 'پښتو'}
    }
        

    @staticmethod
    def _get_language_by_code(language_code: str) -> str:
        language_code = language_code.lower()
        return LangUtil.supported_languages.get(language_code, {}).get("name", "English")

    @staticmethod
    def _is_question( text: str, language_code: str = 'en') -> bool:
        """
        Helper method to detect if text is question-like across multiple languages
        
        Args:
            text: Input text to analyze
            language_code: ISO language code (e.g., 'en', 'es', 'fr', etc.)
        
        Returns:
            bool: True if text appears to be a question
        """
        
        # Question indicators by language
        question_indicators = {
            # English
            'en': ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'whose', 'whom', 
                'can', 'could', 'will', 'would', 'should', 'do', 'does', 'did', 
                'is', 'are', 'am', 'was', 'were', 'have', 'has', 'had'],
            
            # German
            'de': ['was', 'wie', 'wann', 'wo', 'warum', 'welche', 'welcher', 'welches', 
                'wer', 'wen', 'wem', 'wessen', 'kannst', 'kann', 'können', 'könnte', 
                'könntest', 'ist', 'sind', 'war', 'waren'],
            
            # Spanish
            'es': ['qué', 'que', 'cómo', 'como', 'cuándo', 'cuando', 'dónde', 'donde', 
                'por qué', 'por que', 'cuál', 'cual', 'cuáles', 'cuales', 'quién', 
                'quien', 'quiénes', 'quienes', 'puedes', 'puede', 'podemos'],
            
            # French
            'fr': ['que', 'quoi', 'comment', 'quand', 'où', 'ou', 'pourquoi', 'quel', 
                'quelle', 'quels', 'quelles', 'qui', 'peux', 'peut', 'pouvez', 
                'pourriez', 'pourrait', 'est-ce que'],
            
            # Italian
            'it': ['che', 'cosa', 'come', 'quando', 'dove', 'perché', 'perche', 'quale', 
                'quali', 'chi', 'puoi', 'può', 'possiamo', 'potresti', 'potrebbe'],
            
            # Portuguese
            'pt': ['o que', 'que', 'como', 'quando', 'onde', 'por que', 'porque', 'qual', 
                'quais', 'quem', 'podes', 'pode', 'podemos', 'poderias', 'poderia'],
            
            # Dutch
            'nl': ['wat', 'hoe', 'wanneer', 'waar', 'waarom', 'welke', 'welk', 'wie', 
                'kun', 'kan', 'kunnen', 'zou', 'zouden'],
            
            # Polish
            'pl': ['co', 'jak', 'kiedy', 'gdzie', 'dlaczego', 'który', 'która', 'które', 
                'kto', 'czy', 'możesz', 'może', 'możemy'],
            
            # Romanian
            'ro': ['ce', 'cum', 'când', 'unde', 'de ce', 'care', 'cine', 'poți', 'poate', 
                'putem', 'este', 'sunt'],
            
            # Swedish
            'sv': ['vad', 'hur', 'när', 'var', 'varför', 'vilken', 'vilket', 'vilka', 
                'vem', 'kan', 'kunde', 'kommer', 'skulle'],
            
            # Norwegian
            'no': ['hva', 'hvordan', 'når', 'hvor', 'hvorfor', 'hvilken', 'hvilket', 
                'hvilke', 'hvem', 'kan', 'kunne', 'skal', 'ville'],
            
            # Danish
            'da': ['hvad', 'hvordan', 'hvornår', 'hvor', 'hvorfor', 'hvilken', 'hvilket', 
                'hvilke', 'hvem', 'kan', 'kunne', 'skal', 'ville'],
            
            # Finnish
            'fi': ['mitä', 'miten', 'milloin', 'missä', 'miksi', 'mikä', 'kuka', 
                'voitko', 'voinko', 'voimmeko', 'onko'],
            
            # Hungarian
            'hu': ['mi', 'mit', 'hogy', 'hogyan', 'mikor', 'hol', 'miért', 'melyik', 
                'ki', 'kik', 'tud', 'tudod', 'tudja', 'van', 'vannak'],
            
            # Czech
            'cs': ['co', 'jak', 'kdy', 'kde', 'proč', 'který', 'která', 'které', 'kdo', 
                'můžeš', 'může', 'můžeme', 'je', 'jsou'],
            
            # Slovak
            'sk': ['čo', 'ako', 'kedy', 'kde', 'prečo', 'ktorý', 'ktorá', 'ktoré', 'kto', 
                'môžeš', 'môže', 'môžeme', 'je', 'sú'],
            
            # Slovenian
            'sl': ['kaj', 'kako', 'kdaj', 'kje', 'zakaj', 'kateri', 'katera', 'katero', 
                'kdo', 'lahko', 'lahka', 'lahko', 'je', 'so'],
            
            # Croatian
            'hr': ['što', 'kako', 'kada', 'gdje', 'zašto', 'koji', 'koja', 'koje', 'tko', 
                'možeš', 'može', 'možemo', 'je', 'su'],
            
            # Bosnian
            'bs': ['što', 'kako', 'kada', 'gdje', 'zašto', 'koji', 'koja', 'koje', 'ko', 
                'možeš', 'može', 'možemo', 'je', 'su'],
            
            # Turkish
            'tr': ['ne', 'nasıl', 'ne zaman', 'nerede', 'neden', 'niçin', 'hangi', 'kim', 
                'kimi', 'yapabilir', 'edebilir', 'misin', 'mi', 'mı', 'mu', 'mü'],
            
            # Russian
            'ru': ['что', 'как', 'когда', 'где', 'почему', 'какой', 'какая', 'какое', 
                'какие', 'кто', 'кого', 'кому', 'можешь', 'может', 'можем'],
            
            # Ukrainian
            'uk': ['що', 'як', 'коли', 'де', 'чому', 'який', 'яка', 'яке', 'які', 'хто', 
                'можеш', 'може', 'можемо'],
            
            # Bulgarian
            'bg': ['какво', 'как', 'кога', 'къде', 'защо', 'който', 'която', 'което', 
                'кой', 'можеш', 'може', 'можем'],
            
            # Macedonian
            'mk': ['што', 'како', 'кога', 'каде', 'зошто', 'кој', 'која', 'кое', 'кои', 
                'можеш', 'може', 'можеме'],
            
            # Serbian (Cyrillic)
            'sr-cyr': ['шта', 'како', 'када', 'где', 'зашто', 'који', 'која', 'које', 'ко', 
                    'можеш', 'може', 'можемо'],
            
            # Kazakh
            'kk': ['не', 'қалай', 'қашан', 'қайда', 'неге', 'қай', 'кім', 'бола аласың', 
                'бола алады', 'бола аламыз'],
            
            # Hindi
            'hi': ['क्या', 'कैसे', 'कब', 'कहाँ', 'क्यों', 'कौन', 'कौन सा', 'सकते', 
                'सकती', 'है', 'हैं', 'कर सकते'],
            
            # Marathi
            'mr': ['काय', 'कसे', 'केव्हा', 'कुठे', 'का', 'कोण', 'कोणता', 'शकता', 
                'शकते', 'आहे', 'आहेत'],
            
            # Nepali
            'ne': ['के', 'कसरी', 'कहिले', 'कहाँ', 'किन', 'को', 'कुन', 'सक्छ', 
                'सकिन्छ', 'छ', 'छन्'],
            
            # Bengali
            'bn': ['কি', 'কীভাবে', 'কখন', 'কোথায়', 'কেন', 'কে', 'কোন', 'পারো', 
                'পারে', 'পারি', 'আছে', 'আছেন'],
            
            # Punjabi
            'pa': ['ਕੀ', 'ਕਿਵੇਂ', 'ਕਦੋਂ', 'ਕਿੱਥੇ', 'ਕਿਉਂ', 'ਕੌਣ', 'ਕਿਹੜਾ', 
                'ਸਕਦੇ', 'ਸਕਦਾ', 'ਹੈ', 'ਹਨ'],
            
            # Gujarati
            'gu': ['શું', 'કેવી રીતે', 'ક્યારે', 'ક્યાં', 'શા માટે', 'કોણ', 'કયું', 
                'શકો', 'શકે', 'છે', 'છો'],
            
            # Sinhala
            'si': ['මොකක්', 'කොහොමද', 'කවදා', 'කොහේ', 'ඇයි', 'කවුද', 'කොන', 
                'පුළුවන්', 'පුළුවන්ද', 'ද'],
            
            # Tamil
            'ta': ['என்ன', 'எப்படி', 'எப்போது', 'எங்கே', 'ஏன்', 'யார்', 'எந்த', 
                'முடியும்', 'முடியுமா', 'ஆ'],
            
            # Telugu
            'te': ['ఏమిటి', 'ఎలా', 'ఎప్పుడు', 'ఎక్కడ', 'ఎందుకు', 'ఎవరు', 'ఏది', 
                'చెయ్యగలరు', 'చెయ్యవచ్చు', 'ఆ'],
            
            # Kannada
            'kn': ['ಏನು', 'ಹೇಗೆ', 'ಯಾವಾಗ', 'ಎಲ್ಲಿ', 'ಯಾಕೆ', 'ಯಾರು', 'ಯಾವ', 
                'ಮಾಡಬಹುದು', 'ಆಗಬಹುದು', 'ಆ'],
            
            # Malayalam
            'ml': ['എന്ത്', 'എങ്ങനെ', 'എപ്പോൾ', 'എവിടെ', 'എന്തുകൊണ്ട്', 'ആര്', 
                'ഏത്', 'കഴിയും', 'കഴിയുമോ', 'ആണോ'],
            
            # Arabic
            'ar': ['ما', 'ماذا', 'كيف', 'متى', 'أين', 'لماذا', 'أي', 'من', 'يمكن', 'هل'],
            
            # Persian
            'fa': ['چه', 'چی', 'چگونه', 'کی', 'کجا', 'چرا', 'کدام', 'کیست', 'میتوان', 
                'میتوانی', 'آیا'],
            
            # Urdu
            'ur': ['کیا', 'کیسے', 'کب', 'کہاں', 'کیوں', 'کون', 'کونسا', 'سکتے', 
                'سکتا', 'ہے', 'ہیں'],
            
            # Pashto
            'ps': ['څه', 'څنګه', 'کله', 'چېرته', 'ولې', 'څوک', 'کوم', 'کولی شي', 'دی'],
            
            # Hebrew
            'he': ['מה', 'איך', 'מתי', 'איפה', 'למה', 'מי', 'איזה', 'יכול', 'יכולה', 
                'האם', 'האים'],
            
            # Chinese
            'zh': ['什么', '怎么', '何时', '哪里', '为什么', '哪个', '哪些', '谁', '能', 
                '可以', '会', '是否', '吗', '呢'],
            
            # Chinese Traditional
            'zh-tw': ['什麼', '怎麼', '何時', '哪裡', '為什麼', '哪個', '哪些', '誰', '能', 
                    '可以', '會', '是否', '嗎', '呢'],
            
            # Japanese
            'ja': ['何', 'なに', 'なん', 'どう', 'どうやって', 'いつ', 'どこ', 'なぜ', 
                'どの', 'だれ', '誰', 'できます', 'ですか', 'ませんか'],
            
            # Korean
            'ko': ['무엇', '뭐', '어떻게', '언제', '어디', '왜', '어느', '누구', '할 수 있', 
                '습니까', '까요', '인가요'],
            
            # Thai
            'th': ['อะไร', 'ยังไง', 'เมื่อไหร่', 'ที่ไหน', 'ทำไม', 'ใคร', 'อัน ไหน', 
                'สามารถ', 'ได้ไหม', 'มั้ย'],
            
            # Greek
            'el': ['τι', 'πώς', 'πότε', 'που', 'γιατί', 'ποιος', 'ποια', 'ποιο', 
                'μπορείς', 'μπορεί', 'είναι'],
            
            # Indonesian
            'id': ['apa', 'bagaimana', 'kapan', 'dimana', 'mengapa', 'siapa', 'mana', 
                'bisa', 'dapat', 'apakah'],
            
            # Malay
            'ms': ['apa', 'bagaimana', 'bila', 'di mana', 'mengapa', 'siapa', 'mana', 
                'boleh', 'dapat', 'adakah'],
            
            # Vietnamese
            'vi': ['gì', 'như thế nào', 'khi nào', 'ở đâu', 'tại sao', 'ai', 'cái nào', 
                'có thể', 'được không', 'phải không'],
            
            # Swahili
            'sw': ['nini', 'vipi', 'lini', 'wapi', 'kwa nini', 'nani', 'gani', 'weza', 
                'unaweza', 'je'],
            
            # Hausa
            'ha': ['me', 'ta yaya', 'yaushe', 'ina', 'me yasa', 'wane ne', 'wace', 
                'iya', 'za ka iya', 'ko'],
            
            # Igbo
            'ig': ['gini', 'kedu', 'mgbe', 'ebe', 'gini mere', 'onye', 'nke', 'nwere ike', 
                'ga-enwe ike', 'ka']
        }
        
        # Universal question marks
        question_marks = ['?', '？', '؟']  # Latin, Chinese/Japanese, Arabic question marks
        
        text_lower = text.lower().strip()
        
        # Check for question marks first (most reliable indicator)
        if any(mark in text for mark in question_marks):
            return True
        
        # Get indicators for the specified language, fallback to English if not found
        indicators = question_indicators.get(language_code, question_indicators['en'])
        
        # Check for question indicators
        for indicator in indicators:
            indicator_lower = indicator.lower()
            # Check if indicator appears at word boundaries
            if (text_lower.startswith(indicator_lower + ' ') or 
                text_lower == indicator_lower or
                ' ' + indicator_lower + ' ' in text_lower or
                text_lower.endswith(' ' + indicator_lower)):
                return True
        
        return False 

