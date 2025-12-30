"""
说唱技巧识别模块
每个技巧都有独立的检测函数，方便维护和扩展
"""
import re
import numpy as np
from scipy import sparse

# 尝试导入NLTK用于词性标注（方法2：词性识别）
try:
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # 确保必要的NLTK数据已下载
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        try:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        except:
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                try:
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                except:
                    pass
except ImportError:
    NLTK_AVAILABLE = False


# ============================================================================
# Full-Court Shot (+5.0 Points) - 高风险高回报的bar
# ============================================================================

# 稀有强词（高权重，明确指向Full-Court Shot）
FULL_COURT_STRONG_KEYWORDS = {
    'documentary', 'investigation', 'autopsy', 'missing persons',
    'museum', 'exhibit', 'exhibition', 'gallery', 'showcase',
    'footage', 'premiere', 'premiered', 'aired', 'broadcast',
    'disappearance', 'vanished', 'erased', 'flatlined', 'folded', 'logged off'
}

# 中等强度词（中权重）
FULL_COURT_MEDIUM_KEYWORDS = {
    'camera', 'clip', 'clipped', 'replay', 'recording', 'video', 'tape', 'film',
    'display', 'installation', 'opened', 'opening', 'unveiled', 'debuted', 'launched',
    'spirit', 'energy', 'soul', 'ego', 'presence', 'essence', 'hype',
    'disappeared', 'faded', 'gone', 'empty', 'body', 'left', 'went'
}

# 泛词（低权重或可能扣分）
FULL_COURT_WEAK_KEYWORDS = {
    'moment', 'scene', 'shot', 'frame', 'got', 'made', 'created', 'built',
    'released', 'published', 'streamed', 'pulled', 'grabbed', 'took'
}

# 击溃语义关键词（必须出现至少一类）
VICTIM_STATE_KEYWORDS = {
    # 消失/死亡/崩溃
    'vanished', 'disappeared', 'erased', 'flatlined', 'folded', 'logged off',
    'died', 'dead', 'gone', 'faded', 'crumbled', 'collapsed', 'vanish',
    # 羞辱画面/状态
    'empty chair', 'autopsy', 'missing persons', 'wanted', 'judgment', 'verdict',
    'coroner', 'morgue', 'body bag', 'crime scene', 'evidence',
    # 抽象概念（可能被击溃）
    'confidence', 'ego', 'pride', 'bravado', 'hype', 'aura', 'momentum',
    'tough talk', 'voice cracked', 'stumbled', 'paused', 'folded', 'froze',
    # 动作/状态（表示被击溃）
    'repossessed', 'stolen', 'deleted', 'rerouted', 'packed', 'shipped',
    'rage quit', 'witness protection', 'mission failure', 'loss',
    # 环境/氛围变化（表示对手被压制）
    'went quiet', 'went silent', 'silence', 'quiet', 'stunned', 'shook'
}

# 捕获/构建模式（单行紧凑版，避免正则空格问题，允许省略主语I）
FULL_COURT_CREATION_PATTERNS = [
    r'\b(pulled|grabbed|captured|caught|snapped|clipped|recorded|ran|ran\s+the|got|saved)\s+(the|that|this|a|an)?\s*\w*\s*(footage|replay|recording|video|tape|film|moment|shot|scene|clip|feed)',  # 添加got, saved, feed
    r'\b(pulled|grabbed|captured|caught|snapped|clipped|recorded|got)\s+(the|that|this|a|an)?\s*(gps\s+history)',
    r'\b(built|constructed|assembled|erected|created)\s+(the|a|an)?\s*\w*\s*(museum|exhibit|exhibition|display|gallery|showcase|installation)',
    r'\b(printed|wrote|filed|issued|opened|logged|tracked|made|created|pushed|posted|curated|read)\s+(the|a|an)?\s*\w*\s*(receipts|paperwork|chart|report|bulletin|ticket|update|order|reel|lesson|timeline|exhibit|chapter|study|screen\s+capture|verdict|alert)',
    r'\b(pulled|grabbed|captured|caught|snapped|clipped|recorded|printed|wrote|filed|issued|opened|logged|tracked|made|created|pushed|posted|curated|read|got)\s+(the|that|this|a|an)?\s*(gps\s+history)',
]

# 直接捕获信号（不要求后面跟footage类名词）
FULL_COURT_DIRECT_CAPTURE_PATTERNS = [
    r'\b(screen\s+recorded|screen\s+captured|screen\s+capture|recorded)\b',  # screen recorded/captured直接命中
]

# 标题命名模式（改进：不依赖引号和大写，用连接词/结尾为边界）
FULL_COURT_TITLE_PATTERNS = [
    r'\b(titled|named|called|labeled|tagged)\s+(it|this|that|the)\s+([^\s]+(?:\s+[^\s]+)*?)(?:\s+(?:and|then|so|$))',  # 捕获到and/then/so/结尾
    r'\b(titled|named|called|labeled|tagged)\s+(it|this|that|the)\s+([^\s]+(?:\s+[^\s]+)*?)(?=\s+(?:and|then|so|dropped|released|opened|published|aired|premiered|streamed|uploaded|unveiled|put\s+out|broadcast|$))',  # 更严格的边界，包括aired, unveiled, put out, broadcast
    r'\b(titled|named|called|labeled|tagged)\s+(it|this|that|the)\s+\w+',  # 备用：简单匹配
]

# 发布/开放模式（单行紧凑版，补充series/season/cut/report/case file）
FULL_COURT_RELEASE_PATTERNS = [
    r'\b(dropped|released|unveiled|debuted|aired|broadcast|streamed|premiered|uploaded|published|put\s+out)\s+(the|a|an)?\s*\w*\s*(documentary|investigation|film|video|tape|episode|show|report|findings|case\s+file|season|evidence|patch\s+notes|logs|screenshot|travel\s+doc|storm\s+report|unboxing|mission\s+documentary|educational\s+documentary|directors\s+cut|series|trial\s+documentary|full\s+series|cut|case|doc|file|special)',  # 添加put out, special
    r'\b(opened|opening|launched|presented|attached|included|showed|pinned|set|played|held|posted|unveiled)\s+(the|a|an)?\s*\w*\s*(exhibit|exhibition|display|museum|gallery|show|conference|documentary|evidence|logs|screenshot|receipts|everywhere|update|on\s+repeat|it\s+downtown|it)',  # 添加unveiled it
]

# 击溃语义模式（必须出现）
VICTIM_STATE_PATTERNS = [
    # 直接消失/死亡词汇
    r'\b(vanished|disappeared|erased|flatlined|folded|logged\s+off|died|dead|gone|faded|crumbled|collapsed|vanish)',
    # 羞辱画面
    r'\b(empty\s+chair|autopsy|missing\s+persons|wanted|judgment|verdict|coroner|morgue|body\s+bag|crime\s+scene)',
    # 抽象概念 + 消失动作（更严格的匹配，避免"energy was different"这种）
    r'\b(spirit|energy|soul|ego|presence|essence|hype|confidence|pride|bravado|aura|momentum|tough\s+talk|voice|bars|act|legend|performance|scoreboard)\s+(left|went|fled|vanished|disappeared|faded|was\s+(already\s+)?gone|is\s+gone|got\s+(stolen|deleted|rerouted|repossessed|packed|shipped)|already\s+gone|cracked|stumbled|paused|folded|rage\s+quit|ended|turned\s+into|did\s+not\s+even\s+need)',
    r'\b(confidence|ego|pride|bravado|hype|aura|momentum|tough\s+talk|voice|bars|act|legend|performance)\s+(vanish|vanished|disappeared|gone|stolen|deleted|rerouted|repossessed|cracked|stumbled|paused|folded|rage\s+quit|ended|turned\s+into)',
    # 避免匹配"energy was different"这种弱表达
    r'\b(energy|vibe|momentum)\s+was\s+(different|off|strange)',  # 这个模式用于排除，不是匹配
    # 消失+媒体
    r'\b(erased|vanished|disappeared|faded|gone)\s+(on|from|in)\s+(camera|film|tape|video|recording)',
    # 被转化为证据/文件
    r'\b(turned|converted|transformed)\s+\w+\s+(into|to)\s+(evidence|case\s+file|paperwork|report|document)',
    # 被发送/进入保护/失败
    r'\b(sent|went|got)\s+\w+\s+(into|to)\s+(witness\s+protection|evidence|mission\s+failure)',
    # 特定击溃表达
    r'\b(confidence|ego|pride)\s+rage\s+quit',
    r'\b(scoreboard|stage)\s+(did\s+not\s+even\s+need|turned\s+into)',
    r'\b(momentum|act)\s+got\s+(rerouted|packed|shipped)',
    r'\b(way\s+you\s+faded|class\s+is\s+gonna\s+study\s+that\s+loss)',
    r'\b(stage|courtroom)\s+turned\s+into',
    r'\b(momentum)\s+got\s+rerouted',
    r'\b(way\s+you\s+faded|you\s+faded)',
    # 直接动作（you + 击溃动作）
    r'\b(you|your)\s+(stumbled|froze|paused|folded|crashed|went\s+quiet|vanished|disappeared|erased|faded|gone)',
    r'\b(you|your)\s+(confidence|ego|pride|bravado|hype|aura|momentum)\s+(crashed|vanished|disappeared|erased|faded|gone|stolen|deleted)',
]

# 反向惩罚词（出现这些词会扣分，特别是标题中的）
VAGUENESS_PENALTY_WORDS = {
    'something', 'stuff', 'maybe', 'interesting', 'chapter', 'timeline', 
    'highlights', 'things', 'whatever', 'kinda', 'sort of', 'a moment',
    'a thing', 'a chapter', 'the timeline', 'the exhibit', 'the highlights',
    'watch this', 'news today', 'big event', 'evidence', 'missing', 'update complete',
    'bug found', 'need help', 'travel time', 'money talk', 'lessons',
    'camera time', 'a moment in time', 'some kind of', 'somehow', 'did a thing',
    'felt like', 'somewhere', 'was loud', 'exists', 'changed', 'moved around',
    'felt expensive', 'showed everything', 'so deep', 'needed', 'could measure',
    'was long', 'was off', 'was different', 'was strange'
}

def detect_victim_state_by_pos(text: str) -> bool:
    """
    方法2：通过词性识别击溃语义
    
    识别模式：
    1. 过去式动词（VBD/VBN）表示消失/失败动作
    2. 某些形容词（JJ）表示失败状态
    3. 某些名词（NN）表示失败/消失概念
    4. 特定语法结构：you/your + 过去式动词
    
    返回:
        bool: 是否检测到击溃语义
    """
    if not NLTK_AVAILABLE:
        return False
    
    try:
        # 分词和词性标注
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # 击溃语义的动词（原形和过去式/过去分词）
        defeat_verbs = {
            # 原形
            'vanish', 'disappear', 'erase', 'fade', 'collapse', 'crumble',
            'fold', 'fail', 'lose', 'fall', 'crash', 'break', 'destroy',
            'eliminate', 'defeat', 'overcome', 'conquer', 'crush', 'stumble',
            'freeze', 'pause', 'stop', 'end', 'die', 'flatline', 'quit',
            # 过去式/过去分词
            'vanished', 'disappeared', 'erased', 'faded', 'collapsed', 'crumbled',
            'folded', 'failed', 'lost', 'fell', 'crashed', 'broke', 'broken',
            'destroyed', 'eliminated', 'defeated', 'overcame', 'overcome', 'conquered',
            'crushed', 'stumbled', 'froze', 'frozen', 'paused', 'stopped', 'ended',
            'died', 'dead', 'flatlined', 'quit', 'quitted'
        }
        
        # 击溃语义的形容词
        defeat_adjectives = {
            'gone', 'dead', 'empty', 'lost', 'defeated', 'broken', 'crushed',
            'destroyed', 'eliminated', 'finished', 'over', 'done', 'stunned',
            'shocked', 'silent', 'quiet'
        }
        
        # 击溃语义的名词
        defeat_nouns = {
            'loss', 'defeat', 'failure', 'collapse', 'end', 'death', 'autopsy',
            'evidence', 'verdict', 'judgment', 'missing', 'disappearance'
        }
        
        # 检查词性模式
        for i, (word, pos) in enumerate(pos_tags):
            word_lower = word.lower()
            
            # 1. 过去式动词（VBD）或过去分词（VBN）表示消失/失败
            if pos in ['VBD', 'VBN'] and word_lower in defeat_verbs:
                # 检查上下文：前面是否有you/your，或后面有表示击溃的补语
                if i > 0:
                    prev_word = pos_tags[i-1][0].lower()
                    if prev_word in ['you', 'your']:
                        return True
                if i < len(pos_tags) - 1:
                    next_pos = pos_tags[i+1][1]
                    if next_pos in ['NN', 'NNS', 'DT']:  # 后面跟名词或限定词
                        return True
                return True
            
            # 2. 形容词（JJ）表示失败状态
            if pos == 'JJ' and word_lower in defeat_adjectives:
                # 检查是否在"was/got/became + 形容词"结构中
                if i > 0:
                    prev_word = pos_tags[i-1][0].lower()
                    if prev_word in ['was', 'got', 'became', 'turned', 'went']:
                        return True
                return True
            
            # 3. 名词（NN/NNS）表示失败/消失概念
            if pos in ['NN', 'NNS'] and word_lower in defeat_nouns:
                # 检查是否在"into/to + 名词"或"the/a/an + 名词"结构中
                if i > 0:
                    prev_word = pos_tags[i-1][0].lower()
                    if prev_word in ['into', 'to', 'the', 'a', 'an']:
                        return True
                return True
            
            # 4. 特定结构：you/your + 过去式动词
            if word_lower in ['you', 'your'] and i < len(pos_tags) - 1:
                next_word, next_pos = pos_tags[i+1]
                if next_pos in ['VBD', 'VBN'] and next_word.lower() in defeat_verbs:
                    return True
        
        # 5. 检查连续模式：抽象概念 + 过去式动词
        for i in range(len(pos_tags) - 1):
            word1, pos1 = pos_tags[i]
            word2, pos2 = pos_tags[i+1]
            
            # 抽象概念（NN） + 过去式动词（VBD/VBN）
            abstract_nouns = {'confidence', 'ego', 'pride', 'hype', 'aura', 'momentum', 'spirit', 'energy'}
            if (pos1 in ['NN', 'NNS'] and word1.lower() in abstract_nouns and 
                pos2 in ['VBD', 'VBN'] and word2.lower() in defeat_verbs):
                return True
        
        return False
        
    except Exception:
        # 如果NLTK处理失败，返回False（回退到方法1）
        return False

def normalize_text(text: str) -> str:
    """
    文本归一化：
    1. 全部转小写
    2. 把连字符合并：screen-recorded -> screen recorded
    3. 多空格压成一个
    4. 删除引号（考虑无标点文本）
    """
    s = text.lower()
    # 连字符合并
    s = re.sub(r'([a-z]+)-([a-z]+)', r'\1 \2', s)
    # 多空格压成一个
    s = re.sub(r'\s+', ' ', s)
    # 删除引号
    s = s.replace('"', '').replace("'", '')
    return s.strip()

def detect_full_court_shot(text: str) -> float:
    """
    检测 Full-Court Shot 技巧
    高风险高回报的bar，意外的重击，将事件转化为媒体内容
    
    核心要求：
    1. 必须包含击溃语义（victim state）
    2. 必须包含完整流程（creation + naming + release）
    3. 标题命名是关键亮点
    
    语法特征（最强信号）：
    - 三连动词句式：I + VERB1 + OBJ1 + VERB2 + OBJ2 + VERB3 + OBJ3
    - 两段结构：时间框架 + 对手状态 + I + 三连动作链
    - 连接词偏好：and, and then, so I
    - 人称配置：you/your (受害者) + I (制作人)
    """
    # 文本归一化
    s_lower = normalize_text(text)
    score = 0.0
    
    # ========================================================================
    # 硬门槛1：击溃语义必须出现（victim state）
    # ========================================================================
    has_victim_state = False
    
    # ========================================================================
    # 方法1：关键词库识别（原有方法）
    # ========================================================================
    # 检查击溃语义关键词（支持多词短语）
    for kw in VICTIM_STATE_KEYWORDS:
        if ' ' in kw:
            # 多词短语，使用正则匹配
            if re.search(r'\b' + re.escape(kw) + r'\b', s_lower):
                has_victim_state = True
                break
        else:
            # 单词，直接查找
            if kw in s_lower:
                has_victim_state = True
                break
    
    # 检查击溃语义模式
    if not has_victim_state:
        for pattern in VICTIM_STATE_PATTERNS:
            # 跳过排除模式（用于反向检查）
            if r'was\s+(different|off|strange)' in pattern:
                continue
            if re.search(pattern, s_lower):
                has_victim_state = True
                break
    
    # ========================================================================
    # 方法2：词性识别（新增方法，与方法1并列）
    # ========================================================================
    if not has_victim_state and NLTK_AVAILABLE:
        has_victim_state = detect_victim_state_by_pos(text)
    
    # 如果还没有找到击溃语义，检查标题中是否包含击溃语义词汇
    # 这对于简洁句式（省略主语I）特别重要，标题本身可能包含击溃语义
    if not has_victim_state:
        # 提取标题文本
        title_text = ""
        for pattern in FULL_COURT_TITLE_PATTERNS:
            match = re.search(pattern, s_lower)
            if match:
                # 尝试从匹配的group中提取标题（如果pattern有捕获组）
                if len(match.groups()) >= 3:
                    # 第三个group通常是标题
                    title_text = match.group(3).strip()
                elif len(match.groups()) >= 1:
                    # 如果只有一个group，可能是标题
                    title_text = match.group(1).strip()
                
                # 如果从group中提取失败，从match.end()开始提取
                if not title_text or len(title_text) < 2:
                    title_start = match.end()
                    # 找到标题结束位置（下一个动词或连接词）
                    title_end_match = re.search(r'\s+(and|then|so|dropped|released|opened|published|aired|premiered|streamed|uploaded|$)', s_lower[title_start:])
                    if title_end_match:
                        title_text = s_lower[title_start:title_start + title_end_match.start()].strip()
                    else:
                        # 如果没有找到结束标记，提取到句子结尾或最多50字符
                        remaining_text = s_lower[title_start:]
                        # 找到下一个主要动词作为边界
                        next_verb_match = re.search(r'\s+(dropped|released|opened|published|aired|premiered|streamed|uploaded)', remaining_text)
                        if next_verb_match:
                            title_text = remaining_text[:next_verb_match.start()].strip()
                        else:
                            title_text = remaining_text[:50].strip()  # 最多50字符
                break
        
        # 检查标题中是否包含击溃语义关键词
        if title_text:
            # 检查标题中的击溃语义关键词（包括完整短语和单词）
            title_victim_keywords = {
                'disappearance', 'disappeared', 'vanished', 'erased', 'folded',
                'logout', 'heist', 'fold', 'autopsy', 'missing', 'exact frame',
                'where', 'went', 'fall', 'exhibit', 'guilty', 'wrong turn',
                'severe', 'collapse', 'return', 'sender', 'abort', 'fold on',
                'the logout', 'energy heist', 'the fold', 'live disappearance',
                'rap autopsy', 'exact frame', 'where his', 'where your',
                'the fall of', 'exhibit a', 'missing since', 'wrong turn',
                'severe confidence', 'return to sender', 'abort confidence',
                'fold on beat'
            }
            # 先检查完整短语（多词）
            for kw in title_victim_keywords:
                if ' ' in kw:
                    # 多词短语，使用正则匹配
                    if re.search(r'\b' + re.escape(kw) + r'\b', title_text):
                        has_victim_state = True
                        break
                else:
                    # 单词，直接查找
                    if kw in title_text:
                        has_victim_state = True
                        break
    
    # 排除弱表达（如"energy was different"）- 这些不算击溃语义
    weak_expressions = [
        r'\b(energy|vibe|momentum)\s+was\s+(different|off|strange)',
        r'\b(moment|performance)\s+(was|did)\s+(loud|a\s+thing)',
        r'\b(crowd|room)\s+(reacted|felt)\s+(somehow|some\s+kind\s+of)',
        r'\b(room|stage|moment)\s+felt\s+(some\s+kind\s+of|like)',
        r'\b(energy|vibe|momentum)\s+(was|moved)\s+(different|around|somewhere)',
        r'\b(performance|moment)\s+did\s+a\s+thing',
        r'\b(moment|footage)\s+was\s+(loud|exists)',
        r'\b(round|things)\s+(was|were)\s+(intense|changed)',
        r'\b(confidence|vibe)\s+had\s+a\s+moment',
        r'\b(something)\s+was\s+off',
        r'\b(energy)\s+moved\s+around',
        r'\b(performance)\s+felt\s+expensive',
        r'\b(cameras)\s+showed\s+everything',
        r'\b(story)\s+was\s+so\s+deep',
        r'\b(highlight\s+reel)\s+was\s+long',
        r'\b(stage)\s+felt\s+like\s+history',
        r'\b(hype)\s+was\s+somewhere',
    ]
    for weak_pattern in weak_expressions:
        if re.search(weak_pattern, s_lower):
            has_victim_state = False
            break
    
    # 硬门槛：必须有击溃语义才能继续（移除流程兜底逻辑，防止误报）
    # 只有真正有击溃语义的句子才能通过，不能仅凭流程就判定
    if not has_victim_state:
        return 0.0
    
    # ========================================================================
    # 硬门槛2：流程检测（升级为硬规则兜底）
    # ========================================================================
    # 检测创建步骤（包括直接捕获信号）
    has_creation = False
    for pattern in FULL_COURT_CREATION_PATTERNS:
        if re.search(pattern, s_lower):
            has_creation = True
            break
    
    # 检测直接捕获信号（screen recorded等，不要求后面跟footage类名词）
    if not has_creation:
        for pattern in FULL_COURT_DIRECT_CAPTURE_PATTERNS:
            if re.search(pattern, s_lower):
                has_creation = True
                break
    
    # 检测命名步骤（标题是关键亮点）
    has_naming = False
    for pattern in FULL_COURT_TITLE_PATTERNS:
        if re.search(pattern, s_lower):
            has_naming = True
            break
    
    # 检测发布步骤
    has_release = False
    for pattern in FULL_COURT_RELEASE_PATTERNS:
        if re.search(pattern, s_lower):
            has_release = True
            break
    
    # 流程硬规则：如果三步流程齐全，给高分（但必须已有击溃语义）
    if has_creation and has_naming and has_release:
        # 完整三步流程 + 击溃语义，直接给高分
        score = 0.85  # 硬规则兜底，确保流程齐全且有击溃语义的句子不被漏检
    elif has_creation and has_naming:
        # 两步流程（包含命名），基础分中等
        score = 0.4
    else:
        # 没有完整流程或缺少命名，返回低分
        if not (has_creation and has_naming):
            return 0.0
    
    # ========================================================================
    # 关键词加权评分（稀有词权重高，泛词权重低）
    # ========================================================================
    # 稀有强词（高权重）
    strong_count = sum(1 for kw in FULL_COURT_STRONG_KEYWORDS if kw in s_lower)
    score += min(strong_count * 0.12, 0.25)  # 每个0.12分，上限0.25
    
    # 中等强度词（中权重）
    medium_count = sum(1 for kw in FULL_COURT_MEDIUM_KEYWORDS if kw in s_lower)
    score += min(medium_count * 0.05, 0.15)  # 每个0.05分，上限0.15
    
    # 泛词（低权重，可能扣分）
    weak_count = sum(1 for kw in FULL_COURT_WEAK_KEYWORDS if kw in s_lower)
    if weak_count > 3:  # 泛词太多，可能是普通叙述
        score -= min((weak_count - 3) * 0.03, 0.1)  # 扣分
    
    # 弱对象惩罚（timeline/highlights/chapter等，防止误报）
    weak_objects = {'timeline', 'highlights', 'chapter', 'story', 'part', 'segment'}
    weak_object_count = sum(1 for obj in weak_objects if obj in s_lower)
    if weak_object_count > 0:
        # 如果只有弱对象，没有强媒体/强展览对象，降低上限
        strong_media_objects = {'documentary', 'investigation', 'report', 'exhibit', 'exhibition', 'museum', 'film', 'video', 'case file'}
        has_strong_object = any(obj in s_lower for obj in strong_media_objects)
        if not has_strong_object:
            score = min(score, 0.5)  # 弱对象且无强对象，上限0.5
    
    # ========================================================================
    # 反向惩罚：模糊词检测（特别是标题中的）
    # ========================================================================
    # 检查标题中的模糊词（最严重，直接扣分或拒绝）
    title_vagueness = False
    title_vagueness_words = []
    for pattern in FULL_COURT_TITLE_PATTERNS:
        match = re.search(pattern, s_lower)
        if match:
            # 提取标题文本（从titled/named/called到and/then/so/结尾）
            title_start = match.end()
            # 找到标题结束位置（and/then/so/结尾）
            title_end_match = re.search(r'\s+(and|then|so|dropped|released|opened|published|aired|$)', s_lower[title_start:])
            if title_end_match:
                title_text = s_lower[title_start:title_start + title_end_match.start()]
            else:
                title_text = s_lower[title_start:title_start + 50]  # 最多50字符
            
            # 检查标题中是否有模糊词
            for word in VAGUENESS_PENALTY_WORDS:
                if word in title_text:
                    title_vagueness = True
                    title_vagueness_words.append(word)
                    break
            if title_vagueness:
                break
    
    # 标题中有模糊词，直接大幅扣分或拒绝
    if title_vagueness:
        score -= 0.6  # 标题中有模糊词，大幅扣分（从0.4提高到0.6）
        # 如果分数被扣到很低，直接拒绝
        if score < 0.3:
            return 0.0
    
    # 检查文本其他位置的模糊词
    vagueness_count = sum(1 for word in VAGUENESS_PENALTY_WORDS if word in s_lower)
    if vagueness_count > 0 and not title_vagueness:
        score -= min(vagueness_count * 0.2, 0.4)  # 其他位置的模糊词（从0.15提高到0.2）
    
    # 检查弱表达模式（即使通过了击溃语义检查，也要扣分）
    weak_expression_penalties = [
        (r'\b(some\s+kind\s+of|somehow|did\s+a\s+thing|felt\s+like|was\s+off|was\s+different)', 0.3),
        (r'\b(made\s+content\s+about|noticed\s+it|saw\s+it)', 0.2),
        (r'\b(felt|reacted|moved|changed|existed|showed|needed|could\s+measure|was\s+long)', 0.15),
    ]
    for weak_pattern, penalty in weak_expression_penalties:
        if re.search(weak_pattern, s_lower):
            score -= penalty
            break
    
    # ========================================================================
    # 语法特征检测（最强信号）- 基于语言学分析
    # ========================================================================
    
    # 1. 检测"三连动词句式"：I + VERB1 + OBJ1 + VERB2 + OBJ2 + VERB3 + OBJ3
    # 这是Full-Court Shot最典型的语法特征
    # 也支持省略主语I的简洁句式：VERB1 + OBJ1 + VERB2 + OBJ2 + VERB3 + OBJ3
    verb1_pattern = r'\b(pulled|grabbed|captured|caught|snapped|clipped|recorded|ran|got|printed|wrote|filed|issued|opened|logged|tracked|made|created|pushed|posted|built|constructed|assembled|erected)\s+'
    verb2_pattern = r'\b(titled|named|called|labeled|tagged)\s+(it|this|that|the)\s+'
    verb3_pattern = r'\b(dropped|released|unveiled|debuted|aired|broadcast|streamed|premiered|uploaded|published|opened|launched|presented)\s+'
    
    # 检测三连动词模式（允许中间有连接词和少量其他词）
    # 支持有主语I和省略主语I两种情况
    triple_verb_pattern_with_i = r'\bi\s+' + verb1_pattern + r'[^.]*?' + verb2_pattern + r'[^.]*?' + verb3_pattern
    triple_verb_pattern_no_i = verb1_pattern + r'[^.]*?' + verb2_pattern + r'[^.]*?' + verb3_pattern  # 省略主语I
    
    if re.search(triple_verb_pattern_with_i, s_lower, re.IGNORECASE) or re.search(triple_verb_pattern_no_i, s_lower, re.IGNORECASE):
        score += 0.3  # 三连动词句式是强信号
    
    # 2. 检测时间框架开头（让句子更像新闻播报）
    time_frame_patterns = [
        r'\b(by\s+the\s+end\s+of|the\s+moment|as\s+soon\s+as|after\s+that\s+round|mid\s+battle|soon\s+as|the\s+second|when\s+the|that\s+round|one\s+beat\s+switch\s+later|by\s+round\s+(two|three|four|five|\d+)|once)',
    ]
    has_time_frame = any(re.search(p, s_lower) for p in time_frame_patterns)
    if has_time_frame:
        score += 0.1
    
    # 3. 检测两段结构：时间框架 + 对手状态 + I + 三连动作
    # 检测"your X was Y"或"your X vanished/disappeared"后跟"I + 动作"
    two_part_structure = re.search(
        r'(?:by\s+the\s+end|the\s+moment|after|mid|when|that\s+round|soon\s+as).*?your\s+\w+\s+(?:was|got|left|went|vanished|disappeared|erased|faded|gone|got\s+(?:stolen|deleted|rerouted|repossessed|packed|shipped))[^.]*?\bi\s+',
        s_lower,
        re.IGNORECASE
    )
    if two_part_structure:
        score += 0.15  # 两段结构是典型特征
    
    # 4. 检测连接词偏好（and, and then, so I）
    connector_patterns = [
        r'\band\s+and\s+and',  # 连续and
        r'\band\s+then',  # and then
        r'\bso\s+i\s+',  # so I
        r'\band\s+i\s+',  # and I
    ]
    connector_count = sum(1 for p in connector_patterns if re.search(p, s_lower))
    if connector_count > 0:
        score += min(connector_count * 0.05, 0.1)  # 连接词强化流程感
    
    # 5. 检测人称配置：you/your (受害者) + I (制作人)
    has_you = bool(re.search(r'\b(you|your)\s+', s_lower))
    has_i = bool(re.search(r'\bi\s+', s_lower))
    if has_you and has_i:
        score += 0.1  # 人称配置制造压迫感
    
    # 6. 检测标题语法特征（名词短语或从句缩写式标题）
    title_grammar_patterns = [
        r'where\s+(his|the|your)\s+\w+\s+went',  # Where His X Went
        r'the\s+day\s+(he|you|they)\s+\w+',  # The Day He Y
        r'live\s+\w+',  # Live Noun
        r'missing\s+since\s+\w+',  # Missing Since Noun
        r'\w+\s+autopsy',  # X Autopsy
        r'\w+\s+heist',  # X Heist
        r'the\s+fall\s+of',  # The Fall Of
        r'exhibit\s+[a-z]',  # Exhibit A
        r'the\s+exact\s+frame',  # The Exact Frame
    ]
    title_grammar_count = sum(1 for p in title_grammar_patterns if re.search(p, s_lower))
    if title_grammar_count > 0:
        score += min(title_grammar_count * 0.08, 0.15)  # 标题语法特征
    
    # ========================================================================
    # 额外加分项
    # ========================================================================
    # 检测"消失+媒体"的强组合
    if re.search(r'\b(erased|vanished|disappeared|faded|gone)\s+(on|from|in)\s+(camera|film|tape|video|recording)', s_lower):
        score += 0.1
    
    # 检测意外的转折
    if re.search(r'\b(but|yet|however|though|although)\s+', s_lower):
        score += 0.05
    
    # 确保分数在合理范围内
    return max(0.0, min(score, 1.0))


# ============================================================================
# Slam Dunk (+4.25 Points) - 震撼性的重击
# ============================================================================

SLAM_DUNK_KEYWORDS = {
    'legendary', 't-shirt', 'remember', 'die', 'death', 'kill', 'murder', 'dead',
    'only', 'every time', 'somebody', 'gotta', 'gonna', 'must', 'first',
    'devastating', 'haymaker', 'instant', 'explosive', 'reaction'
}

SLAM_DUNK_PATTERNS = [
    r'\bonly\s+\w+\s+on\s+(a|the)\s+\w+',
    r'\bevery\s+time\s+they?\s+\w+',
    r'\bsomebody\s+(gotta|gonna|must|has\s+to)\s+\w+',
    r'\b(gotta|gonna|must)\s+(die|kill|murder)',
    r'\b(legendary|famous|great)\s+.*\s+(but|yet|only)',
]

def detect_slam_dunk(text: str) -> float:
    """
    检测 Slam Dunk 技巧
    震撼性的重击，瞬间爆炸反应
    """
    s_lower = text.lower()
    score = 0.0
    
    # 关键词匹配
    keyword_count = sum(1 for kw in SLAM_DUNK_KEYWORDS if kw in s_lower)
    score += min(keyword_count * 0.12, 0.4)
    
    # 模式匹配
    pattern_matches = sum(1 for pattern in SLAM_DUNK_PATTERNS if re.search(pattern, s_lower))
    score += min(pattern_matches * 0.25, 0.5)
    
    # 检测强烈的对比结构
    if re.search(r'\b(only|just|merely)\s+\w+\s+.*\s+(but|yet|however)', s_lower):
        score += 0.2
    
    return min(score, 1.0)


# ============================================================================
# Half-Court Shot (+3.75 Points) - 创意风险命中
# ============================================================================

HALF_COURT_KEYWORDS = {
    'trauma', 'triumph', 'flipped', 'turned', 'transformed', 'rearrange', 'direction',
    'crooked', 'compass', 'deep', 'crowd', 'check', 'cheer', 'creative', 'risk',
    'lands', 'perfectly', 'philosophy', 'metaphor', 'abstract'
}

HALF_COURT_PATTERNS = [
    r'\b(flipped|turned|transformed|changed)\s+\w+\s+(into|to|from)',
    r'\b(bar|line|verse)\s+so\s+deep',
    r'\b(rearrange|change|shift|alter)\s+\w+\s+(direction|course|path)',
    r'\bcrooked\s+\w+',
    r'\b(crowd|audience|people)\s+(ain\'?t|don\'?t)\s+know\s+if',
    r'\b(should|could|might)\s+(cheer|applaud|react)\s+.*\s+or\s+(check|help|worry)',
]

def detect_half_court_shot(text: str) -> float:
    """
    检测 Half-Court Shot 技巧
    创意风险命中，完美落地
    """
    s_lower = text.lower()
    score = 0.0
    
    # 关键词匹配
    keyword_count = sum(1 for kw in HALF_COURT_KEYWORDS if kw in s_lower)
    score += min(keyword_count * 0.15, 0.4)
    
    # 模式匹配
    pattern_matches = sum(1 for pattern in HALF_COURT_PATTERNS if re.search(pattern, s_lower))
    score += min(pattern_matches * 0.2, 0.4)
    
    # 检测情感转换（trauma -> triumph）
    if re.search(r'\b(trauma|pain|hurt|suffering)\s+.*\s+(into|to|became)\s+(triumph|victory|success|win)', s_lower):
        score += 0.2
    
    return min(score, 1.0)


# ============================================================================
# Alley-Oop/Assist (+3.5 Points) - 多人配合
# ============================================================================

ALLEY_OOP_KEYWORDS = {
    'line', 'him', 'up', 'cool', 'team', 'partner', 'assist', 'tag', 'together',
    'sync', 'crowd', 'synced', 'moment', 'eruption', 'coordinate', 'collaborate'
}

ALLEY_OOP_PATTERNS = [
    r'[\'"]\s*\w+[\'"]\s*:',
    r'\b(line|set|get|put)\s+(him|them|it)\s+up',
    r'\bcool\s*[—\-–]\s*(i\'?ll|i\s+will)',
    r'\b(partner|teammate|crew)\s*:',
    r'\b(tag|team|together|with)\s+\w+',
]

def detect_alley_oop(text: str) -> float:
    """
    检测 Alley-Oop/Assist 技巧
    多人配合，团队协作
    """
    s_lower = text.lower()
    score = 0.0
    
    # 关键词匹配
    keyword_count = sum(1 for kw in ALLEY_OOP_KEYWORDS if kw in s_lower)
    score += min(keyword_count * 0.12, 0.3)
    
    # 模式匹配
    pattern_matches = sum(1 for pattern in ALLEY_OOP_PATTERNS if re.search(pattern, s_lower))
    score += min(pattern_matches * 0.3, 0.5)
    
    # 检测引号内的对话（多人配合的标记）
    quoted_dialogue = len(re.findall(r'[\'"].*?[\'"]', text))
    if quoted_dialogue >= 2:
        score += 0.2
    
    # 检测指令性语言
    if re.search(r'\b(line|set|get|put)\s+(him|them|it|you)\s+up', s_lower):
        score += 0.2
    
    return min(score, 1.0)


# ============================================================================
# 主函数：检测所有技巧
# ============================================================================

def detect_rap_techniques(texts):
    """
    检测所有说唱技巧特征
    
    返回 (n, 4) 稀疏矩阵，每列对应一个技巧的强度分数 (0-1):
    - 第0列: Full-Court Shot
    - 第1列: Slam Dunk
    - 第2列: Half-Court Shot
    - 第3列: Alley-Oop/Assist
    
    参数:
        texts: 文本列表（字符串列表）
    
    返回:
        scipy.sparse.csr_matrix: 稀疏矩阵，形状为 (len(texts), 4)
    """
    rows = []
    
    for text in texts:
        s = text if isinstance(text, str) else ""
        
        # 调用各个技巧检测函数
        full_court_score = detect_full_court_shot(s)
        slam_dunk_score = detect_slam_dunk(s)
        half_court_score = detect_half_court_shot(s)
        alley_oop_score = detect_alley_oop(s)
        
        rows.append([full_court_score, slam_dunk_score, half_court_score, alley_oop_score])
    
    A = np.asarray(rows, dtype=float)
    return sparse.csr_matrix(A)


# ============================================================================
# 技巧信息（用于显示和分析）
# ============================================================================

TECHNIQUE_INFO = {
    "full_court_shot": {
        "name": "Full-Court Shot",
        "points": "+5.0 Points",
        "description": "高风险高回报的bar，意外的重击"
    },
    "slam_dunk": {
        "name": "Slam Dunk",
        "points": "+4.25 Points",
        "description": "震撼性的重击，瞬间爆炸反应"
    },
    "half_court_shot": {
        "name": "Half-Court Shot",
        "points": "+3.75 Points",
        "description": "创意风险命中，完美落地"
    },
    "alley_oop": {
        "name": "Alley-Oop/Assist",
        "points": "+3.5 Points",
        "description": "多人配合，团队协作"
    }
}

def get_technique_info(technique_key: str) -> dict:
    """获取技巧信息"""
    return TECHNIQUE_INFO.get(technique_key, {})

