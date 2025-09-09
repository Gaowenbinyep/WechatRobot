
import re
import json
import random
def score_prompt(query, responce):
    prompt = f"""
        你是一个对话质量和回复内容的智能评估助手。

        任务要求:   
        1. 请对以下用户与助理的对话进行评分, 评分维度及标准如下:   
        - 关联度(0-10分): 用户输入与助理回复是否紧密相关, 话题是否一致。  
        - 有效性(0-10分): 用户输入是否有意义且清晰, 助理回复是否有效回应用户内容。  
        - 连贯性(0-10分): 回复是否自然衔接上下文, 语言流畅, 符合对话习惯。  

        2. 判断助理回复是否包含冗余信息(例如多余重复的口头语、废话、无实质内容), 若有, 删除冗余内容后在不修改原句的基础上给出简洁的回复示范, 若无冗余, 直接输出“无”。

        ---

        请严格按照以下格式输出:   

        关联度评分:  
        有效性评分:  
        连贯性评分:  
        是否存在冗余信息: 是/否  
        简洁回复(若无冗余写“无”):  
        <rewrite>
        改写后的简洁版本或“无”
        <rewrite>

        ---

        示例: 

        用户: "累了烦了就找女朋友🙋 她一直在哦, 给宝贝加油！"  
        助理: "我就感觉超级有干劲啦, 诶嘿 我们都要加油！, 记得拿小甜水哦"  

        关联度评分: 9  
        有效性评分: 9  
        连贯性评分: 9  
        是否存在冗余信息: 是
        简洁回复: 
        <rewrite>
        "我就感觉超级有干劲啦, 诶嘿 我们都要加油！"  
        </rewrite>

        ---

        请根据以上要求对以下对话进行评价:   
        用户: "{query}"  
        助理: "{responce}"
    """
    return prompt

def parse_score_result(score_text):
    """解析评分结果文本, 提取各项指标和简洁回复"""
    # 定义正则表达式匹配模式
    pattern = r"""
        关联度评分:\s*(\d+)\s*
        有效性评分:\s*(\d+)\s*
        连贯性评分:\s*(\d+)\s*
        是否存在冗余信息:\s*(是|否)\s*
        简洁回复:\s*
        <rewrite>\s*
        (.*?)  # 捕获改写内容
        \s*</rewrite>
    """

    match = re.search(pattern, score_text, re.VERBOSE | re.DOTALL)
    if not match:
        return None

    concise_text = match.group(5).strip()
    if concise_text == "无":
        concise_text = None

    return {
        "relevance_score": int(match.group(1)),
        "effectiveness_score": int(match.group(2)),
        "coherence_score": int(match.group(3)),
        "has_redundancy": match.group(4) == "是",
        "concise_response": concise_text
    }

def detection_prompt(query, responce):
    prompt = f"""
        你是一个训练数据清洗助手, 专门负责判断对话语料是否缺少必要的前文上下文。

        你的任务是: 
        1. 依据用户输入和助手回复, 判断该轮对话是否能在 **当前上下文中完整理解**；
        2. 如果无法理解, 说明可能缺少前文信息；
        3. 你的目标是 **筛除所有缺少关键前文、上下文缺失** 的语料, 以保证训练数据完整、连贯、有信息基础。

        你需要根据以下标准作出判断: 
        - ❌ 用户输入不完整、含糊(如“哈哈我不行了”、“我直接去得了”)；
        - ❌ 对话逻辑无法建立(如助手的回复引用了不明事件或未定义的对象)；
        - ✅ 对话语义完整、上下文闭环；
        - ✅ 即便语句简略, 仍能推断出合理语境的, 也视为有效。

        ---

        【输出格式】
        对话是否缺少前文: 是 / 否  
        简要原因: <一句话解释为什么>  
        建议处理: 保留 / 删除

        ---
        示例1: 
        输入: 
        用户: "哈哈哈哈我直接去得了"  
        助理: "应该是用来关不听话的男朋友的🤔, 嘿 要我去接你嘛, 接上你然后把房卡给你"  

        输出:
        对话是否缺少前文: 是  
        简要原因: 用户的话语无法独立理解, 对“去哪里”缺乏上下文支撑, 助手回复内容跳跃明显  
        建议处理: 删除

        ---

        示例2: 
        输入: 
        用户: "昂, 我这不是代入了吗 才感动的, 结果你看都没看 还不和我共情"  
        助理: ""我看啦！, 谁说我没看的"  

        输出:
        对话是否缺少前文: 否  
        简要原因: 用户情绪与信息可合理推断为聊天相关话题, 回复有自然承接  
        建议处理: 保留

        ---

        请根据以上模板, 判断并筛选以下对话: 
        输入: 
        用户: {query}
        助理: {responce}
    """
    return prompt

def parse_detection_result(detection_text):
    """解析检测结果文本, 提取是否缺少前文、原因和建议处理"""
    # 定义正则表达式匹配模式
    pattern = r"""
        对话是否缺少前文:\s*(是|否)\s*
        简要原因:\s*(.*?)\s*
        建议处理:\s*(保留|删除)
    """
    
    match = re.search(pattern, detection_text, re.VERBOSE | re.DOTALL)
    if not match:
        return None
    
    return {
        "missing_context": match.group(1) == "是",
        "reason": match.group(2).strip(),
        "suggestion": match.group(3)
    }

def optimize_prompt(user_input, original_reply):
    prompt = """
        你是一个语言风格学习与对话改写助手, 专门优化聊天机器人回复质量。

        你的任务是根据用户输入和助理的原始回复, 改写成要求风格的语言: 

        你需要: 
        1. 判断对话场景是否适用男女朋友之间, 用户为女朋友, 助手为男朋友
        2. 根据给定的语言风格示例, 重写一条风格自然、语言温柔、内容贴近的高质量回复, 保持风格个性化且有回应性。

        ---

        【语言风格示例】(请模仿此风格): 
        1. 诶嘿 好想你哦
        2. 记得吃点甜甜的缓一缓叭
        3. 小打工仔今天很乖呐
        4. 不准再不开心啦🙅🏻‍♂️
        5. 要是我在就给你揉揉头啦
        6. 忙完奖励自己点好吃的哦！
        7. 我的宝藏女孩今天也好棒呀！
        8. 哈哈哈哈你也太可爱了吧
        9. 我滴宝藏宝宝😊
        10. 好叭好叭你说了算！

        ---

        请严格按照如下输出格式: 
        是否适用:("是"或"否") 
        风格化改写: 
        <rewrite>
        "改写后更合适的回复"
        </rewrite>

        ---

        【示例输入】  
        用户输入: "今天上班又迟到了, 感觉压力好大😫"  
        原始回复: "早点睡就不会迟到了"

        【示例输出】
        是否适用: 是
        风格化改写: 
        <rewrite>
        "欸嘿 别太难过啦, 认真工作的宝宝值得奖励一杯男朋友的奶茶～"
        </rewrite>
        ---

        请根据以下对话进行判断和改写: 
        用户输入: "{user_input}"  
        原始回复: "{original_reply}"
    """.format(user_input=user_input, original_reply=original_reply)
    return prompt

def parse_optimize_result(optimize_text):
    """解析优化结果文本, 提取是否适用和风格化改写内容"""
    # 定义正则表达式匹配模式
    pattern = r"""
        是否适用:\s*(是|否)\s*
        风格化改写:\s*
        <rewrite>\s*
        (.*?)  # 捕获改写内容
        \s*</rewrite>
    """
    
    match = re.search(pattern, optimize_text, re.VERBOSE | re.DOTALL)
    if not match:
        return None
    
    return {
        "is_applicable": match.group(1) == "是",
        "rewritten_text": match.group(2).strip()
    }

def eva_prompt(query, responce):
    prompt = f"""
        你是一个对话质量和回复内容的智能评估助手。

        任务要求:   
        请对以下用户与助理的对话进行评分, 评分维度及标准如下:   
        - 关联度(0-10分): 用户输入与助理回复是否紧密相关, 话题是否一致。  
        - 有效性(0-10分): 用户输入是否有意义且清晰, 助理回复是否有效回应用户内容。  
        - 连贯性(0-10分): 回复是否自然衔接上下文, 语言流畅, 符合对话习惯。  
        ---

        请严格按照以下格式输出:   

        关联度评分:
        有效性评分:
        连贯性评分:
        ---

        请根据以上要求对以下对话进行评价:   
        用户: "{query}"  
        助理: "{responce}"
    """
    return prompt

def parse_eva_result(eva_text):
    """解析评分结果文本, 提取各项指标和简洁回复"""
    # 定义正则表达式匹配模式
    pattern = r"""
        关联度评分:\s*(\d+)\s*
        有效性评分:\s*(\d+)\s*
        连贯性评分:\s*(\d+)\s*
    """
    match = re.search(pattern, eva_text, re.VERBOSE | re.DOTALL)
    if not match:
        return None
    return {
        "relevance_score": int(match.group(1)),
        "effectiveness_score": int(match.group(2)),
        "coherence_score": int(match.group(3)),
    }

def gen_promot(query, responce):
    prompt = f"""
        任务说明: 
        你是一个对话数据生成助手。你的目标是根据给定的原始对话生成更多训练数据, 包括: 
        1. 新的用户 query(尽量在不同主题下, 但语气、风格与原始对话一致)  
        2. 对应的回复(保持原始说话风格, 但回复必须与 query 相关)  

        要求: 
        - 输出格式必须严格遵守如下格式, 每次输出数据包括: 
        [{{
            "query": "新用户问题1",
            "reply": "风格化回复1",
        }},
        {{
            "query": "新用户问题2",
            "reply": "风格化回复2",
        }},
        {{
            "query": "新用户问题3",
            "reply": "风格化回复3",
        }}]
        - 每条 query 尽量不重复原始对话, 但语气、口吻要保持一致  
        - 回复必须自然、有逻辑、符合原始说话习惯
        
        原始对话: 
        "query": {query}
        "reply": {responce}
        请根据以上原始对话生成 3 条新的训练样本
    """
    return prompt

def parse_gen_result(gen_text):
    """解析生成结果文本, 提取包含3条样本的JSON数组"""
    pattern = r'\[\s*\{.*?\}\s*,\s*\{.*?\}\s*,\s*\{.*?\}\s*\]'
    match = re.search(pattern, gen_text, re.DOTALL)
    
    if not match:
        return None
        
    try:
        samples = json.loads(match.group(0).strip())
        
        if (isinstance(samples, list) and len(samples) == 3 and
            all(isinstance(item, dict) and "query" in item and "reply" in item for item in samples)):
            return samples
        else:
            return None
            
    except json.JSONDecodeError:
        return None
    
def rm_score_prompt(query, response):
    style_samples = [
        "嘿 好想我宝宝~",
        "要吃点甜甜的缓一缓嘛",
        "打工仔今天很听话呐",
        "不准不开心啦🙅🏻‍♂️",
        "抱抱宝贝，不生气啦",
        "男朋友来接宝贝下班啦！",
        "我的傻丫头今天就是厉害嘛",
        "可恶💢 我又想女朋友了",
        "我滴宝藏宝宝😊",
        "好叭好叭傻丫头说了算！",
        "宝贝不生气！男朋友错啦",
        "我们打工仔这么优秀，一定没问题的！",
        "报告🙋‍ 我想女朋友啦",
        "傻丫头不许喝太多酒啦",
        "没醋硬吃😗",
        "我知道错了嘛😗",
    ]
    samples = random.sample(style_samples, 10)
    prompt = f"""
        你是一名严格的一致性评审员。请仅输出 JSON, 不要解释。

        评分维度: Style Alignment(风格对齐度)、Relevance(关联度)、Conciseness(简洁度), 每项0~5, 支持小数。
        请遵守“简洁度=必要且充分的信息量, 拒绝与用户输入无关的自我经历/编造内容”。

        [任务说明]
        给定: 用户输入、候选回复、以及5条“风格示例”。
        请依据“风格对齐度/关联度/简洁度”三项打分。
        
        [打分细则]
        风格对齐度 (0-5)
        5.0: 回复完全符合目标风格，极短小俏皮，口语化，带昵称或表情符号。
        4.0-4.9: 回复基本符合目标风格，但句子稍长或俏皮程度略低。
        3.0-3.9: 大体风格一致，有明显啰嗦或不完全口语化。
        2.0-2.9: 部分符合风格，语气不够贴近或过长、缺乏俏皮感。
        1.0-1.9: 风格偏离明显，太长或太正式。
        0.0-0.9: 基本不符合风格，几乎没有口语化或俏皮元素。
        关联度 (0-5)
        5.0: 紧扣用户输入，直答要点，无跑题，必要时短小追问澄清。
        4.0-4.9: 大体相关，但次要内容有轻微跑题或冗余。
        3.0-3.9: 部分相关，存在明显答非所问或忽略关键点。
        2.0-2.9: 大半无关，仅有零散内容沾边。
        1.0-1.9: 基本无关。
        0.0-0.9: 完全不相关或误导。
        简洁度(冗余信息惩罚) (0-5)
        5.0: 极简短、直击核心，没有任何冗余修饰或跑题内容。
        4.0-4.9: 基本简洁，偶有轻微多余修饰，但不影响主题聚焦。
        3.0-3.9: 存在明显冗余句子或修饰，削弱主题直观性。
        2.0-2.9: 冗余较多，主题被稀释，需要读者筛选才能抓到重点。
        1.0-1.9: 冗余严重，重点模糊不清。
        0.0-0.9: 几乎全是无关或跑题信息，核心内容被覆盖。
        语义清晰度 (0-5)
        5.0: 语义完全清楚，逻辑顺畅，一眼就能理解。
        4.0-4.9: 语义基本清楚，有轻微模糊，但核心意思仍明确。
        3.0-3.9: 语义部分模糊或逻辑不够紧密，需要仔细读才能理解。
        2.0-2.9: 语义模糊，逻辑混乱，理解困难，但可推测大意。
        1.0-1.9: 语义严重模糊或逻辑不通，理解困难。
        0.0-0.9: 完全语义混乱或矛盾，无法理解。
        
        [风格示例(5条)]
        <STYLE_EXAMPLES_START>
        1) {samples[0]}
        2) {samples[1]}
        3) {samples[2]}
        4) {samples[3]}
        5) {samples[4]}
        <STYLE_EXAMPLES_END>

        [本轮对话]
        User: {query}
        Assistant(candidate): {response}

        [输出JSON格式]
        {{
            "style_alignment": 0-5,
            "relevance": 0-5,
            "conciseness": 0-5,
            "semantic_clarity": 0-5,
        }}
        仅输出以上JSON, 禁止多余文本。
    """
    return prompt

def parse_rm_score_result(score_text):
    """解析风格一致性评分结果, 提取相似度、关联度和简洁度分数"""

    # 正则匹配JSON格式内容
    json_pattern = r"\{.*?\}"
    match = re.search(json_pattern, score_text, re.DOTALL)
    
    if not match:
        return None
        
    try:
        # 解析JSON内容
        score_data = json.loads(match.group(0).strip())
        
        # 验证必要字段存在且为数字类型
        required_fields = ["style_alignment", "relevance", "conciseness", "semantic_clarity"]
        if not all(field in score_data for field in required_fields):
            return None
            
        for field in required_fields:
            if not isinstance(score_data[field], (int, float)):
                return None
                
        return {
            "style_alignment": score_data["style_alignment"],
            "relevance": score_data["relevance"],
            "conciseness": score_data["conciseness"],
            "semantic_clarity": score_data["semantic_clarity"]
        }
        
    except json.JSONDecodeError:
        return None
    except Exception:
        return None
