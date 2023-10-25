from fastapi import Body
from fastapi.responses import StreamingResponse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from configs import LLM_MODEL, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI, fschat_openai_api_address, get_model_worker_config
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List
from server.chat.utils import History
from server.utils import get_prompt_template


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               history: List[History] = Body([],
                                             description="历史对话",
                                             examples=[[
                                                 {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                 {"role": "assistant", "content": "虎头虎脑"}]]
                                             ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("llm_chat",
                                       description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    history = [History.from_data(h) for h in history]

    async def chat_iterator(query: str,
                            history: List[History] = [],
                            model_name: str = LLM_MODEL,
                            prompt_name: str = prompt_name,
                            ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            callbacks=[callback],
        )

        prompt_template = get_prompt_template(prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    return StreamingResponse(chat_iterator(query=query,
                                           history=history,
                                           model_name=model_name,
                                           prompt_name=prompt_name),
                             media_type="text/event-stream")


def chat_local(query: str,
                history: List[History] = [],
                model_name: str = LLM_MODEL,
                temperature: float = TEMPERATURE,
                chunk_size: int = 1000,
                chunk_overlap: int = 0
                ) -> str:
    history = [History.from_data(h) for h in history]
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    config = get_model_worker_config(model_name)
    model = ChatOpenAI(
        verbose=True,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        openai_proxy=config.get("openai_proxy")
    )
    prompt_template = get_prompt_template(model_name)
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])
    chain = LLMChain(prompt=chat_prompt, llm=model)
    return chain.run({"input": query})

def summaryWithLLM(doc: str,
            model_name: str = LLM_MODEL,
            temperature: float = TEMPERATURE,
            chunk_size: int = 2000,
            chunk_overlap: int = 0
            ) -> str:
    # 定义单个文档的总结提示词模板
    query = doc
    prompt_template = """总结下文内容:

    {text}

    总结内容:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_query = text_splitter.split_text(query)
    docs = [Document(page_content=t) for t in split_query]
    config = get_model_worker_config(model_name)
    model = ChatOpenAI(
        verbose=True,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        openai_proxy=config.get("openai_proxy")
    )
    # 定义refine合并总结内容的提示词模板
    refine_template = (
        "你的工作是负责生成一个最终的文本摘要\n"
        "这是现有的摘要信息: {existing_answer}\n"
        "根据新的背景信息完善现有的摘要"
        "背景信息如下\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "根据背景信息，完善现有的摘要"
        "如果背景信息没有用，则返回现有的摘要信息。"
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    prompt_template = """将下文内容简化概括为大概2000字的简介:

        {text}

        简介:"""

    COM_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    from langchain.chains.summarize import load_summarize_chain
    # chain = load_summarize_chain(model, chain_type="map_reduce", return_intermediate_steps=True, verbose=True,
    #                              map_prompt=PROMPT, combine_prompt=COM_PROMPT, output_key="output_text",)
    chain = load_summarize_chain(model, chain_type="refine", return_intermediate_steps=True, verbose=True,
                                 question_prompt=PROMPT, refine_prompt=refine_prompt, output_key="output_text", )
    # 执行摘要任务
    result = chain({"input_documents": docs}, return_only_outputs=True)
    # print(result["output_text"])
    print("\n".join(result["intermediate_steps"]))
    return "\n".join(result["intermediate_steps"])

if __name__ == "__main__":
    # summary_local(query="你好,介绍一下清华大学")
    summaryWithLLM(doc = """
    **习近平**（1953年6月15日—），男，陕西富平人，生于北京，祖籍河南邓州，中国共产党、中华人民共和国政治人物，正国级领导人，2012年11月至今任中共中央总书记及中共中央军委主席，2013年3月至今任中华人民共和国主席及国家中央军委主席，是中国党、政、军当前及第五代的最高领导人。

习近平在北京出生并长大，是中国开国元老习仲勋与其第二任夫人齐心的长子，也是首位出生在中华人民共和国成立后的中共最高领导人。习近平在北京接受了中小学教育，1969年，因文化大革命对家庭带来的冲击而被迫中止学业，作为知识青年前往陕西省延安市延川县梁家河村参加劳动与工作，在此期间于1974年1月10日加入中国共产党，并在后期担任了梁家河的村党支部书记。1975年进入清华大学化工系就读，1979年毕业后先后任国务院办公厅及中央军委办公厅秘书。1982年，离京赴河北省正定县先后任县委副书记、书记，开始在地方任职。1985年赴福建，先后在福建省厦门市、宁德地区、福州市任职，1999年任福建省人民政府省长，成为正部级官员。2002年起，先后任中共浙江省委书记和中共上海市委书记。2007年10月，当选为中共中央政治局常委和中共中央书记处书记，并先后兼任或当选中共中央党校校长、国家副主席、党和国家中央军委副主席等职务。

2012年11月，在中共十八届一中全会当选为中共中央总书记和中共中央军委主席（2017、2022年获得连任），2013年3月当选为中国国家主席和国家中央军委主席（2018、2023年全票获得连任），自从行修宪)打破任期制以来，是唯一连任三届国家主席的领导人，成为党和国家最高领导人至今。2016年10月，在中共十八届六中全会上确立了其领导核心的地位，兼任中国共产党第十五届中央候补委员，第十六届至第二十届中央委员，第十七届至第二十届中央政治局委员、常委，第十八至二十届中央委员会总书记。第九届至第十四届全国人大代表。

习近平第一任妻子是柯玲玲。现任妻子为中国女高音歌唱家彭丽媛，育有一女习明泽。

## 早年从政经历

### 初入军政两界

1979年4月毕业之后，分配到国务院办公厅和中共中央军委办公厅担任其父好友当时的中共中央政治局委员兼国防部部长耿飚的三个秘书之一，而在当秘书期间（1979-1982年），也是现役军人。

在邓小平发起的军队大裁员情况下，其转向地方。1979年习近平与中国前驻英国大使柯华)的女儿柯玲玲（又名柯小明，生于1951年，时年28岁）结婚，1982年离婚。在父亲习仲勋的帮助下，选择了离北京相对较近的正定县。1982年3月25日，习近平上任，；上任后习近平身着下放时穿的绿色军大衣，到各地走访调查。他简朴的风格使当地官员王幼辉感到吃惊。由于正定地处南北交通动脉，境内道路路况糟糕，习近平果断采取行动动员4.32万人参加养护公路课程。在1983年严打)中，正定县也举行公审公判活动。期间，习近平被指在正定县内实行了严厉的计划生育政策，40万人中有3.1万名妇女做了绝育手术，3万人安装了宫内绝育工具。1983年底，年仅30周岁的习近平升任并成为正定县最年轻的县委书记。而《中国时报》的报道则指，位于正定县东面且同属石家庄地区的无极县的县委书记栗战书在这一时期与习近平结识，造就了未来双方的合作。 

《老人报》的一篇文章指，他在任期间大力推动大包干生产责任制，并向上级反映正定县情况，使得正定县粮食征购任务减少了2000万斤；此外走遍整个乡镇，进行考察，他当初的同事均以“平易近人，温和儒雅，和蔼可亲”形容习近平，习近平任内的另一项政绩是大力发展旅游业，以《红楼梦》为脚本模拟建造“荣国府”，也因《三国演义》中的蜀汉名将赵云出身正定县，而兴建了常山公园等旅游景点，每年因此为正定县带来一千万人民币以上的旅游收入。1985年4月28日到5月9日，身为河北省玉米加工考察团团长，并以河北省石家庄地区食品协会主席的身份访问美国艾奥瓦州马斯卡廷县。

### 福建生涯

1988年6月26日，调任（到任）中共福建省宁德地区地委书记、宁德军分区党委第一书记，期间惩治官员腐败四百余人，并重点查办福鼎县林增团、宁德地区侨联副主席郑锡煊等腐败要案。1989年六四事件期间，习近平下令阻止试图进入宁德的温州学生发动串连。

1990年6月21日，升任中共福州市委书记、市人大常委会主任，并兼任闽江职业大学（今闽江学院）校长。在其任内兴建长乐国际机场、建造福州至厦门公路、开发马尾新港区、引进外资企业600多家，其中台资企业占五成。同时还谋划了福州3820工程，把福州定位为国际化大都市，并认为南边有广州，北边有上海，按照这个距离，两者之间应该出现一个大城市，这个大城市应该是福州。

1993年，任中共福建省委常委兼福州市委书记。1996年5月9日，习近平晋升中共福建省委副书记，不再担任中共福州市委书记、闽江职业大学校长；在1997年9月召开的中共十五大上，首次当选为中央候补委员；1999年8月9日，出任福建省人民政府副省长、代理省长；2000年1月27日任福建省省长，期间协助中央处理朱镕基总理督办的厦门“远华案”。

### 主政浙沪

2002年10月12日，习近平转任浙江省人民政府副省长、代省长。在任期间，于清华大学人文社会学院马克思主义理论与思想政治教育专业在职研究生班，以《农村市场化研究》为学位论文题目，获法学博士学位。在2002年11月召开的中共十六大上，首次当选为中央委员；11月21日，随着浙江省委原书记张德江升任中央政治局委员并调任广东，习近平升任中共浙江省委书记。2003年1月22日，兼任浙江省人大常委会主任，他在任期间延续了以前在各地的下级调研习惯；并提出“八八战略”、“五大百亿工程”，重视民生发展和产业结构调整、吸引外资并增强民间资金流动；2006年，浙江省城镇人均可支配收入超过一万八千元人民币，农民人均收入超过七千元人民币，位列中国各省区第一名。

2007年3月24日，习近平被调任中共上海市委书记。同年6月20日，上海市徐汇区第十四届人大常委会第四次会议补选习近平为第十二届上海市人大代表。浙江省人大代表资格因职务调离浙江省行政区区域而取消。2008年1月29日，习近平在上海市第十三届人大第一次会议上当选第十一届全国人大代表（上海代表团）。

### 晋升政治局常委

参见：中国共产党集体领导制度

2008年，时任中共中央政治局常委、国家副主席习近平与美国总统小布什会晤。

2007年10月22日，中共十七届一中全会召开，习近平与李克强等一起当上中央政治局常委（排名第六），外界视两人为胡锦涛和温家宝的接班人。同年12月22日，习近平又从原政治局常委曾庆红手中，接过了中共中央党校校长的职务。

2008年3月15日，在第十一届全国人大一次会议上，习近平成为中华人民共和国副主席。任职期间，习近平分管党建、组工、港澳、北京奥运筹办等工作。

2009年2月11日，以国家副主席身份出访墨西哥，在当地会见华侨时，他说：“有些吃饱了没事干的外国人，对我们的事情指手划脚。中国一不输出革命，二不输出饥饿和贫困，三不去折腾你们，还有什么好说的。”在一些群体中引发争议。

2010年10月18日，中共十七届五中全会决定增补习近平为中共中央军委副主席。2010年10月28日，全国人大常委会表决，决定习近平为国家军委副主席。

## 中共中央总书记

### 第一任期

参见：习李体制


2012年11月15日，在中国共产党第十八次全国代表大会闭幕后随即召开的中共十八届一中全会上，习近平当选新一届中央委员会总书记和中央军事委员会主席，开始第一任期。胡锦涛一次交出总书记和军委主席两个最高权力位置“裸退”，这在中华人民共和国历史上是第一次。习近平称赞“胡锦涛带头离开领导岗位是高风亮节的表现”，标志着废除干部领导职务终身制进一步完善。2013年3月14日，在第十二届全国人大第一次会议上，习近平当选中华人民共和国主席和国家军委主席，成为代表国家象征的国家元首。中共十八届六中全会后中国共产党官方对习近平的称呼普遍是“以习近平同志为核心的党中央”，象征习近平正式被中国共产党确立为中央领导集体的核心


在财经方面，推出中国（上海）自由贸易试验区，并在其后将该措施推广到全国多个省市。亦推动举行中国国际进口博览会。同时，推行深化国税地税征管体制改革，整合组建新的税务系统，结束了1994年分税制改革所形成的国家、地方税务机构分离的格局。同时，2015年发生的股灾也显示证券市场及金融监管体系尚需进一步完善。

在区域发展方面，提出京津冀一体化、长江经济带及粤港澳大湾区等区域经济举措。

在司法方面，废止劳动教养制度，进行法官检察官员额制改革，新增了人民陪审员、人民调解员体系，亦设置杭州互联网法院、上海金融法院等专门法院。

在民生方面，修改了计划生育法，推行全面二孩政策。并将国家卫生和计划生育委员会撤销，并入新成立的“国家卫生健康委员会”。

在军事方面，推行深化国防和军队改革，进行军委多部门制改革，成立陆军领导机构、火箭军、战略支援部队以及联勤保障部队，撤销了公安现役部队，七大军区改为五大战区，并进行陆军合成化改革及要求军队和武警部队全面停止有偿服务，成为1949年建国以来规模最大的军事改革。

在教育方面，推出世界一流大学和一流学科建设（双一流）取代了原有的211工程及985工程，成为大学建设的评价指标。

在环境方面，提出2030年单位国内生产总值二氧化碳排放比2005年下降60%至65%，森林蓄积量比2005年增加45亿立方米。他多次强调“绿水青山就是金山银山”、“人与自然要和谐共生”。提高环保考评在官员晋升中的权重。并对环保考察不合格的官员实行一票否决。其任内大幅改善了雾霾问题。京津冀煤改气政策使得华北地区高污染散煤使用大幅减少。但改造过程中，出现天然气供应不足，新取暖设备改造的费用及如何验收等短期问题，招致忽视农村取暖问题的批评。

在监察方面，反腐行动迅速扩大，各地众多腐败窝案被集中查处，一大批包括现任及前任的正副国家级领导人在内的各级涉腐干部被查处、问责及判刑。其中，中共中央政治局委员孙政才因严重违纪被中纪委接受调查，并最终被开除党籍与公职，成为任内首位接受调查的在任中央政治局委员。同时实施中央巡视组、派驻纪检监察组，将监察职能从行政机关剥离等举措。开展国家监察体制改革，设立国家监察委员会和地方各级监察委员会。

在国际和区域方面，提出了“一带一路”的合作倡议与“人类命运共同体”的概念，并提出要开展“中国特色大国外交”。并推动建立了亚洲基础设施投资银行等国际性金融机构，意在增强中国的世界影响力。2016年9月，中国在杭州举办了G20峰会。任期内发生的香港、台湾、南海等问题也对其执政提出挑战与考验。

### 第二任期

参见：以习近平同志为核心的党中央

2017年10月召开的中国共产党第十九次全国代表大会上，习近平当选第十九届中央委员，并全票通过将以其命名的“习近平新时代中国特色社会主义思想”写进《中国共产党章程》，成为党的指导思想。在2017年10月25日召开的中共十九届一中全会上，以全票连任中央委员会总书记和中央军事委员会主席的职务，开始第二任期。2018年3月17日，习近平在十三届全国人大一次会议上全票连任中华人民共和国主席。

在立法方面，通过宪法修正案)，内容包括赋予设区的市地方立法权、设立国家监察机关、取消国家主席和国家副主席任期限制及从宪法层面上确立宪法宣誓制度等事项，共21条。其中的第45条删除国家主席、副主席“连续任职不得超过两届”的限制，引发国内国际关注。通过《中华人民共和国监察法》，将行政监察部门与检察院反贪机关整合设立了国家监察委员会。通过《中华人民共和国民法典》，自此形成中国境内完整统一的民法典，但离婚冷静期等问题也引发诸多争议。

在党务方面，中共中央政治局全体委员和常委被规定每年向总书记述职一次。2018年3月，习近平首次审阅各中央政治局委员提交的报告，并对各政治局委员提出重要要求。同时将之前的领导小组改设为委员会，如中央财经领导小组改为中央财经委员会，中央网络和信息安全领导小组改设为中央网信委员会，将领导小组常态化。习近平也多次重提毛泽东时代的“党领导一切”，而“党政军民学，东西南北中，党是领导一切的”这句毛时代的口号在中共十九大上被写入党章。2021年11月11日，十九届六中全会正式提出了“两个确立”。2022年10月，“两个维护”写入修改后的党章。

在财经方面，提出建设海南自由贸易港，在上海证券交易所推行科创板及上市注册制。此外，成立国家医疗保障局以推进医保体制改革。推行企业职工基本养老保险全国统筹，以及中央企业国有股权的10%划归全国社会保障基金理事会，以解决社会保险基金缺口问题。2020年疫情以来，当局推行减税降费政策，以舒缓经济疲态。同时大力推行以工代赈，扩大政府投资，开工建设项目，以刺激经济。指出中国不能施行福利主义，超出能力的“福利主义”是不可持续的；防止资本野蛮生长和无序扩张；绝不允许再大范围大面积“拉闸限电”。

在教育方面，推行双减政策减轻家长负担。又采取高职专项扩招，加大国企事业单位对应届生招聘力度以求减轻就业压力。近几年来每年中国新增就业都在1000万以上。2022年2月，第二轮双一流名单发布，增加了部分学校，并取消了一流大学。同时大力发展职业教育，

在民生方面，继续脱贫攻坚战。2020年11月，全国所有贫困县全部宣布摘帽。5年间每年脱贫1000万以上。2021年2月25日，习近平宣布“在解决困扰中华民族几千年的绝对贫困问题上取得了伟大历史性成就，创造了人类减贫史上的奇迹”。2020年10月，中共十九届五中全会召开，会议宣布要在全面建成小康社会的同时，开启社会主义现代化国家建设新征程。2021年7月1日，在庆祝中国共产党成立100周年大会上，习近平宣告在中华大地上全面建成了小康社会。同时猪肉周期也对民生造成影响。非洲猪瘟疫情的爆发对国内猪肉供应造成冲击，猪肉价格大幅提升，冲击民众正常消费。同时再次修改了计划生育政策，实施三孩生育政策。

在军事方面，发生2020年中印边境冲突，之后相继发生多起边境对峙事件。扩充中国核武库，中国首艘国产航母山东舰交付。

在监察方面，继续进行大规模的反腐败行动，中国的清廉指数全球排名由2018年的87名上升至2022年的66名。

在防疫方面，新冠肺炎疫情在2020年1月武汉市爆发。当届政府迅速采取武汉封城、春节假期延长等一系列措施。2月春节后，在除武汉及湖北省部分其它区域外，推动复工复产，用约三个月时间基本控制国内疫情传播，在世界各国对疫情的控制上处于前列。全球虽处于新冠疫情大流行，但4月8日武汉解封后，中国本土传播长期处于极低水平。2020年国庆中秋长假出行人次高达6.37亿人次，为疫情前的79%，2021年五一长假出行人次高达2.3亿人次，甚至高于疫情前的水平。在2021年国庆期间上映的电影《长津湖》更是刷新了中国电影票房纪录，观影人次超过一亿。2022年2月以闭环管理的方式成功举办了北京冬奥会。2022年3月以来，中国多地出现疫情，习近平多次发出指示坚持动态清零政策，要求尽快遏制新冠疫情扩散蔓延势头，并强调防控方针是由党的性质和宗旨决定的。习近平认为宁可稍微影响一点点经济发展，也要保障人民群众生命安全和身体健康。

在国际和外交方面，美国特朗普政府于2018年3月发动中美贸易战，对中美关系造成冲击。俄乌战争全面爆发后，习近平没有制裁俄罗斯，并明确表示反对在乌克兰使用核武器，强调核战争打不赢也打不得。2022年4月，习近平发表了《全球安全倡议》。该倡议涉及中国的外交事务原则。2022年9月14日至16日，习近平出席在撒马尔罕举行的上海合作组织成员国元首理事会第二十二次会议，并对哈萨克斯坦、乌兹别克斯坦两国进行国事访问，这是COVID-19疫情以来中国最高领导人首次出国访问。

### 第三任期

参见：两个维护和两个确立

2022年10月召开的中国共产党第二十次全国代表大会上，习近平首次提出要以中国式现代化推进中华民族伟大复兴。之后，69岁的习近平当选第二十届中央委员，并在中共二十届一中全会上再次连任中共中央总书记和中共中央军委主席，开始第三任期。改革开放确立“两届任期”惯例以后注 4\，他是中共历史上首位打破传统的主要领导人。与习近平关系密切的李强)、蔡奇、丁薛祥等人都进入中央政治局常委会。2023年3月10日，习近平在十四届全国人大一次会议连任国家主席和国家军委主席。

在防疫方面，2022年11月10日，习近平在中共中央政治局常委会会议上，继续强调坚定不移贯彻“动态清零”总方针，研究部署进一步优化防控工作的二十条措施。但在11月底，全国多地爆发反对动态清零政策运动。

在财经方面，巴西、阿根廷、泰国等国相继宣布或讨论使用人民币结算国际贸易，人民币国际化加速推进。

在港澳台方面，2023年因蔡英文过境美国，开展环台岛战备警巡和“联合利剑”演习，山东号亦首次参加。

在国际和外交方面，在疫情结束后，习近平继续加紧推动“中国特色大国外交”；接连通过多个重要的国际外交场合上与多个国家的领袖进行会晤，包括在亚太经济合作组织（APEC）的领袖峰会上，发表重要谈话；斡旋沙特与伊朗复交，同时多国政要接连访问中国，削弱了美国遏制中国的影响，并且有意促成俄乌战争停火；举办第一届中国—中亚峰会，进一步扩大中国在中亚的影响力；通过加深和沙特阿拉伯的合作，和巴勒斯坦建立战略伙伴关系等行动，扩大中国在中东的影响力；举办第三届一带一路国际合作高峰论坛，发表主旨演讲。

## 兴趣爱好

习近平曾经在福建任职多年，爱好闽南语，听得懂闽南语但不会讲，并在2015年听到朱立伦与国民党智库执行长尹启铭讲的闽南语笑话而捧腹大笑，习近平认为闽南语流行歌曲《爱拼才会赢》象征着厦门的拼搏精神。同时也会听得懂基本的福州话并且会说一些简单的福州话，但发音并不准确。

在体育运动方面，习近平喜爱足球，其少年求学时期即喜好足球运动，曾是学校足球队成员，并常亲自前往观看球赛。担任国家领导职务后，曾多次在公开场合表达对中国足球运动发展的期许，甚至实地展现其足球功底，也曾主导中国足坛反腐风暴。2008年7月考察北京奥运会秦皇岛赛区时，曾看望中国国家女子足球队并获赠球衣。2009年10月作为中华人民共和国副主席访问德国期间，曾表示“中国有一流的球迷和世界可观的足球市场，但目前水平还比较低，希望可以迎头赶上。”2012年2月访问爱尔兰都柏林时，曾参观爱尔兰盖尔运动协会总部，并公开展示球技。

同时，他也喜爱游泳和围棋，并将此作为日常休闲活动。2013年6月7日至8日，习近平与时任美国总统的贝拉克·奥巴马在加州安纳伯格庄园举行私人会谈，习近平称每天游泳1千米。习近平任国务院副总理耿飚秘书时，他俩有着共同的爱好，都喜欢下围棋。耿飚让身边所有的工作人员都要学下围棋，他认为这能够训练他们的大局观。习近平还专门找好友聂卫平学下棋。其对爬山、排球等运动亦比较喜爱，也曾在年轻时练习过拳击。


    
    """)
