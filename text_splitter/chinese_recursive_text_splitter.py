import re
from typing import List, Optional, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]


if __name__ == "__main__":
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=300,
        chunk_overlap=90
    )
    ls = [
        "1939年底1940年初，毛泽东先后发表了《〈共产党人〉发刊词》《中国革命和中国共产党》《新民主主义论》等重要著作，在中国共产党内首次创造性地提出并阐述了新民主主义理论。此后，他在《论联合政府》《论人民民主专政》等著作中又作了进一步的阐述和发挥，使其更加系统和完整。新民主主义理论系统地分析了近代中国社会的性质和中国革命的对象、性质、动力、前途等，阐述了中国共产党关于新民主主义革命与新民主主义社会的理论，回答了“中国向何处去”的问题，开创了中国社会发展的新纪元。 理论>简介 1939年底1940年初，毛泽东先后发表了《〈共产党人〉发刊词》《中国革命和中国共产党》《新民主主义论》等重要著作，在中国共产党内首次创造性地提出并阐述了新民主主义理论。此后，他在《论联合政府》《论人民民主专政》等著作中又作了进一步>的阐述和发挥，使其更加系统和完整。新民主主义理论系统地分析了近代中国社会的性质和中国革命的对象、性质、动力、前途等，阐述了中国共产党关于新民主主义革命与新民主主义社会的理论，回答了“中国向何处去”的问题，开创了中国社会发展的新纪元。 面对“中国向何处去”的问题的一个回答 “新民主主义论”系统地回答了当时中国民主革命和未来建设新中国的一系列根本问题。这是毛泽东思想成熟的标志。毛泽东此时之所以将中国共产党人自己的理论成果命名为新民主主义理论，还有一个特殊的历史背景。 >全国抗战爆发以后，中国共产党从原来遭受反动派严密封锁的狭小天地里走出来，变成全国性的大党，公开走上全国政治生活的大舞台，受到人们越来越密切的关注。全国各阶级、各阶层都渴望了解中国共产党对时局和中国未来前途的看法。 为了建立和巩固抗>日民族统一战线，中国共产党作了重大政策调整。全国抗战爆发后，国共两党实现了第二次合作。中国共产党对于抗日民族统一战线是诚心维护的，但自抗战进入相持阶段后，随着中共政治影响的扩大和八路军、新四军的壮大，国民党内的顽固派在军事上制造反共摩擦。就这样，面对“中国向何处去”的问题，中国共产党必须对此系统地表明自己的立场和观点，在全国人民面前旗帜鲜明地提出区别于其他政党的政治主张来。正是在这样的背景下，毛泽东写作并发表了上述文章，系统论述了新民主主义理论。《新民主主义论》发表后产生了深远的影响，这从当时各类人群对《新民主主义论》的评述中可以窥见一斑。1940年1月9日，毛泽东在陕甘宁边区文化协会第一次代表大会上发表题为《新民主主义的政治与新民主主义的文化》的长篇演讲。据当时台下听讲者回忆：“这个长篇>讲话，从下午一直讲到入夜点起煤气灯的时分”，在会场的五六百名听众“被他的精辟见解和生动话语所鼓舞、所吸引，聚精会神，屏息静听，情绪热烈，不时响起一阵阵的掌声”。《新民主主义论》一篇既出，中共党内同志都被其精辟的内容吸引与折服。当时在>晋察冀边区的邓拓读到《新民主主义论》后，写下了一首激情昂扬的诗——《读毛主席〈新民主主义论〉》：“万水千山只等闲，长城绕指到眉端。阵图开处无强敌，翰墨拈来尽巨观。风雨关河方板荡，运筹帷幄忘屯艰。苍龙可缚缨在手，且上群峰绝顶看！” “新民主主义论”的独创性特征 新民主主义革命的目标是建立新民主主义社会。毛泽东的这些著作描绘了新民主主义社会的蓝图。 第一，关于新民主主义社会的性质。毛泽东指出中国革命分“两步走”：第一步是通过新民主主义革命，完成反帝反封建的任务，建立新民>主主义社会；第二步是在此基础上，继续进行社会主义革命，使中国逐步过渡到社会主义社会。新民主主义社会是新民主主义革命与社会主义革命之间的过渡形态，是中国由半殖民地半封建社会进入社会主义社会的必经阶段。它不同于社会主义社会，因为其根本任务不是反对资本主义，而是反对帝国主义、封建主义；它也不同于资本主义社会，因为其领导力量不是资产阶级，而是无产阶级。因此可以说，它既具有社会主义社会的某些因素，也保存有资本主义的某些因素，但其前途必然是社会主义，而不是资本主义。 >第二，关于新民主主义社会的基本纲领。在政治上，要建立“无产阶级领导下的一切反帝反封建的人们联合专政的民主共和国，这就是新民主主义的共和国”。在经济上，要使一切“大银行、大工业、大商业归这个共和国的国家所有”；“这个共和国并不没收其他资>本主义的私有财产，并不禁止‘不能操纵国民生计’的资本主义生产的发展”；“这个共和国将采取某种必要的方法，没收地主的土地，分配给无地和少地的农民”。在文化上，要挣脱帝国主义、封建主义文化思想的奴役，实行人民大众的反帝反封建的文化，即“民族的科学的大众的文化”。 毛泽东之前的马克思主义社会发展学说中虽然没有新民主主义社会一说，但是，由于中国国情的特殊性，社会发展阶段的情形也应具特殊性，新民主主义社会无论从其命名看还是从其内容看，都显示了其独创性特征。　“新民主主义论”是以毛泽东为代表的中国共产党人把马克思主义基本原理同中国革命具体实践相结合的产物。这一理论的提出，是对科学社会主义的重大发展，对新民主主义革命的胜利和由新民主主义向社会主义过渡产生了深远影响。 解决了两个革命阶段的衔接即如何“两步走”>的问题 中国共产党虽然自党的二大已经认识到现阶段的中国革命是要完成反帝反封建的民主革命的任务，社会主义革命是下一阶段的任务，但并没有说明达到这个目标的具体途径。长期以来，犯右倾错误的人不了解无产阶级领导的民主革命同资产阶级领导的民>主革命的区别，因而也就不了解民主革命同社会主义革命之间的联系，如“二次革命论”；而犯“左”倾错误的人则混淆民主革命同社会主义革命的界限，在民主革命阶段内不适当地提出某些社会主义的任务，如“不断革命论”。全国抗战爆发前后，党内有人说，抗日战争的胜利将引导中国到非资本主义的前途。但是“非资本主义前途”是一个不很明确的概念。人们不清楚，抗战的胜利是直接达到社会主义呢，还是要经过其他阶段？毛泽东明确地回答了这个十分重要的中国发展的前途问题。他指出，抗日战争的胜利应使中国摆脱殖民地、半殖民地、半封建的地位，但中国既不可能成为资本主义国家，也还不可能立即进入社会主义社会。新民主主义革命和社会主义革命是两个不同的革命阶段，不能“毕其功于一役”，但两个革命阶段必须也必然是衔接的，不容横插一个资产阶级专政。他提出并论证了建立新民主主义社会的必要性和可能性，指明新民主主义社会是走向社会主义前途的过渡阶段。这正好解决了两个革命阶段的衔接即如何“两步走”的问题，为党纠正和防止右的和“左”的错误，制定和执行正确的政策，奠定了理论基础。 解决了落后>的旧中国跨越“卡夫丁峡谷”的难题 马克思、恩格斯曾描绘人类社会发展的一般规律，认为社会主义应是建立在生产力高度发达的基础上。不过，他们晚年也针对东方落后国家提出过跨越资本主义的“卡夫丁峡谷”、直接向社会主义过渡的设想，列宁还开始了在俄>国的实践。但如何实现这一跨越，他们并没有作出明确的理论说明。毛泽东从中国具体实际出发，分析了中国政治、经济、文化状况和阶级状况，创造性地提出了“新民主主义论”，解决了这一难题。他提出，新民主主义社会是一个既不同于资本主义也不同于社会主义的特殊的社会形态。政治上，它实行工人、农民、小资产阶级和民族资产阶级的联合专政，但领导权是属于无产阶级的。经济上，在没收官僚资本、废除封建土地关系以后，实行国营经济为主导，国有经济、合作经济、个体经济、私人资本主义和国家资本主义五种经济成分并存的经济结构，允许资本主义发展，目的是发展生产力为进入社会主义创造物质条件。文化上，新民主主义文化是民族的、科学的、大众的文化，既有民族的形式，更有共产主义的内容，是将来走向社会主义的意识形态上的保证。这样一种社会形态，显然是人类历史上未有过的，也是马克思主义经典作家不曾描述过的，它是毛泽东把马克思主义中国化的理论创新。 为社会主义初级阶段理论提供了理论来源和实践经验 毛泽东的“新民主主义论”虽然极具理论价值，在新中国成立初期也曾实践了一段时间，但由于各种复杂原因，原来预计至少要搞15年至20年的新民主主义社会提前结束，在生产力落后状况尚未得到根本改变的情况下就开始向社会主义过渡，而且步骤过急过快，由此遗留下许多问题。党的十一届三中全会以来，以邓小平同志为核心的第二代中央领导集体在总结历史经验教训的基础上，系统地提出了社会主义初级阶段理论，指出我国虽然已经是社会主义社会，但是还处于并且将长期处于社会主义的初级阶段，因此这个阶段的主要任务仍是发展生产力。显而易见，这一理论与“新民主主义论”存在诸多相似之处，特别是体现在经济纲领上。在“新民主主义论”中，新民主主义社会的经济形态包括五种经济成分，实行的是在国营经济的领导下，多种经济成分并存、共同发展的基本经济制度。在社会主义初级阶段理论中，根据我国尚处在社会主义初级阶段的国情，也把坚持和完善社会主义公有制为主体、多种所有制经济共同发展作为我国的一项基本经济制度。由此可见，“新民主主义论”中的经济纲领为社会主义初级阶段理论提供了模式参考和实践经验，后者是对前者的继承和发展。 毛泽东新民主主义理论的形成，标志着马克>思主义中国化实现了第一次历史性飞跃。相比较而言，“新民主主义革命论”中最具创新性的是开辟了一条农村包围城市、武装夺取政权的中国特色革命道路。而从总体上看，“新民主主义论”或许更具独创性，有许多观点都是马克思主义经典作家未曾有过的。"]

    # for inum, text in enumerate(ls):
    #     print(inum)
    #     # chunks = text_splitter.split_text(text)
    #     doc = Document(page_content=text)
    #     chunks = text_splitter.split_documents(doc)
    #     count = 0
    #     for chunk in chunks:
    #         print(chunk)
    #         count += 1
    #     print(count)
