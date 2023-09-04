import MeCab
import unidic

tagger = MeCab.Tagger()  # 「tagger = MeCab.Tagger('-d ' + unidic.DICDIR)」
with open("document.txt") as f:
    sample_txt = f.read()

sample_txt = """
10度の[ラ・リーガ]、7度の[コパ・デル・レイ]、4度の[UEFAチャンピオンズリーグ]（以下CLと表記）を含むバルセロナ歴代最多35回のタイトル獲得に貢献し、クラブ歴代通算最多得点（672ゴール）の記録を保持しており、[ラ・リーガ]（474ゴール）歴代最多ゴール記録、CL歴代最多ハットトリック記録（8）、ラ・リーガ歴代最多アシスト記録（192）を保持している。また、歴代最多7度の[バロンドール]を受賞し[注
2]、また前人未踏の4年連続バロンドール受賞も達成、6度のCL得点王と歴代最多6度の[ゴールデンシュー]（欧州得点王）を獲得し、多くのサッカー関係者や選手に史上最高のサッカー選手と称されている[3][4][5][6][7]

"""
result = tagger.parse(sample_txt)
print(result)
import pdb;pdb.set_trace()
open('document.json', 'w').write(documents[0].text)