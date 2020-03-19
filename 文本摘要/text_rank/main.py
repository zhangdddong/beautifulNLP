#!/usr/bin/python3
# -*- coding: UTF-8 -*-
__author__ = 'zd'

import joblib
import data_utils
import model
import global_parameters as config

from flask import Flask, request, jsonify
app = Flask(__name__)


stop_words = data_utils.read_stopwords()
w2v_model = joblib.load(config.w2v_model_path)


@app.route('/get_summary', methods=['POST'])
def get_summary():
    content = request.form.get('content')   # Body x-www 中书写请求
    # content = request.json['content']     # Bady raw 中书写请求 同时选择json
    print(content)
    final_list = model.get_first_summaries(content, stop_words, w2v_model)
    summaries = model.get_last_summaries(content, final_list, stop_words, w2v_model)

    summary = ','.join(summaries)
    return jsonify({'summary': summary})


if __name__ == '__main__':

    content = "记得很小的时候，我到楼下去玩，一不小心让碎玻璃割伤了腿，疼得我“哇哇”大哭。爸爸问讯赶来，把我背到了医院，仔仔细细地为我清理伤口《爸爸是医生》、缝合、包扎，妈妈则在一旁流眼泪，一副胆战心惊的样子。我的腿慢慢好了，爸爸妈妈的脸上，才渐渐有了笑容。 一天下午，放学时，忽然下起了倾盆大雨。我站在学校门口，喃喃自语：“我该怎么办？”正在我发愁的时候，爸爸打着伞来了。“儿子，走，回家！”我高兴得喜出望外。这时，爸爸又说话了：“今天的雨太大了，地上到处是水坑，我背你回家！”话音未落，爸爸背起我就走了。一会儿，又听到爸爸说：“把伞往后挪一点，要不挡住我眼了。”我说：“好！”回到家，发现爸爸的衣服全湿透了，接连打了好几个喷嚏。我的眼泪涌出来了。 “可怜天下父母心”，这几年里，妈妈为我洗了多少衣服，爸爸多少次陪我学习玩耍，我已经记不清了。让我看在眼里、记在心里的是妈妈的皱纹、爸爸两鬓的白发。我的每一步成长，都包含了父母太多的辛勤汗水和无限爱心，“可怜天下父母心”！没有人怀疑，父母的爱是最伟大的、最无私的！"
    final_list = model.get_first_summaries(content, stop_words, w2v_model)
    summaries = model.get_last_summaries(content, final_list, stop_words, w2v_model)

    summary = ','.join(summaries)
    print(summary)

    # postman访问http://127.0.0.1:5000/get_summary，POST请求，并传入数据。
    # app.run(host='127.0.0.1', port=5000)
