import os
from flask import Flask, jsonify, request
from predict import NER

os.makedirs('temp/', exist_ok=True)
app = Flask('Full-text Extractor for Administrative Documents')

ner = NER()


@app.route('/ner/', methods=['POST'])
def fultext_extractor():
    try:
        text = request.json['text']
        entities = ner.predict(text)

        result = []
        for w, l in entities:
            result.append({"entity": w,
                           "label": l})

        return jsonify(result=result), 200
    except Exception as ex:
        print(str(ex))
        return jsonify(Message="Something went wrong"), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6969, debug=True)
