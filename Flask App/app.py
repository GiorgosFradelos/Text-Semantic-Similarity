
from flask import Flask, request, redirect, render_template, url_for

import numpy as np

import models






#simple_model = BERT_model(L=8, H=128, A=2, trainable=False)




labels = ["contradiction", "entailment", "neutral"]
def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([str(sentence1), str(sentence2)])
    test_data = models.BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = simple_model.predict(test_data)[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return pred, proba


app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    #main_page
    return render_template('index.html')

@app.route('/load_models', methods=["POST"])
def load_models():
    global simple_model, siamese_model
    simple_model = models.BERT_model(L=12, H=512, A=8, trainable=False)
    siamese_model = models.siamese_bert_model(L=8, H=128, A=2, trainable=False)
    return render_template('index.html', status='models loaded')


@app.route('/simple_BERT', methods=['GET', 'POST'])
def simple_bert():
    return render_template('simple_BERT.html')

@app.route('/simple_bert_inference', methods=['GET','POST'])
def predict_simple_bert():
    sent1 = request.form['sent1']
    sent2 = request.form['sent2']

    sentence_pairs = np.array([[str(sent1), str(sent2)]])
    test_data = models.BertSemanticDataGenerator(
        sentence_pairs,
        labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

  #  pred, proba = check_similarity(sent1, sent2)

    proba = simple_model.predict(test_data)[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]


    return render_template('simple_BERT.html', pred=pred, prob=proba)



''' Siamese BERT '''
@app.route('/Siamese_BERT', methods=['GET', 'POST'])
def siamese_bert():
    return render_template('siamese_BERT.html')

@app.route('/siamese_bert_similarity', methods=['GET','POST'])
def siamese_bert_similarity():

    sent1 = str(request.form['sent1'])
    sent2 = str(request.form['sent2'])

    sent1 = np.array([sent1])
    sent2 = np.array([sent2])


    test_data = models.SiameseBertSemanticDataGenerator(
        sent1,
        sent2,
        labels=None,
        batch_size=1,
        shuffle=False,
        include_targets=False,
    )

    preds = siamese_model.predict(test_data)[0]
    preds = '%.3f'% float(preds)

    return render_template('siamese_BERT.html', pred=preds)


if __name__ == '__main__':
    app.run(port=8080, debug=True)
