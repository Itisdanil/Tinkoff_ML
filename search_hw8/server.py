from flask import Flask, render_template, request
from search import remove_stop_words, retrieve, build_index
from time import time

app = Flask(__name__, template_folder='.')
df = build_index()


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = retrieve(query, df)
    results = [[df['title'][doc[1]], df['content'][doc[1]][:200]] + [f'{round(doc[0], 2)}'] for doc in documents]
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Yandex',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
