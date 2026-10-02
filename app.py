"""app.py - Flask web app: enter a user ID, choose a method, see top-10 recommendations."""
from flask import Flask, render_template, request
from recommender import recommend, METHODS, n_users

app = Flask(__name__)


@app.route("/")                                       # the home page URL
def index():
    return render_template("index.html", methods=list(METHODS), n_users=n_users)


@app.route("/recommend")
def show_recommendations():
    uid = request.args.get("user_id", type=int)        # read ?user_id=7 from the URL
    method = request.args.get("method", "hybrid")
    if uid is None or uid < 1 or method not in METHODS:   # validate input before using it
        return render_template("index.html", methods=list(METHODS), n_users=n_users,
                               error="Please enter a valid user ID and method.")
    recs, used = recommend(uid, method)
    return render_template("recommendations.html", recs=recs.to_dict("records"), user_id=uid, method=used)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)                 # debug OFF (the original used debug=True: unsafe)
