from flask import Flask, render_template, request
import Clustering

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/descripcion/")
def descripcion():
    return render_template("descripcion.html")


@app.route("/dataset/")
def dataset():
    return render_template("dataset.html")


@app.route("/conceptos/")
def conceptos():
    return render_template("conceptos.html")


@app.route("/modelo/", methods=["GET", "POST"])
def modelo():
    nclusters = int(request.form.get("nclusters", 4))
    nclusters = max(2, min(nclusters, 6))
    info = Clustering.RealizarClusteringPlayStore(nclusters=nclusters)
    return render_template("modelo.html", info=info, nclusters=nclusters)


@app.route("/interpretacion/")
def interpretacion():
    return render_template("interpretacion.html")


if __name__ == "__main__":
    app.run(debug=True)