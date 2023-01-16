from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # parse input data
    data = request.get_json()
    input_data = data['input']

    # create spark session
    conf = SparkConf().setAppName("Linear Regression")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # create dataframe
    df = spark.createDataFrame(input_data)

    # create vector assembler
    assembler = VectorAssembler(
        inputCols=["col1", "col2", "col3"], outputCol="features")

    # create linear regression model
    lr = LinearRegression(featuresCol="features", labelCol="label")

    # create pipeline
    pipeline = Pipeline(stages=[assembler, lr])

    # fit model to data
    model = pipeline.fit(df)

    # predict on input data
    predictions = model.transform(df)

    # select only the prediction column
    output = predictions.select("prediction").collect()

    # return output as json
    return jsonify({'output': output})


if __name__ == '__main__':
    app.run(port=5000)
``
