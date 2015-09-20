package job

import ml.{DataFrameSVM, PipelineDsl}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.{Vector => Vec}
import org.apache.spark.sql.DataFrame


object Predict extends SpringLeaf with PipelineDsl with DataFrameSVM {
  def run = {
    try {
      val df = loadSVM("build/test/")

      val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedTarget")
        .fit(df)

      val p = new Pipeline().setStages(Array(labelIndexer))

      val Array(trainingData, testData) = p.fit(df).transform(df).randomSplit(Array(0.8, 0.2)).map(_.cache())

      val randomForestClassifier = new RandomForestClassifier()
        .setFeaturesCol("features")
        .setLabelCol("indexedTarget")
        .setMaxDepth(10)
        .setMaxBins(10)

//      val size = testData.select("features").take(1)(0).getAs[Vec]("features").size
//      val c = new MultilayerPerceptronClassifier()
//        .setFeaturesCol("features")
//        .setLabelCol("indexedTarget")
//        .setMaxIter(500)
//        .setTol(1E-5)
//        .setLayers(Array(size, 100, 2))


      val binaryClassifier = new BinaryClassificationEvaluator()
        .setLabelCol("indexedTarget")
        .setRawPredictionCol("probability")
        .setMetricName("areaUnderROC")

      val params = new ParamGridBuilder()
        .addGrid(randomForestClassifier.maxDepth, Array(10, 15))
        .addGrid(randomForestClassifier.maxBins, Array(10, 15))
        .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
        .build()

      val cv = new CrossValidator()
        .setEstimator(randomForestClassifier)
        .setEvaluator(binaryClassifier)
        .setNumFolds(4)
        .setEstimatorParamMaps(params)

      println(binaryClassifier.evaluate(cv.fit(trainingData).transform(trainingData)))
      println(binaryClassifier.evaluate(cv.fit(trainingData).transform(testData)))
      println(cv.params)
    } finally {
      sc.stop()
    }
  }

  def buildPipeline(training: DataFrame, textCols: Set[String]) = {
    val labelIndexer = new StringIndexer()
      .setInputCol("target")
      .setOutputCol("indexedTarget")
      .fit(training)
  }
}
