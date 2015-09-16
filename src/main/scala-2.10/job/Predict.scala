package job

import ml.PipelineDsl
import org.apache.spark.ml.classification.{RandomForestClassifier, MultilayerPerceptronClassifier, GBTClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{IndexToString, Normalizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{MyOneHotEncoder, Pipeline, PipelineStage}
import org.apache.spark.mllib.linalg.{Vector => Vec}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

import scala.util.control.NonFatal


object Predict extends SpringLeaf with PipelineDsl {
  def run = {
    try {
      val df = loadSVM("build/train/")

//      import org.apache.spark.sql.functions._
//
//      val format = new java.text.SimpleDateFormat("ddMMMyy:HH:mm:ss")
//      val toDays = udf[Double, String] { s =>
//        try {
//          format.parse(s).getTime / (1000 * 60 * 60 * 24)
//        } catch {
//          case e:java.text.ParseException => 0.0
//        }
//      }
//
//      val dates = Set(
//        "VAR_0073", "VAR_0075", "VAR_0156", "VAR_0157", "VAR_0158",
//        "VAR_0159", "VAR_0166", "VAR_0167", "VAR_0168", "VAR_0176",
//        "VAR_0177", "VAR_0178", "VAR_0179", "VAR_0204", "VAR_0217"
//      )
//
//      def fixDates(df: DataFrame, others: Set[String]): DataFrame =
//        others.foldLeft(df) { case (df, c) => df.withColumn(c, toDays(df(c))) }
//
      val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedTarget")
        .fit(df)

      val p = new Pipeline().setStages(Array(labelIndexer))

      val Array(trainingData, testData) = p.fit(df).transform(df).randomSplit(Array(0.8, 0.2)).map(_.cache())

      val c = new RandomForestClassifier()
        .setFeaturesCol("features")
        .setLabelCol("indexedTarget")
        .setMaxDepth(5)

//      val size = testData.select("features").take(1)(0).getAs[Vec]("features").size
//      val c = new MultilayerPerceptronClassifier()
//        .setFeaturesCol("features")
//        .setLabelCol("indexedTarget")
//        .setMaxIter(500)
//        .setTol(1E-5)
//        .setLayers(Array(size, 100, 2))


      val m = new BinaryClassificationEvaluator()
        .setLabelCol("indexedTarget")
        .setRawPredictionCol("probability")
        .setMetricName("areaUnderROC")

      println(m.evaluate(c.fit(trainingData).transform(trainingData)))
      println(m.evaluate(c.fit(trainingData).transform(testData)))


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
