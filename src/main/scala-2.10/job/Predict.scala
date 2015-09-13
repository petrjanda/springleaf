package job

import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, Normalizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{MyOneHotEncoder, Pipeline, PipelineStage}
import org.apache.spark.mllib.linalg.{Vector => Vec}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame


object Predict extends SpringLeaf {
  def run = {
    try {
      val df = loadTrainData

      val binary = Set("VAR_0008", "VAR_0011", "VAR_0012", "VAR_0009", "VAR_0010")
      val categorical = Set("VAR_0001", "VAR_0005")
      val other = Set("VAR_0007", "VAR_0013", "VAR_0006")

      import org.apache.spark.sql.functions._

      val toDouble = udf[Double, String] { s =>
        try {
          s.toDouble
        } catch {
          case e: NumberFormatException => if (s == "NA") 0.0 else throw e
        }
      }

      def fixSchema(df: DataFrame, others: Set[String]): DataFrame =
        others.foldLeft(df) { case (df, c) => df.withColumn(c, toDouble(df(c))) }

      val data = preprocess(fixSchema(df, others = other), categorical)
        .select("indexedTarget", "features")

      val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3)).map(_.cache())

      testData.take(5).foreach(println)

      val c = new GBTClassifier()
        .setFeaturesCol("features")
        .setLabelCol("indexedTarget")
        .setMaxIter(20)

      val m = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedTarget")
        .setPredictionCol("prediction")

      println(m.evaluate(c.fit(trainingData).transform(testData)))


    } finally {
      sc.stop()
    }
  }


  def toSVM(df: DataFrame, targetColumn: String, featuresColumn: String): RDD[LabeledPoint] = {
    df.map { row =>
      LabeledPoint(
        row.getDouble(row.fieldIndex(targetColumn)),
        row(row.fieldIndex(featuresColumn)).asInstanceOf[Vec]
      )
    }
  }

  def preprocess(df: DataFrame, textCols: Set[String]): DataFrame = {
    import org.apache.spark.sql.functions._
    val toDouble = udf[Double, String]( _.toDouble)

    buildPipeline(df, textCols)
      .fit(df)
      .transform(df)
      .withColumn("doubleTarget", toDouble(df("target")))
  }

  def buildPipeline(training: DataFrame, textCols: Set[String]): Pipeline = {
    val featureTypes = training
      .drop("target")
      //      .drop("VAR_0009")
      //      .drop("VAR_0010")
      //      .drop("VAR_0042")
      //      .drop("VAR_0043")
      //      .drop("VAR_0044")
      //      .drop("VAR_0196")
      //      .drop("VAR_0202")
      //      .drop("VAR_0205") // TEMPORARY!!!!
      //      .drop("VAR_0230") // TEMPORARY!!!!
      //      .drop("VAR_0214")
      //      .drop("VAR_0216")
      //      .drop("VAR_0207")
      //      .drop("VAR_0213")
      //      .drop("VAR_0229")
      //      .drop("VAR_0239")
      //      .drop("VAR_0840")
      .dtypes.toMap

    val intCols = featureTypes
      .filter { case (_, v) => Set("IntegerType", "DoubleType")(v) }
      .map(_._1)

    // FEATURE ENGINEERING

    val labelIndexer = new StringIndexer()
      .setInputCol("target")
      .setOutputCol("indexedTarget")
      .fit(training)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Numerical
    val numericalAssembler = new VectorAssembler().
      setInputCols(intCols.toArray).
      setOutputCol("numFeatures")

    val normalizer = new Normalizer().
      setInputCol("numFeatures").
      setOutputCol("normNumFeatures").
      setP(1.0)

    def encode(column: String): (String, List[PipelineStage], Array[String]) = {
      val indexer = new StringIndexer()
        .setInputCol(column)
        .setOutputCol("_" + column + "_indexed")


      val model = indexer.fit(training)

      val encoder = new MyOneHotEncoder()
        .setInputCol("_" + column + "_indexed")
        .setOutputCol("_" + column + "_vec")
        .setDropLast(false)

      (column, List(indexer, encoder), model.labels)
    }

    val textAssembler = new VectorAssembler()
      .setInputCols(textCols.map("_" + _ + "_vec").toArray)
      .setOutputCol("textFeatures")

    val pipeline = new Pipeline().setStages(
      (textCols.foldLeft(List.empty[PipelineStage]) { case(t, k) => encode(k)._2 ::: t } ::: List(textAssembler)).toArray
    )

    val featuresAssembler = new VectorAssembler()
      .setInputCols(Array("textFeatures", "normNumFeatures"))
      .setOutputCol("features")

    // Pipeline
    new Pipeline().setStages(
      (
        textCols.foldLeft(List.empty[PipelineStage]) { case(t, k) => encode(k)._2 ::: t } ::: List(textAssembler, numericalAssembler, normalizer, featuresAssembler, labelIndexer)
      ).toArray
    )
  }
}
