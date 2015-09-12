import org.apache.spark.ml.{PipelineModel, PipelineStage, Pipeline}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Vector => Vec}

object Main extends App {
  val conf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("springleaf")

  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)

  try {
    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("csv/train_mini.csv")

    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3)).map(_.cache())

    MLUtils.saveAsLibSVMFile(
      toSVM(
        preprocess(trainingData),
        "targetDouble",
        "normNumFeatures"
      ),
      "svm/"
    )
  } finally {
    sc.stop()
  }

  def toSVM(df: DataFrame, targetColumn: String, featuresColumn: String): RDD[LabeledPoint] = {
    df.map { row =>
      LabeledPoint(
        row.getDouble(row.fieldIndex(targetColumn)),
        row(row.fieldIndex(featuresColumn)).asInstanceOf[Vec]
      )
    }
  }
  
  def preprocess(df: DataFrame): DataFrame = {
    import org.apache.spark.sql.functions._
    val toDouble = udf[Double, String]( _.toDouble)

    buildPipeline(df)
      .fit(df)
      .transform(df)
      .withColumn("targetDouble", toDouble(df("target")))
  }

  def buildPipeline(training: DataFrame): Pipeline = {
    val featureTypes = training
      .drop("target")
      .dtypes.toMap

    val numCols = featureTypes
      .filter { case (_, v) => v == "IntegerType" }
      .map(_._1)

    // FEATURE ENGINEERING

    // Numerical
    val numericalAssembler = new VectorAssembler().
      setInputCols(numCols.toArray).
      setOutputCol("numFeatures")

    val normalizer = new Normalizer().
      setInputCol("numFeatures").
      setOutputCol("normNumFeatures").
      setP(1.0)

    // Pipeline
    new Pipeline().setStages(
      List(
        numericalAssembler, normalizer
      ).toArray
    )
  }
}