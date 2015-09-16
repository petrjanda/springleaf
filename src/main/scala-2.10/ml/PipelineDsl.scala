package ml

import job.Preprocess._
import ml.encoders.{CategoricalEncoder, NumericalEncoder}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.mllib.linalg.{Vector => Vec}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.Pipeline

trait PipelineDsl {
  def assemblerB(input: List[String], output: String): VectorAssembler =
    new VectorAssembler()
      .setInputCols(input.toArray)
      .setOutputCol(output)

  def normalizerB(input: String, output: String): Normalizer =
    new Normalizer()
      .setInputCol(input)
      .setOutputCol(output)

  // TODO: Refactor to DataFrame DSL
  def saveSVM(df: DataFrame, path: String, labelCol: String = "label", featuresCol: String = "features"): Unit =
    MLUtils.saveAsLibSVMFile(
      df.map { row =>
        LabeledPoint(
          row.getDouble(row.fieldIndex(labelCol)),
          row.getAs[Vec](row.fieldIndex(featuresCol))
        )
      }, path)

  def loadSVM(path: String, labelCol: String = "label", featuresCol: String = "features")(implicit sc: SparkContext, sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._

    MLUtils.loadLibSVMFile(sc, path).toDF(labelCol, featuresCol)
  }

  def toDouble(df: DataFrame, cols: Set[String]): DataFrame = {
    cols.foldLeft(df) { case(df, col) => df.withColumn(col, df(col).cast(DoubleType)) }
  }

  implicit class DataFramePimp(df: DataFrame) {
    def encodeNumerical() =
      NumericalEncoder.run(df)

    def encodeDates(cols: Set[String]) = {
      import org.apache.spark.sql.functions._

      val format = new java.text.SimpleDateFormat("ddMMMyy:HH:mm:ss")
      val toDays = udf[Double, String] { s =>
        try {
          format.parse(s).getTime / (1000 * 60 * 60 * 24)
        } catch {
          case e:java.text.ParseException => 0.0
        }
      }

      def fixDates(df: DataFrame, others: Set[String]): DataFrame =
        others.foldLeft(df) { case (df, c) => df.withColumn(c, toDays(df(c))) }

      fixDates(df, cols)
    }

    def encodeCategorical(cols: Set[String]) =
      CategoricalEncoder.run(df, cols)

    def correctDoubles(cols: Set[String]) =
      toDouble(df, cols)

    def drop(cols: Set[String]) =
      cols.foldLeft(df) { _.drop(_) }

    def colsByType(types: Set[String]) =
      df
        .dtypes
        .toMap
        .filter { case (_, v) => types(v) }
        .map(_._1)

    def fitAndTransform(pipeline: Pipeline): DataFrame =
      pipeline.fit(df).transform(df)
  }
}
