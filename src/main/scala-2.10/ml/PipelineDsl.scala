package ml

import job.Preprocess._
import ml.encoders.{CategoricalEncoder, NumericalEncoder}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.mllib.linalg.{Vector => Vec}
import org.apache.spark.sql.types.{StringType, DoubleType}
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
}

trait DataFrameSVM {
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
}

trait DataFrameDSL extends DataFrameSVM {
  implicit class DataFramePimp(df: DataFrame) {
    def encodeNumerical() =
      NumericalEncoder.run(df)

    def encodeDates(cols: Set[String], transformer: java.util.Date => Int) = {
      import org.apache.spark.sql.functions._

      val format = new java.text.SimpleDateFormat("ddMMMyy:HH:mm:ss")
      val toDays = udf[Double, String] { s =>
        try {
          transformer(format.parse(s))
        } catch {
          case e:java.text.ParseException => 0.0
        }
      }

      def fixDates(df: DataFrame, others: Set[String]): DataFrame =
        others.foldLeft(df) { case (df, c) => df.withColumn(c, toDays(df(c))) }

      fixDates(df, cols)
    }

    def extractLabels(cols: Set[String]): Map[String, scala.collection.Map[String, Long]] = {
      cols.map { c =>
        (c, df.select(col(c).cast(StringType)).map(_.getString(0)).countByValue())
      }.toMap
    }


    def encodeCategorical(cols: Map[String, List[String]]) =
      CategoricalEncoder.run(df, cols)

    def correctDoubles(cols: Set[String]) =
      cols.foldLeft(df) { case(df, col) => df.withColumn(col, df(col).cast(DoubleType)) }

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


    def assemble(cols: Set[String], col: String) = {
      df.fitAndTransform(new Pipeline().setStages(List(assemblerB(cols.toList, col)).toArray))
    }

    def toSVM(path: String, labelCol: String) = {
      saveSVM(df, path, labelCol = labelCol)

      df
    }

    def fromSVM(path: String) = loadSVM(path)

    def inspect(lines: Int = 10): DataFrame = {
      locally { df =>
        df.printSchema()
        df.show(lines)
      }
    }

    def locally[T](todo: DataFrame => T): DataFrame = {
      todo(df)

      df
    }
  }
}
