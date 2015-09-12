import org.apache.spark.ml.attribute._
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.ml.{MyOneHotEncoder, PipelineModel, PipelineStage, Pipeline}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{Column, DataFrame, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Vector => Vec}

import scala.collection.immutable.Range
import scala.util.Try

object Main extends App {
  val conf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("springleaf")
    .setExecutorEnv("spark.executor.memory", "4G")

  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)

  try {
    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("../csv/train_sample.csv")
      .cache()

    val textCols = df.dtypes
      .filter { case (_, v) => v == "StringType" }
      .filterNot { case(k, _) => Set(
      "VAR_0073", "VAR_0075", "VAR_0156", "VAR_0157", "VAR_0158",
      "VAR_0159", "VAR_0166", "VAR_0167", "VAR_0168", "VAR_0176",
      "VAR_0177", "VAR_0178", "VAR_0179", "VAR_0204", "VAR_0217")(k)
  }
      .map(_._1)


//    val counts = textCols.take(10).map { c =>
//      val distinct = df
//        .select(c)
//        .distinct()
//        .limit(100)
//        .collect
//        .size
////        .map(_.getString(0))
//
//      println(s"--> $c ~ ${distinct}")
//
//      (c, distinct)
//    }.toList
//
//    def isUnary(d: (String, Int)) = d._2 == 1
//    def isBinary(d: (String, Int)) = d._2 == 2
//    def isCategorical(d: (String, Int), limit: Int = 10) = d._2 > 2 && d._2 < limit
//
//    val unary = counts.filter(isUnary(_)).map(_._1).toSet
//    val binary = counts.filter(isBinary(_)).map(_._1).toSet
//    val categorical = counts.filter(isCategorical(_)).map(_._1).toSet
//    val other = counts.map(_._1).toSet -- unary -- binary -- categorical
//
//    println(s"unary --> $unary")
//    println(s"binary --> $binary")
//    println(s"categorical --> $categorical")
//    println(s"other --> ${counts.map(_._1).toSet -- unary -- binary -- categorical}")

    val binary  = Set("VAR_0008", "VAR_0011", "VAR_0012", "VAR_0009", "VAR_0010")
    val categorical  = Set("VAR_0001", "VAR_0005")
    val other  = Set("VAR_0007", "VAR_0013", "VAR_0006")

    import org.apache.spark.sql.functions._

    val toDouble = udf[Double, String] { s =>
      try { s.toDouble } catch {
        case e:NumberFormatException => if(s == "NA") 0.0 else throw e
      }
    }

    def fixSchema(df: DataFrame, others: Set[String]): DataFrame =
      others.foldLeft(df) { case (df, c) => df.withColumn(c, toDouble(df(c))) }

    val data = preprocess(fixSchema(df, others = other), categorical)
      .select("target", "features")

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3)).map(_.cache())

    val c = new GBTClassifier()
      .setFeaturesCol("features")
      .setLabelCol("target")
      .setMaxIter(20)

    val m = new MulticlassClassificationEvaluator()
      .setLabelCol("target")
      .setPredictionCol("prediction")

    println(m.evaluate(c.fit(trainingData).transform(testData)))




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
  
  def preprocess(df: DataFrame, textCols: Set[String]): DataFrame = {
    import org.apache.spark.sql.functions._
    val toDouble = udf[Double, String]( _.toDouble)

    buildPipeline(df, textCols)
      .fit(df)
      .transform(df)
      .withColumn("target", toDouble(df("target")))
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
        textCols.foldLeft(List.empty[PipelineStage]) { case(t, k) => encode(k)._2 ::: t } ::: List(textAssembler, numericalAssembler, normalizer, featuresAssembler)
      ).toArray
    )
  }
}

package org.apache.spark.ml {

  class MyOneHotEncoder extends OneHotEncoder {
    override def transformSchema(schema: StructType): StructType = {
      val inputColName = $(inputCol)
      val outputColName = $(outputCol)

      SchemaUtils.checkColumnType(schema, inputColName, DoubleType)
      val inputFields = schema.fields
      require(!inputFields.exists(_.name == outputColName),
        s"Output column $outputColName already exists.")

      val inputAttr = Attribute.fromStructField(schema(inputColName))
      val outputAttrNames: Option[Array[String]] = inputAttr match {
        case nominal: NominalAttribute =>
          if (nominal.values.isDefined) {
            nominal.values
          } else if (nominal.numValues.isDefined) {
            nominal.numValues.map(n => Array.tabulate(n)(_.toString))
          } else {
            None
          }
        case binary: BinaryAttribute =>
          if (binary.values.isDefined) {
            binary.values
          } else {
            Some(Array.tabulate(2)(_.toString))
          }
        case _: NumericAttribute =>
          throw new RuntimeException(
            s"The input column $inputColName cannot be numeric.")
        case _ =>
          None // optimistic about unknown attributes
      }

      val filteredOutputAttrNames = outputAttrNames.map { names =>
        if ($(dropLast)) {
          require(names.length > 1,
            s"The input column $inputColName should have at least two distinct values.")
          names.dropRight(1)
        } else {
          names
        }
      }

      val outputAttrGroup = if (filteredOutputAttrNames.isDefined) {
        val attrs: Array[Attribute] = filteredOutputAttrNames.get.map { name =>
          println(s"--> $name")
          val a = new BinaryAttribute(Some("default"))

          if(!name.isEmpty) a.withName(name) else a
        }
        new AttributeGroup($(outputCol), attrs)
      } else {
        new AttributeGroup($(outputCol))
      }

      val outputFields = inputFields :+ outputAttrGroup.toStructField()
      StructType(outputFields)
    }
  }

}