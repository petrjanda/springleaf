package ml.encoders

import ml.PipelineDsl
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerBase, StringIndexerModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, MyOneHotEncoder, Pipeline, PipelineStage}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StringType}

object CategoricalEncoder extends PipelineDsl {
  def run(df: DataFrame, features: Map[String, List[String]]): DataFrame = {
    val encoders = features.map { case(k, v) => encode(df, k, v.toArray) }.foldLeft(List.empty[PipelineStage]) { case (t, i) => t ::: i._2 }
    val textAssembler = assemblerB(features.map("_" + _._1 + "_vec").toList, "textFeatures")

    val pipeline = new Pipeline().setStages((encoders ::: List(textAssembler)).toArray)

    pipeline.fit(df).transform(df)
  }

  def encode(df: DataFrame, column: String, labels: Array[String]): (String, List[PipelineStage], Array[String]) = {
    val indexer = new StringIndexer()
      .setInputCol(column)
      .setOutputCol("_" + column + "_indexed")

    val s = System.currentTimeMillis()
    val model = indexer.fit(labels)
    println(s"$column --> ${System.currentTimeMillis() - s} millis")

    val encoder = new MyOneHotEncoder()
      .setInputCol("_" + column + "_indexed")
      .setOutputCol("_" + column + "_vec")
      .setDropLast(false)

    (column, List(model, encoder), model.labels)
  }
}