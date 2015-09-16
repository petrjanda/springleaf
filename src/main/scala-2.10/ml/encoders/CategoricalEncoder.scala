package ml.encoders

import ml.PipelineDsl
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{MyOneHotEncoder, Pipeline, PipelineStage}
import org.apache.spark.sql.DataFrame

object CategoricalEncoder extends PipelineDsl {
  def run(df: DataFrame, features: Set[String]): DataFrame = {
    val encoders = features.foldLeft(List.empty[PipelineStage]) {
      _ ::: encode(df, _)._2
    }
    val textAssembler = assemblerB(features.map("_" + _ + "_vec").toList, "textFeatures")

    val pipeline = new Pipeline().setStages((encoders ::: List(textAssembler)).toArray)

    pipeline.fit(df).transform(df)
  }

  def encode(df: DataFrame, column: String): (String, List[PipelineStage], Array[String]) = {
    val indexer = new StringIndexer()
      .setInputCol(column)
      .setOutputCol("_" + column + "_indexed")

    val model = indexer.fit(df)

    val encoder = new MyOneHotEncoder()
      .setInputCol("_" + column + "_indexed")
      .setOutputCol("_" + column + "_vec")
      .setDropLast(false)

    (column, List(indexer, encoder), model.labels)
  }
}
