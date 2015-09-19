package ml.encoders

import ml.{DataFrameDSL, PipelineDsl}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame

object NumericalEncoder extends DataFrameDSL with PipelineDsl {
  def run(df: DataFrame): DataFrame = {
    val intCols = df
      .drop("target")
      .colsByType(Set("IntegerType", "DoubleType"))

    val numAssembler = assemblerB(intCols.toList, "numFeatures")
    val normalizer = normalizerB("numFeatures", "normNumFeatures").setP(2.0)

    val pipeline = new Pipeline().setStages((List(numAssembler, normalizer)).toArray)

    pipeline.fit(df).transform(df)
  }
}
