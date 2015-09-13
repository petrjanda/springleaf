package job

trait SpringLeaf extends SparkJob {
  def loadTrainData = {
    sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("../csv/train_sample.csv")
      .cache()
  }
}
