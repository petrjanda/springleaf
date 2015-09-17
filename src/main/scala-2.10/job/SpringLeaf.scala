package job

trait SpringLeaf extends SparkJob with Serializable {
  def loadTrainData(path: String) = {
    sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(path)
//      .load("csv/train_mini.csv")
      .cache()
  }
}
