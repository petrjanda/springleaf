package job

trait SpringLeaf extends SparkJob with Serializable {
  def loadTrainData = {
    sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("../csv/train.csv")
//      .load("csv/train_mini.csv")
      .cache()
  }
}
