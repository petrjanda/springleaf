package job

import org.apache.spark.mllib.linalg.{Vector => Vec}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

trait SparkJob {
  val conf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("springleaf")
    .setExecutorEnv("spark.executor.memory", "4G")

  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)
}
