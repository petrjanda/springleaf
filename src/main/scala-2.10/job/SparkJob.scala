package job

import org.apache.spark.mllib.linalg.{Vector => Vec}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

trait SparkJob extends Serializable {
  val conf = new SparkConf()
    .setMaster("spark://localhost:7077")
    .setAppName("springleaf")
    .set("spark.driver.memory", "14G")
    .set("spark.executor.memory", "10G")

  implicit val sc = new SparkContext(conf)
  implicit val sqlContext = new SQLContext(sc)

//  sc.addJar("/Users/petr/Research/spark/lib/jars/driver/spark-csv-assembly-1.2.0.jar")
//  sc.addJar("/Users/petr/Research/spark/spark-csv/target/scala-2.10/spark-csv-assembly-1.2.0.jar")
  sc.addJar("/Users/petr/.ivy2/cache/org.apache.commons/commons-csv/jars/commons-csv-1.1.jar")
  sc.addJar("/Users/petr/.ivy2/cache/com.databricks/spark-csv_2.10/jars/spark-csv_2.10-1.2.0.jar")
  sc.addJar("/Users/petr/Research/springleaf/spark/target/scala-2.10/spark_2.10-1.0.jar")
}
