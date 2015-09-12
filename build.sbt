name := "spark"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies := Seq(
  "org.apache.spark" %% "spark-core" % "1.5.0", // % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.5.0", // % "provided",
  "org.apache.spark" %% "spark-sql" % "1.5.0", // % "provided",
  "org.apache.spark" %% "spark-graphx" % "1.5.0", // % "provided",
  "com.databricks" %% "spark-csv" % "1.2.0", // % "provided",
  "org.scalatest" % "scalatest_2.10" % "2.2.1" % "test"
)