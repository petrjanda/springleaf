package job

import org.apache.spark.mllib.linalg.{Vector => Vec}

object ExploreText extends SpringLeaf {
  def run = {
    try {
      val df = loadTrainData("../csv/train.csv")

      val textCols = df.dtypes
        .filter { case (_, v) => v == "StringType" }
        .filterNot { case(k, _) => Set(
        "VAR_0073", "VAR_0075", "VAR_0156", "VAR_0157", "VAR_0158",
        "VAR_0159", "VAR_0166", "VAR_0167", "VAR_0168", "VAR_0176",
        "VAR_0177", "VAR_0178", "VAR_0179", "VAR_0204", "VAR_0217")(k)
      }
        .map(_._1)

      println(s"textcols --> ${textCols.toList}")

      val counts = textCols.map { c =>
        val distinct = df
          .select(c)
          .distinct()
          .limit(100)
          .collect
          .size
  //        .map(_.getString(0))

        println(s"--> $c ~ ${distinct}")

        (c, distinct)
      }.toList

      def isUnary(d: (String, Int)) = d._2 == 1
      def isBinary(d: (String, Int)) = d._2 == 2
      def isCategorical(d: (String, Int), limit: Int = 10) = d._2 > 2 && d._2 < limit

      val unary = counts.filter(isUnary(_)).map(_._1).toSet
      val binary = counts.filter(isBinary(_)).map(_._1).toSet
      val categorical = counts.filter(isCategorical(_)).map(_._1).toSet
      val other = counts.map(_._1).toSet -- unary -- binary -- categorical

      println(s"unary --> $unary")
      println(s"binary --> $binary")
      println(s"categorical --> $categorical")
      println(s"other --> ${counts.map(_._1).toSet -- unary -- binary -- categorical}")
    } finally {
      sc.stop()
    }
  }
}
