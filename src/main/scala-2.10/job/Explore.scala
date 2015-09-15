package job

import org.apache.spark.ml.{MyOneHotEncoder, PipelineStage, Pipeline}
import org.apache.spark.ml.feature.{StringIndexer, Normalizer, VectorAssembler}
import org.apache.spark.mllib.linalg.{Vector => Vec}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.{StringType, StructField, StructType, DoubleType}
import org.apache.spark.sql.{DataFrame, Row}

object Explore extends SpringLeaf with Serializable {
  def run = try {
    val bad = Set("VAR_0214", "VAR_0207", "VAR_0213", "VAR_0840")
    val numWithNa = Set("VAR_0228", "VAR_0209", "VAR_0299", "VAR_0317", "VAR_0235", "VAR_0071", "VAR_0492", "VAR_0242", "VAR_0058", "VAR_0337", "VAR_0300", "VAR_0266", "VAR_0085", "VAR_0288", "VAR_0105", "VAR_0154", "VAR_0134", "VAR_0064", "VAR_0295", "VAR_0061", "VAR_0036", "VAR_0155", "VAR_0007", "VAR_0231", "VAR_0441", "VAR_0407", "VAR_0082", "VAR_0257", "VAR_0074", "VAR_0143", "VAR_0434", "VAR_0453", "VAR_0330", "VAR_0225", "VAR_0315", "VAR_0079", "VAR_0144", "VAR_0322", "VAR_0369", "VAR_0210", "VAR_0050", "VAR_0273", "VAR_0514", "VAR_0385", "VAR_0165", "VAR_0120", "VAR_0280", "VAR_0358", "VAR_0069", "VAR_0333", "VAR_0262", "VAR_0304", "VAR_0423", "VAR_0206", "VAR_0481", "VAR_0515", "VAR_0198", "VAR_0349", "VAR_0127", "VAR_0480", "VAR_0169", "VAR_0083", "VAR_0338", "VAR_0065", "VAR_0301", "VAR_0361", "VAR_0086", "VAR_0035", "VAR_0435", "VAR_0147", "VAR_0135", "VAR_0298", "VAR_0113", "VAR_0408", "VAR_0340", "VAR_0208", "VAR_0057", "VAR_0309", "VAR_0442", "VAR_0366", "VAR_0419", "VAR_0068", "VAR_0341", "VAR_0080", "VAR_0072", "VAR_0316", "VAR_0359", "VAR_0274", "VAR_0323", "VAR_0418", "VAR_0015", "VAR_0078", "VAR_0049", "VAR_0516", "VAR_0334", "VAR_0296", "VAR_0365", "VAR_0474", "VAR_0200", "VAR_0054", "VAR_0089", "VAR_0270", "VAR_0211", "VAR_0263", "VAR_0424", "VAR_0531", "VAR_0465", "VAR_0360", "VAR_0447", "VAR_0087", "VAR_0066", "VAR_0241", "VAR_0313", "VAR_0016", "VAR_0324", "VAR_0367", "VAR_0128", "VAR_0335", "VAR_0056", "VAR_0404", "VAR_0076", "VAR_0037", "VAR_0436", "VAR_0174", "VAR_0279", "VAR_0439", "VAR_0062", "VAR_0489", "VAR_0342", "VAR_0319", "VAR_0034", "VAR_0051", "VAR_0233", "VAR_0149", "VAR_0171", "VAR_0440", "VAR_0331", "VAR_0033", "VAR_0425", "VAR_0293", "VAR_0506", "VAR_0454", "VAR_0483", "VAR_0201", "VAR_0512", "VAR_0013", "VAR_0212", "VAR_0237", "VAR_0282", "VAR_0067", "VAR_0055", "VAR_0302", "VAR_0517", "VAR_0243", "VAR_0458", "VAR_0475", "VAR_0443", "VAR_0136", "VAR_0254", "VAR_0320", "VAR_0060", "VAR_0164", "VAR_0370", "VAR_0363", "VAR_0088", "VAR_0314", "VAR_0448", "VAR_0129", "VAR_0318", "VAR_0137", "VAR_0006", "VAR_0234", "VAR_0070", "VAR_0493", "VAR_0227", "VAR_0289", "VAR_0329", "VAR_0224", "VAR_0267", "VAR_0145", "VAR_0017", "VAR_0133", "VAR_0256", "VAR_0077", "VAR_0142", "VAR_0451", "VAR_0081", "VAR_0245", "VAR_0063", "VAR_0294", "VAR_0427", "VAR_0336", "VAR_0403", "VAR_0433", "VAR_0325", "VAR_0368", "VAR_0161", "VAR_0104", "VAR_0310", "VAR_0332", "VAR_0205", "VAR_0175", "VAR_0084", "VAR_0321", "VAR_0255", "VAR_0444", "VAR_0417", "VAR_0391", "VAR_0364", "VAR_0272", "VAR_0410", "VAR_0426", "VAR_0059", "VAR_0014", "VAR_0121", "VAR_0238", "VAR_0406", "VAR_0297", "VAR_0303")
    val categorical = Set("VAR_0460", "VAR_0445", "VAR_0381", "VAR_0413", "VAR_0344", "VAR_0151", "VAR_0429", "VAR_0339", "VAR_0047", "VAR_0328", "VAR_0187", "VAR_0253", "VAR_0351", "VAR_0250", "VAR_0362", "VAR_0386", "VAR_0277", "VAR_0112", "VAR_0123", "VAR_0464", "VAR_0456", "VAR_0525", "VAR_0373", "VAR_0180", "VAR_0140", "VAR_0306", "VAR_0096", "VAR_0194", "VAR_0162", "VAR_0173", "VAR_0102", "VAR_0389", "VAR_0502", "VAR_0379", "VAR_0284", "VAR_0093", "VAR_0184", "VAR_0053", "VAR_0420", "VAR_0467", "VAR_0470", "VAR_0311", "VAR_0348", "VAR_0326", "VAR_0148", "VAR_0392", "VAR_0402", "VAR_0291", "VAR_0504", "VAR_0431", "VAR_0131", "VAR_0181", "VAR_1934", "VAR_0510", "VAR_0501", "VAR_0393", "VAR_0521", "VAR_0491", "VAR_0276", "VAR_0138", "VAR_0124", "VAR_0287", "VAR_0265", "VAR_0372", "VAR_0508", "VAR_0473", "VAR_0519", "VAR_0412", "VAR_0046", "VAR_0345", "VAR_0484", "VAR_0495", "VAR_0232", "VAR_0097", "VAR_0388", "VAR_0457", "VAR_0350", "VAR_0103", "VAR_0001", "VAR_0219", "VAR_0116", "VAR_0524", "VAR_0488", "VAR_0292", "VAR_0091", "VAR_0236", "VAR_0281", "VAR_0499", "VAR_0130", "VAR_0247", "VAR_0170", "VAR_0354", "VAR_0226", "VAR_0195", "VAR_0485", "VAR_0468", "VAR_0258", "VAR_0401", "VAR_0285", "VAR_0312", "VAR_0496", "VAR_0305", "VAR_0094", "VAR_0505", "VAR_0376", "VAR_0327", "VAR_0382", "VAR_0513", "VAR_0477", "VAR_0141", "VAR_0139", "VAR_0244", "VAR_0472", "VAR_0260", "VAR_0045", "VAR_0117", "VAR_0048", "VAR_0400", "VAR_0252", "VAR_0271", "VAR_0098", "VAR_0125", "VAR_0107", "VAR_0509", "VAR_0387", "VAR_0152", "VAR_0415", "VAR_0478", "VAR_0005", "VAR_0356", "VAR_0146", "VAR_0185", "VAR_0428", "VAR_0100", "VAR_0500", "VAR_0114", "VAR_0520", "VAR_0462", "VAR_0269", "VAR_0230", "VAR_0163", "VAR_0182", "VAR_0494", "VAR_0308", "VAR_0220", "VAR_0409", "VAR_0346", "VAR_0259", "VAR_0109", "VAR_0377", "VAR_0432", "VAR_0486", "VAR_0090", "VAR_0248", "VAR_0390", "VAR_0160", "VAR_0416", "VAR_0421", "VAR_0469", "VAR_0405", "VAR_0355", "VAR_0383", "VAR_0497", "VAR_0371", "VAR_0523", "VAR_0099", "VAR_0251", "VAR_0101", "VAR_0449", "VAR_0307", "VAR_0422", "VAR_0414", "VAR_0479", "VAR_0118", "VAR_0466", "VAR_0490", "VAR_0126", "VAR_0186", "VAR_0374", "VAR_0437", "VAR_0461", "VAR_0283", "VAR_0183", "VAR_0343", "VAR_0503", "VAR_0278", "VAR_0111", "VAR_0122", "VAR_0463", "VAR_0153", "VAR_0352", "VAR_0119", "VAR_0052", "VAR_0378", "VAR_0353", "VAR_0150", "VAR_0268", "VAR_0110", "VAR_0511", "VAR_0384", "VAR_0092", "VAR_0471", "VAR_0455", "VAR_0450", "VAR_0132", "VAR_0108", "VAR_0115", "VAR_0095", "VAR_0249", "VAR_0452", "VAR_0498", "VAR_0172", "VAR_0290", "VAR_0347", "VAR_0375", "VAR_0264", "VAR_0430", "VAR_0357", "VAR_0476", "VAR_0487", "VAR_0507", "VAR_0261", "VAR_0380", "VAR_0275", "VAR_0522", "VAR_0518", "VAR_0286", "VAR_0459", "VAR_0482")
    val binary = Set("VAR_0020", "VAR_0197", "VAR_0025", "VAR_0018", "VAR_0397", "VAR_0221", "VAR_0246", "VAR_0214", "VAR_0042", "VAR_0028", "VAR_0526", "VAR_0203", "VAR_0396", "VAR_0188", "VAR_0031", "VAR_0019", "VAR_0008", "VAR_0399", "VAR_0446", "VAR_0106", "VAR_0024", "VAR_0021", "VAR_0222", "VAR_0191", "VAR_0039", "VAR_0011", "VAR_0032", "VAR_0012", "VAR_0038", "VAR_0027", "VAR_0215", "VAR_0527", "VAR_0239", "VAR_0043", "VAR_0229", "VAR_0394", "VAR_0026", "VAR_0009", "VAR_0192", "VAR_0398", "VAR_0022", "VAR_0223", "VAR_0023", "VAR_0040", "VAR_0010", "VAR_0528", "VAR_0044", "VAR_0411", "VAR_0530", "VAR_0030", "VAR_0041", "VAR_0193", "VAR_0202", "VAR_0395", "VAR_0196", "VAR_0189", "VAR_0190", "VAR_0529", "VAR_0438", "VAR_0216", "VAR_0029", "VAR_0199")

    // Numerical features
    val df = toDouble(drop(loadTrainData, bad), numWithNa + "target").na.fill(0.0)
    val intCols = colsByType(df.drop("target"), Set("IntegerType", "DoubleType"))
    val numAssembler = assemblerB(intCols.toList, "numFeatures")
    val normalizer = normalizerB("numFeatures", "normNumFeatures").setP(1.0)

    // Categorical features
    val encoders = categorical.foldLeft(List.empty[PipelineStage]) { _ ::: encode(df, _)._2 }
    val textAssembler = assemblerB(categorical.map("_" + _ + "_vec").toList, "textFeatures")

    // Pipeline
    val assembler = assemblerB(List("textFeatures", "normNumFeatures"), "features")
    val pipeline = new Pipeline().setStages((encoders ::: List(numAssembler, normalizer, textAssembler, assembler)).toArray)

    // Save
    saveSVM(pipeline.fit(df).transform(df), "build/train/", labelCol = "target")
  } finally {
    sc.stop()
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

  def saveSVM(df: DataFrame, path: String, labelCol: String = "label", featuresCol: String = "features"): Unit =
    MLUtils.saveAsLibSVMFile(
      df.map { row =>
      LabeledPoint(
        row.getDouble(row.fieldIndex("target")),
        row.getAs[Vec](row.fieldIndex("features"))
      )
    }, path)

  def drop(df: DataFrame, cols: Set[String]) =
    cols.foldLeft(df) { _.drop(_) }

  def colsByType(df: DataFrame, types: Set[String]) =
    df
      .dtypes
      .toMap
      .filter { case (_, v) => types(v) }
      .map(_._1)

  def normalizerB(input: String, output: String) =
    new Normalizer()
      .setInputCol(input)
      .setOutputCol(output)

  def assemblerB(input: List[String], output: String) =
    new VectorAssembler()
      .setInputCols(input.toArray)
      .setOutputCol(output)

  def toDouble(df: DataFrame, cols: Set[String]): DataFrame = {
    cols.foldLeft(df) { case(df, col) => df.withColumn(col, df(col).cast(DoubleType)) }
  }

}
