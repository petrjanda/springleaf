import job._
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.mllib.linalg.{Vector => Vec}
import org.apache.spark.sql.types.{DoubleType, StructType}

object Main extends App {
  val app :: tail = args.toList

  app match {
    case "text" => ExploreText.run
    case "predict" => Predict.run
    case "preprocess" => Preprocess.run(tail.head, tail.tail.head)
  }
}

package org.apache.spark.ml.feature {

import org.apache.spark.SparkException
import org.apache.spark.ml.{Model, Estimator}
  import org.apache.spark.ml.param.ParamMap
  import org.apache.spark.ml.util.Identifiable
  import org.apache.spark.sql.DataFrame
  import org.apache.spark.sql.types.StringType
  import org.apache.spark.sql.functions._
import org.apache.spark.util.collection.OpenHashMap

class StringIndexer(override val uid: String) extends Estimator[StringIndexerModel]
  with StringIndexerBase {

    def this() = this(Identifiable.randomUID("strIdx"))

    /** @group setParam */
    def setInputCol(value: String): this.type = set(inputCol, value)

    /** @group setParam */
    def setOutputCol(value: String): this.type = set(outputCol, value)

    // TODO: handle unseen labels

    override def fit(dataset: DataFrame): StringIndexerModel = {
      val counts = dataset.select(col($(inputCol)).cast(StringType))
        .map(_.getString(0))
        .countByValue()
      val labels = counts.toSeq.sortBy(-_._2).map(_._1).toArray
      copyValues(new StringIndexerModel(uid, labels).setParent(this))
    }

    def fit(labels: Array[String]): MyStringIndexerModel = {
      copyValues(new MyStringIndexerModel(uid, labels.toArray))
    }

    override def transformSchema(schema: StructType): StructType = {
      validateAndTransformSchema(schema)
    }

    override def copy(extra: ParamMap): StringIndexer = defaultCopy(extra)
  }

class MyStringIndexerModel (
                           override val uid: String,
                           val labels: Array[String]) extends Model[StringIndexerModel] with StringIndexerBase {

  def this(labels: Array[String]) = this(Identifiable.randomUID("strIdx"), labels)

  private val labelToIndex: OpenHashMap[String, Double] = {
    val n = labels.length
    val map = new OpenHashMap[String, Double](n)
    var i = 0
    while (i < n) {
      map.update(labels(i), i)
      i += 1
    }
    map
  }

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    if (!dataset.schema.fieldNames.contains($(inputCol))) {
      logInfo(s"Input column ${$(inputCol)} does not exist during transformation. " +
        "Skip StringIndexerModel.")
      return dataset
    }

    val indexer = udf { label: String =>
      if (labelToIndex.contains(label)) {
        labelToIndex(label)
      } else {
        // TODO: handle unseen labels
        throw new SparkException(s"Unseen label: $label.")
      }
    }
    val outputColName = $(outputCol)
    val metadata = NominalAttribute.defaultAttr
      .withName(outputColName).withValues(labels).toMetadata()

    dataset.withColumn($(outputCol), indexer(dataset($(inputCol))))
//    dataset.select(col("*"),
//      indexer(dataset($(inputCol)).cast(StringType)).as(outputColName, metadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(inputCol))) {
      validateAndTransformSchema(schema)
    } else {
      // If the input column does not exist during transformation, we skip StringIndexerModel.
      schema
    }
  }

  override def copy(extra: ParamMap): StringIndexerModel = {
    val copied = new StringIndexerModel(uid, labels)
    copyValues(copied, extra).setParent(parent)
  }
}

}



package org.apache.spark.ml {

  class MyOneHotEncoder extends OneHotEncoder {
    override def transformSchema(schema: StructType): StructType = {
      val inputColName = $(inputCol)
      val outputColName = $(outputCol)

      SchemaUtils.checkColumnType(schema, inputColName, DoubleType)
      val inputFields = schema.fields
      require(!inputFields.exists(_.name == outputColName),
        s"Output column $outputColName already exists.")

      val inputAttr = Attribute.fromStructField(schema(inputColName))
      val outputAttrNames: Option[Array[String]] = inputAttr match {
        case nominal: NominalAttribute =>
          if (nominal.values.isDefined) {
            nominal.values
          } else if (nominal.numValues.isDefined) {
            nominal.numValues.map(n => Array.tabulate(n)(_.toString))
          } else {
            None
          }
        case binary: BinaryAttribute =>
          if (binary.values.isDefined) {
            binary.values
          } else {
            Some(Array.tabulate(2)(_.toString))
          }
        case _: NumericAttribute =>
          throw new RuntimeException(
            s"The input column $inputColName cannot be numeric.")
        case _ =>
          None // optimistic about unknown attributes
      }

      val filteredOutputAttrNames = outputAttrNames.map { names =>
        if ($(dropLast)) {
          require(names.length > 1,
            s"The input column $inputColName should have at least two distinct values.")
          names.dropRight(1)
        } else {
          names
        }
      }

      val outputAttrGroup = if (filteredOutputAttrNames.isDefined) {
        val attrs: Array[Attribute] = filteredOutputAttrNames.get.map { name =>
          println(s"--> $name")
          val a = new BinaryAttribute(Some("default"))

          if(!name.isEmpty) a.withName(name) else a
        }
        new AttributeGroup($(outputCol), attrs)
      } else {
        new AttributeGroup($(outputCol))
      }

      val outputFields = inputFields :+ outputAttrGroup.toStructField()
      StructType(outputFields)
    }
  }

}