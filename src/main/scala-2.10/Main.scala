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
    case "preprocess" => Preprocess.run(tail.head)
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