import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object task1 {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println(
        "Usage: spark-submit --executor-memory 4G --driver-memory 4G task1.py <review_file_path> <output_file_path>"
      )
      sys.exit(1)
    }

    val reviewFilePath = args(0)
    val outputFilePath = args(1)

    val spark = SparkSession
      .builder()
      .appName("Yelp Data Exploration (Task 1)")
      .getOrCreate()

    try {
      val reviewRDD = spark.sparkContext
        .textFile(reviewFilePath)
        .map(row => row.split("\\|")) // Assuming the data is pipe-separated

      val nReview = reviewRDD.count()

      val nReview2018 = reviewRDD
        .filter(row => {
          val date = row(3)
          val year = date.split(" ")(0).split("-")(0).toInt
          year == 2018
        })
        .count()

      val nUser = reviewRDD.map(row => row(1)).distinct().count()

      val top10User = reviewRDD
        .map(row => (row(1), 1))
        .reduceByKey(_ + _)
        .sortBy(_._2, ascending = false)
        .take(10)

      val nBusiness = reviewRDD.map(row => row(2)).distinct().count()

      val top10Business = reviewRDD
        .map(row => (row(2), 1))
        .reduceByKey(_ + _)
        .sortBy(_._2, ascending = false)
        .take(10)

      val result = Map(
        "n_review" -> nReview,
        "n_review_2018" -> nReview2018,
        "n_user" -> nUser,
        "top10_user" -> top10User,
        "n_business" -> nBusiness,
        "top10_business" -> top10Business
      )

      import spark.implicits._
      val resultDF = result.toList.toDF("Metric", "Value")

      resultDF.write
        .format("json")
        .mode("overwrite")
        .save(outputFilePath)
    } finally {
      spark.stop()
    }
  }
}
