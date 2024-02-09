spark-submit --executor-memory 4G --driver-memory 4G \
    task1.py ../resource/asnlib/publicdata/test_review.json python_task1.json \
    --conf spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j.properties