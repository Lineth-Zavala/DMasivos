import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
Logger.getLogger("org").setLevel(Level.ERROR)

//inicializacion de spark
val Spark= SparkSession.builder()getOrCreate()
val dataset = spark.read.option("header","true").option("inferSchema", "true")csv("Iris.csv")

dataset.show()

//structtype es un tipo de datos incorporado que es una colección de StructFields. extrae las columnas que se le declaren en los strucfield 
val structtype1 = 
StructType (
StructField("sepal_length",DoubleType,true)::
StructField("sepal_width",DoubleType,true)::
StructField("petal_length",DoubleType,true)::
StructField("petal_width",DoubleType,true)::
StructField("species",StringType,true):: Nil)

//funcion para limpiar datos e importacion
val dfstruct = spark.read.option("header","false").schema(structtype1)csv("/home/lineth-zavala/Desktop/Datos Masivos/Evaluacion/iris.csv")

dfstruct.show()
//obtener etiquetas de columna de tipo de flor
val label = new StringIndexer().setInputCol("species").setOutputCol("label")

//obtenie las caracteristicas y las asigna a una etiqueta
val assembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")

//declaracion de entrenamiento y prueba  6 de entrenamiento y 4 de prueba con 4 semillas
val splits = dfstruct.randomSplit(Array(0.6,0.4),seed=1234L)
val train = splits(0)
val test = splits(1)


//declaracion de neuronas entradas, ocultas y salidas
val layers = Array [Int](4,5,4,3)

val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
val pipeline = new Pipeline().setStages(Array(label,assembler,trainer))
val model = pipeline.fit(train)
val result = model.transform(test)
//Mostramos el resultado
val predictions = model.transform(test)
//predictions.show(5)
result.show()

//Selecciona los features y la prueba de error

val predictionAndLabels=result.select("prediction","label")
val evaluator=new MulticlassClassificationEvaluator().setMetricName("accuracy")
//Imprimimos los resultados de exactitud utilizando un evaluador multiclase
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


/*D).Explique detalladamente la funcion matematica de entrenamiento que utilizo con sus propias palabras
La función toma los valores y los pone en combinación lineal el cual al calcular entrena al modelo de prueba.



E).Explique la funcion de error que utilizo para el resultado final

*/