package com.sample.tweets_classifier

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.text.nlclassifier.NLClassifier
import java.math.RoundingMode
import java.text.DecimalFormat


class MainActivity : AppCompatActivity() {

    private lateinit var model: NLClassifier
    private val initializeTask: Task<Void> by lazy { TfLite.initialize(this) }
    private lateinit var interpreter: InterpreterApi


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        model = NLClassifier.createFromFile(applicationContext,MODEL_PATH)

        initializeTask.addOnSuccessListener {
            val interpreterOption =
                InterpreterApi.Options().setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            interpreter = InterpreterApi.create(
                FileUtil.loadMappedFile(applicationContext, MODEL_PATH),
                interpreterOption
            )}
            .addOnFailureListener { e ->
                Log.e("Interpreter", "Cannot initialize interpreter", e)
            }

        val txtText = findViewById<EditText>(R.id.txtClassify)
        val btnClassify = findViewById<Button>(R.id.btnClassify)
        btnClassify.setOnClickListener {
            classifyUsingTaskLibrary(txtText)
            // classifyUsingTfLite(txtText)
        }

    }

    private fun classifyUsingTaskLibrary(txtText: EditText) {
        val userInputText = txtText.text
        // Toast.makeText(applicationContext, userInputText, Toast.LENGTH_LONG).show()
        val predictions: List<Category> = model.classify(userInputText.toString())
        predictions.forEach { category ->
            val df = DecimalFormat("#.###")
            df.roundingMode = RoundingMode.CEILING
            val outputText = "${category.label} = ${df.format(category.score)}"
            Toast.makeText(applicationContext, outputText, Toast.LENGTH_LONG).show()
        }
    }

    private fun tokenizeText(voc: Map<String, Int>, sentence: String, seqLenght: Int = 32): IntArray {
        val reg = "[^\\w\\']+".toRegex()
        val startId = voc["<START>"]
        val padId = voc["<PAD>"]
        val unknownId = voc["<UNKNOWN>"]
        // extract words from the raw text
        val tokens = reg.split(sentence.lowercase())
        // turn the list on word into a list
        val tokensList = tokens
            .map {voc.getOrElse(it, { unknownId }) }
            .toMutableList()
        // add start
        tokensList.add(0, startId)
        // copy the list of values into a 255 vector, and padding
        val tokensVec =
            tokensList.toTypedArray()
                .copyOf(seqLenght)
        // add padding
        if(sentence.length < seqLenght)
            tokensVec.fill(padId, fromIndex = tokens.size + 1)

        return tokensVec.filterNotNull().toIntArray()

    }


    private fun makePredictions(voc: Map<String, Int>, rawText: String){
        val inputValues = tokenizeText(voc,rawText)
        Log.d("model", rawText)
        Log.d("model", inputValues.joinToString(","))
        val inputs = arrayOf(inputValues)
        val outputs = arrayOf(FloatArray(2))
        interpreter.run(inputs, outputs)
        outputs.forEach {
            it.forEachIndexed { classIdx, score ->
                Log.d("model", "$classIdx : $score")
            }
        }
    }

    private fun classifyUsingTfLite(txtText: EditText) {
        val inputStream = assets.open("vocab.txt")
        val reader = inputStream.bufferedReader()
        // create voc dictionary
        val voc: Map<String, Int> = reader.useLines { lines ->
            lines.toList().map {
                val sequence = it.split(' ')
                sequence[0] to sequence[1].toInt()
            }
        }.toMap()
        makePredictions(voc, txtText.toString())
    }

companion object {
    val MODEL_PATH = "model.tflite"
}

}