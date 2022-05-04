/*
 * Copyright (C) 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import io.reactivex.rxjava3.android.schedulers.AndroidSchedulers
import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.core.ObservableOnSubscribe
import io.reactivex.rxjava3.schedulers.Schedulers
import org.tensorflow.lite.examples.classification.util.CameraProcess
import org.tensorflow.lite.examples.classification.util.ImageProcess
import org.tensorflow.lite.examples.classification.util.Yolov5TFLiteDetector
import org.tensorflow.lite.examples.classification.viewmodel.Recognition
import org.tensorflow.lite.examples.classification.viewmodel.RecognitionListViewModel
import java.util.concurrent.Executors

// Constants
private const val MAX_RESULT_DISPLAY = 3 // Maximum number of results displayed
private const val TAG = "TFL Classify" // Name for logging
private const val REQUEST_CODE_PERMISSIONS = 999 // Return code after asking for permission
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA) // permission needed
private var yolov5TFLiteDetector: Yolov5TFLiteDetector? = null

// Listener for the result of the ImageAnalyzer
typealias RecognitionListener = (recognition: List<Recognition>) -> Unit

/**
 * Main entry point into TensorFlow Lite Classifier
 */
class MainActivity : AppCompatActivity() {

    // CameraX variables
    private lateinit var preview: Preview // Preview use case, fast, responsive view of the camera
    private lateinit var imageAnalyzer: ImageAnalysis // Analysis use case, for running ML code
    private lateinit var camera: Camera
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // Views attachment
    private val resultRecyclerView by lazy {
        findViewById<RecyclerView>(R.id.recognitionResults) // Display the result of analysis
    }
    private val viewFinder by lazy {
        findViewById<PreviewView>(R.id.viewFinder) // Display the preview image from Camera
    }
    private val boxLabelCanvas by lazy {findViewById<ImageView>(R.id.box_label_canvas)}

    // Contains the recognition result. Since  it is a viewModel, it will survive screen rotations
    private val recogViewModel: RecognitionListViewModel by viewModels()

    var rotation = 0

    private val cameraProcess: CameraProcess = CameraProcess()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder.scaleType = PreviewView.ScaleType.FILL_START
        // 获取手机摄像头拍照旋转参数

        // 获取手机摄像头拍照旋转参数
        rotation = windowManager.defaultDisplay.rotation
        Log.i("image", "rotation: $rotation")

        // 初始化加载yolov5s
        initModel("best")

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
//            cameraProcess.startCamera(this@MainActivity, ImageAnalyzer(this, viewFinder,rotation, yolov5TFLiteDetector!!, boxLabelCanvas), viewFinder)
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

//        // Initialising the resultRecyclerView and its linked viewAdaptor
//        val viewAdapter = RecognitionAdapter(this)
//        resultRecyclerView.adapter = viewAdapter
//
//        // Disable recycler view animation to reduce flickering, otherwise items can move, fade in
//        // and out as the list change
//        resultRecyclerView.itemAnimator = null
//
//        // Attach an observer on the LiveData field of recognitionList
//        // This will notify the recycler view to update every time when a new list is set on the
//        // LiveData field of recognitionList.
//        recogViewModel.recognitionList.observe(this,
//            Observer {
//                viewAdapter.submitList(it)
//            }
//        )

    }

    private fun initModel(modelName: String) {
        // 加载模型
        try {
            yolov5TFLiteDetector = Yolov5TFLiteDetector()
            yolov5TFLiteDetector!!.setModelFile(modelName)
            //            this.yolov5TFLiteDetector.addNNApiDelegate();
            yolov5TFLiteDetector!!.addGPUDelegate()
            yolov5TFLiteDetector!!.initialModel(this)
            Log.i("model", "Success loading model" + yolov5TFLiteDetector!!.getModelFile())
        } catch (e: java.lang.Exception) {
            Log.e("image", "load model error: " + e.message + e.toString())
        }
    }

    /**
     * Check all permissions are granted - use for Camera permission in this example.
     */
    private fun allPermissionsGranted(): Boolean = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * This gets called after the Camera permission pop up is shown.
     */
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
//                startCamera()
                cameraProcess.startCamera(this@MainActivity, ImageAnalyzer(this, viewFinder,rotation, yolov5TFLiteDetector!!, boxLabelCanvas), viewFinder)
            } else {
                // Exit the app if permission is not granted
                // Best practice is to explain and offer a chance to re-request but this is out of
                // scope in this sample. More details:
                // https://developer.android.com/training/permissions/usage-notes
                Toast.makeText(
                    this,
                    getString(R.string.permission_deny_text),
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    /**
     * Start the Camera which involves:
     *
     * 1. Initialising the preview use case
     * 2. Initialising the image analyser use case
     * 3. Attach both to the lifecycle of this activity
     * 4. Pipe the output of the preview object to the PreviewView on the screen
     */
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()

            imageAnalyzer = ImageAnalysis.Builder()
                // This sets the ideal size for the image to be analyse, CameraX will choose the
                // the most suitable resolution which may not be exactly the same or hold the same
                // aspect ratio
//                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                // How the Image Analyser should pipe in input, 1. every frame but drop no frame, or
                // 2. go to the latest frame and may drop some frame. The default is 2.
                // STRATEGY_KEEP_ONLY_LATEST. The following line is optional, kept here for clarity
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysisUseCase: ImageAnalysis ->
                    analysisUseCase.setAnalyzer(cameraExecutor, ImageAnalyzer(this, viewFinder,rotation, yolov5TFLiteDetector!!, boxLabelCanvas))
                }

            // Select camera, back is the default. If it is not available, choose front camera
            val cameraSelector =
                if (cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA))
                    CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera - try to bind everything at once and CameraX will find
                // the best combination.
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )

                // Attach the preview to preview view, aka View Finder
                preview.setSurfaceProvider(viewFinder.surfaceProvider)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }


    private class ImageAnalyzer(
        val ctx: Context, val previewView: PreviewView, val rotation: Int,
        val yolov5TFLiteDetector: Yolov5TFLiteDetector, val boxLabelCanvas: ImageView
    ) :
        ImageAnalysis.Analyzer {
        val imageProcess: ImageProcess = ImageProcess()

        class Result( var bitmap: Bitmap)

        override fun analyze(image: ImageProxy) {

            val previewHeight = previewView.height
            val previewWidth = previewView.width

            // Здесь Observable помещает логику анализа изображений в подпоток и получает соответствующие
            // данные обратно при рендеринге пользовательского интерфейса, чтобы избежать отставания внешнего пользовательского интерфейса.
            Observable.create(ObservableOnSubscribe<Result?> { emitter ->
                Log.i("image", "previewWidth $previewWidth/$previewHeight")
                Log.e(
                    "ImageProxy",
                    "rotation ${rotation} ,previewHeight ${previewView.height} ,previewWidth ${previewView.width}"
                )

                val yuvBytes = arrayOfNulls<ByteArray>(3)
                val planes = image.planes
                val imageHeight = image.height
                val imageWidth = image.width

                Log.e(
                    "ImageProxy",
                    "imageHeight ${image.height} ,imageWidth ${image.width} "
                )

                imageProcess.fillBytes(planes, yuvBytes)
                val yRowStride = planes[0].rowStride
                val uvRowStride = planes[1].rowStride
                val uvPixelStride = planes[1].pixelStride
                val rgbBytes = IntArray(imageHeight * imageWidth)
                imageProcess.YUV420ToARGB8888(
                    yuvBytes[0]!!,
                    yuvBytes[1]!!,
                    yuvBytes[2]!!,
                    imageWidth,
                    imageHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes
                )

                // Исходное изображение
                val imageBitmap = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888)
                imageBitmap.setPixels(rgbBytes, 0, imageWidth, 0, 0, imageWidth, imageHeight)

                Log.e(
                    "ImageProxy",
                    "imageBitmap H ${imageBitmap.height} ,imageBitmap W ${imageBitmap.width} "
                )

                // Изображение адаптировано к экрану fill_start формат bitmap
                val scale = Math.max(
                    previewHeight / (if (rotation % 180 == 0) imageWidth else imageHeight).toDouble(),
                    previewWidth / (if (rotation % 180 == 0) imageHeight else imageWidth).toDouble()
                )

                Log.e(
                    "ImageProxy",
                    "scale  ${scale} "
                )
                val fullScreenTransform = imageProcess.getTransformationMatrix(
                    imageWidth, imageHeight,
                    (scale * imageHeight).toInt(), (scale * imageWidth).toInt(),
                    if (rotation % 180 == 0) 90 else 0, false
                )

                // Полноразмерное растровое изображение для предварительного просмотра
                val fullImageBitmap = Bitmap.createBitmap(
                    imageBitmap,
                    0,
                    0,
                    imageWidth,
                    imageHeight,
                    fullScreenTransform,
                    false
                )
                Log.e(
                    "ImageProxyv",
                    "fullImageBitmap H ${fullImageBitmap.height} ,fullImageBitmap W ${fullImageBitmap.width} "
                )
                // Обрезаем растровое изображение до того же размера, что и предварительный просмотр на экране
                val cropImageBitmap = Bitmap.createBitmap(
                    fullImageBitmap, 0, 0,
                    previewWidth, previewHeight
                )

                Log.e(
                    "ImageProxy",
                    "cropImageBitmap H ${cropImageBitmap.height} ,cropImageBitmap W ${cropImageBitmap.width} "
                )

                // Растровое изображение входа модели
                val previewToModelTransform = imageProcess.getTransformationMatrix(
                    cropImageBitmap.width, cropImageBitmap.height,
                    yolov5TFLiteDetector.inputSize.width,
                    yolov5TFLiteDetector.inputSize.height,
                    0, false
                )
                val modelInputBitmap = Bitmap.createBitmap(
                    cropImageBitmap, 0, 0,
                    cropImageBitmap.width, cropImageBitmap.height,
                    previewToModelTransform, false
                )
                val modelToPreviewTransform = Matrix()
                previewToModelTransform.invert(modelToPreviewTransform)
                Log.e(
                    "ImageProxy",
                    "modelInputBitmap H ${modelInputBitmap.height} ,cropImageBitmap W ${modelInputBitmap.width} "
                )
                val recognitions: ArrayList<org.tensorflow.lite.examples.classification.util.Recognition>? = yolov5TFLiteDetector.detect(modelInputBitmap)
                val emptyCropSizeBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
                Log.e(
                    "ImageProxy",
                    "emptyCropSizeBitmap H ${emptyCropSizeBitmap.height} ,cropImageBitmap W ${emptyCropSizeBitmap.width} "
                )
                val cropCanvas = Canvas(emptyCropSizeBitmap)
                Log.e("image", "brands ${recognitions}")
                // Пограничная кисть
                val boxPaint = Paint()
                boxPaint.strokeWidth = 5f
                boxPaint.style = Paint.Style.STROKE
                boxPaint.color = Color.GREEN
                // Кисть шрифта
                val textPain = Paint()
                textPain.textSize = 50f
                textPain.color = Color.RED
                textPain.style = Paint.Style.FILL
                for (res in recognitions!!) {

                    val location: RectF = res.getLocation()
                    val label: String = res.getLabelName()
                    val confidence: Float = res.getConfidence()
                    modelToPreviewTransform.mapRect(location)
                    cropCanvas.drawRect(location, boxPaint)
                    cropCanvas.drawText(label + ":" + String.format("%.2f", confidence), location.left, location.top, textPain)
                }
                image.close()
                emitter.onNext(Result(emptyCropSizeBitmap))
            }).subscribeOn(Schedulers.io()) // Определите здесь watchee, который является потоком в коде выше, если он не определен,
                // то это главный поток синхронный, а не асинхронный
                // Здесь мы возвращаемся в основной поток, где наблюдатель получает данные, отправленные эмиттером, и обрабатывает их
                ?.observeOn(AndroidSchedulers.mainThread()) // Здесь мы возвращаемся в главный поток, чтобы обработать
                // данные обратного вызова из дочернего потока.
                ?.subscribe { result: Result? ->
                    boxLabelCanvas.setImageBitmap(result?.bitmap)
                }


        }
    }
}
