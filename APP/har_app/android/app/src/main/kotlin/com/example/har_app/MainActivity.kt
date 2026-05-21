package com.example.har_app

import io.flutter.embedding.android.FlutterActivity

class MainActivity : FlutterActivity() {
    companion object {
        init {
            System.loadLibrary("tensorflowlite_flex_jni")
        }
    }
}
