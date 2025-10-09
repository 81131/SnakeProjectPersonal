plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.walker_81131.snake_identifier_ondevice"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_11.toString()
    }

    defaultConfig {
        applicationId = "com.walker_81131.snake_identifier_ondevice"
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // ✅ Make sure ARM64 is included for real devices
        ndk {
            abiFilters.add("armeabi-v7a")
            abiFilters.add("arm64-v8a")
            abiFilters.add("x86")
            abiFilters.add("x86_64")
        }
    }

    buildTypes {
        release {
            // TODO: Replace with your own signing config for release builds
            signingConfig = signingConfigs.getByName("debug")

            // (Optional) If you enable shrinking for release:
            // isMinifyEnabled = true
            // proguardFiles(
            //     getDefaultProguardFile("proguard-android-optimize.txt"),
            //     "proguard-rules.pro"
            // )
        }
    }

    packagingOptions {
        jniLibs {
            useLegacyPackaging = true    // ✅ keep .so files intact for PyTorch Lite
        }
        resources {
            excludes += setOf("META-INF/*")
        }
    }

    // ✅ Ensure .ptl model files are not compressed by AAPT
    aaptOptions {
        noCompress += listOf("ptl")
    }
}

flutter {
    source = "../.."
}
