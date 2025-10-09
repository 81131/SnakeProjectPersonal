plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")     // <- modern id
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.walker_81131.snake_identifier_ondevice"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    defaultConfig {
        applicationId = "com.walker_81131.snake_identifier_ondevice"
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // ABIs that ship to real phones
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
            abiFilters.add("x86")
            abiFilters.add("x86_64")
            // (optional) drop x86/x86_64 in release to reduce size
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
            // Keep these off until you add keep-rules (see step 4)
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    packaging {
        jniLibs { useLegacyPackaging = true } // fine to keep for .so handling
        resources {
            // No blanket META-INF excludes here (see step 2)
            excludes += listOf(
                "META-INF/LICENSE*",
                "META-INF/NOTICE*",
                "META-INF/AL2.0",
                "META-INF/LGPL2.1"
            )
        }
    }

    // aaptOptions is deprecated — use this:
    androidResources {
        noCompress += setOf("ptl")          // keep .ptl uncompressed
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions { jvmTarget = JavaVersion.VERSION_11.toString() }
}
