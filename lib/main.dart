import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:csv/csv.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:pytorch_lite/pytorch_lite.dart';

// This function runs in a background isolate to prevent UI freezing.
// lib/main.dart

// This function will run in a separate isolate to prevent UI freezing
Future<Map<String, dynamic>> _loadDataInBackground(String assetsPath) async {
  // ✅ ADD THIS LINE to initialize services in the background isolate.
  WidgetsFlutterBinding.ensureInitialized();

  // The rest of the function remains the same.
  final modelFuture = PytorchLite.loadClassificationModel(
      '$assetsPath/snake_model.ptl', 224, 224,
      labelPath: '$assetsPath/class_names.txt');
  final snakeDataFuture = rootBundle.loadString('$assetsPath/Snake_Names_And_Venom.csv');

  // Await results
  final model = await modelFuture;
  final snakeDataString = await snakeDataFuture;

  // Process CSV data
  final List<List<dynamic>> csvTable = const CsvToListConverter(eol: '\n').convert(snakeDataString);

  // Return a map of the loaded data
  return {
    'model': model,
    'snakeData': csvTable,
  };
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Snake Identifier',
      theme: ThemeData(primarySwatch: Colors.green),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});
  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
  ClassificationModel? _model;
  List<List<dynamic>>? _snakeData;
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _loading = true;
  String? _error;
  String _prediction = "Point your camera at a snake";
  Map<String, dynamic>? _snakeInfo;
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _loadAndInitialize();
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _loadAndInitialize() async {
    try {
      _cameras = await availableCameras();
      await _loadEverything();
      await _initializeCamera();
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = "Initialization failed: $e";
          _loading = false;
        });
      }
    }
  }

  Future<void> _loadEverything() async {
    if (!mounted) return;
    setState(() => _loading = true);
    try {
      final data = await compute(_loadDataInBackground, 'assets');
      if (mounted) {
        setState(() {
          _model = data['model'];
          _snakeData = data['snakeData'];
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _error = "Failed to load resources: $e");
      }
    }
  }

  Future<void> _initializeCamera() async {
    if (_cameras == null || _cameras!.isEmpty) {
      if (mounted) {
        setState(() {
          _error = "No cameras found.";
          _loading = false;
        });
      }
      return;
    }
    _controller = CameraController(_cameras![0], ResolutionPreset.high, enableAudio: false);
    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() => _loading = false);
      await _controller!.startImageStream((CameraImage image) {
        if (!_isProcessing) {
          _isProcessing = true;
          _runInference(image);
        }
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = "Could not initialize camera: $e";
          _loading = false;
        });
      }
    }
  }

  Future<void> _runInference(CameraImage cameraImage) async {
    if (_model == null || !mounted) {
      _isProcessing = false;
      return;
    }

    try {
      final imageBytes = await compute(_convertCameraImage, cameraImage);
      if (imageBytes == null) {
        _isProcessing = false;
        return;
      }

      // ✅ Corrected: The method is `getImagePredictionList`.
      List? recognitions = await _model!.getImagePredictionList(imageBytes);

      if (recognitions != null && recognitions.isNotEmpty) {
        // The result is a list of scores/confidences. Find the highest one.
        int maxIndex = 0;
        double maxScore = 0.0;
        for (int i = 0; i < recognitions.length; i++) {
          if (recognitions[i] != null && recognitions[i] > maxScore) {
            maxScore = recognitions[i];
            maxIndex = i;
          }
        }

        String label = _model!.labels![maxIndex];
        double confidence = maxScore * 100;

        if (mounted) {
          setState(() {
            _prediction = "${label.replaceAll('_', ' ')}\nConfidence: ${confidence.toStringAsFixed(2)}%";
            _snakeInfo = _findSnakeInfo(label);
          });
        }
      }
    } catch (e) {
      print("Error during inference: $e");
    } finally {
      _isProcessing = false;
    }
  }

  Map<String, dynamic>? _findSnakeInfo(String snakeName) {
    if (_snakeData == null) return null;
    for (var row in _snakeData!) {
      if (row.isNotEmpty && row[0].toString().trim().toLowerCase() == snakeName.trim().toLowerCase()) {
        return {
          'name': row[0],
          'venom': row.length > 1 ? row[1] : 'N/A',
          'details': row.length > 2 ? row[2] : 'No details available.',
        };
      }
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Snake Identifier')),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(child: Text('Error: $_error', textAlign: TextAlign.center, style: const TextStyle(color: Colors.red)));
    }
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Center(child: Text('Initializing camera...'));
    }

    return Stack(
      children: [
        Center(child: CameraPreview(_controller!)),
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: Container(
            padding: const EdgeInsets.all(16.0),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.6),
              borderRadius: const BorderRadius.only(topLeft: Radius.circular(20), topRight: Radius.circular(20)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(_prediction, style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold), textAlign: TextAlign.center),
                if (_snakeInfo != null) ...[
                  const SizedBox(height: 10),
                  Text("Venom: ${_snakeInfo!['venom']}", style: TextStyle(color: _snakeInfo!['venom'].toString().toLowerCase().contains('venomous') ? Colors.redAccent : Colors.greenAccent, fontSize: 16)),
                  const SizedBox(height: 5),
                  Text("${_snakeInfo!['details']}", style: const TextStyle(color: Colors.white, fontSize: 14), textAlign: TextAlign.center),
                ],
              ],
            ),
          ),
        ),
      ],
    );
  }
}

// This function must be a top-level function to be used with `compute`.
Uint8List? _convertCameraImage(CameraImage image) {
  try {
    final img.Image? imgImage = _convertYUV420(image);
    if (imgImage == null) return null;

    // Rotate the image 90 degrees to match portrait orientation.
    final rotatedImage = img.copyRotate(imgImage, angle: 90);
    return Uint8List.fromList(img.encodeJpg(rotatedImage));
  } catch (e) {
    print("Error converting image: $e");
    return null;
  }
}

// Function to convert YUV420 format image to a readable format.
img.Image? _convertYUV420(CameraImage image) {
  final width = image.width;
  final height = image.height;
  final yPlane = image.planes[0].bytes;
  final uPlane = image.planes[1].bytes;
  final vPlane = image.planes[2].bytes;
  final uvRowStride = image.planes[1].bytesPerRow;
  final uvPixelStride = image.planes[1].bytesPerPixel!;

  final out = img.Image(width: width, height: height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final yIndex = y * width + x;
      final uvIndex = uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();

      final yValue = yPlane[yIndex];
      final uValue = uPlane[uvIndex];
      final vValue = vPlane[uvIndex];

      final r = (yValue + 1.402 * (vValue - 128)).toInt().clamp(0, 255);
      final g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128)).toInt().clamp(0, 255);
      final b = (yValue + 1.772 * (uValue - 128)).toInt().clamp(0, 255);

      out.setPixelRgb(x, y, r, g, b);
    }
  }
  return out;
}