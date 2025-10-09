// main.dart
import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:csv/csv.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:pytorch_lite/pytorch_lite.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Snake Identifier (Dev)',
      theme: ThemeData(primarySwatch: Colors.green),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Snake Identifier (Dev)')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () {
                Navigator.push(context,
                    MaterialPageRoute(builder: (_) => const CameraScreen()));
              },
              child: const Text('Open Camera'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.push(context,
                    MaterialPageRoute(builder: (_) => const GalleryScreen()));
              },
              child: const Text('Open Gallery'),
            ),
          ],
        ),
      ),
    );
  }
}

/* -------------------------------------------------------------------------- */
/*                            CAMERA SCREEN WIDGET                            */
/* -------------------------------------------------------------------------- */

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
  DateTime? _lastProcessed;

  @override
  void initState() {
    super.initState();
    _loadAndInitialize();
  }

  @override
  void dispose() {
    _controller?.stopImageStream();
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _loadAndInitialize() async {
    try {
      _cameras = await availableCameras();
      await _loadEverything();
      await _initializeCamera();
    } catch (e, st) {
      _setError("Initialization failed:\n$e\n$st");
    }
  }

  Future<void> _loadEverything() async {
    if (!mounted) return;
    setState(() => _loading = true);
    try {
      final model = await PytorchLite.loadClassificationModel(
        'assets/snake_model.ptl',
        224,
        224,
        labelPath: 'assets/class_names.txt',
      );

      final snakeDataString =
      await rootBundle.loadString('assets/Snake_Names_And_Venom.csv');
      final csvTable =
      const CsvToListConverter(eol: '\n').convert(snakeDataString);

      if (mounted) {
        setState(() {
          _model = model;
          _snakeData = csvTable;
          _loading = false;
          _error = null;
        });
      }
    } catch (e, st) {
      _setError("Failed to load model or CSV:\n$e\n$st");
    }
  }

  Future<void> _initializeCamera() async {
    if (_cameras == null || _cameras!.isEmpty) {
      _setError("No cameras found on this device.");
      return;
    }
    _controller = CameraController(
      _cameras![0],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() => _loading = false);

      await _controller!.startImageStream((CameraImage image) async {
        if (!_isProcessing &&
            (_lastProcessed == null ||
                DateTime.now().difference(_lastProcessed!).inSeconds > 1)) {
          _isProcessing = true;
          _lastProcessed = DateTime.now();
          await _runInference(image);
        }
      });
    } catch (e, st) {
      _setError("Could not initialize camera:\n$e\n$st");
    }
  }

  Future<void> _runInference(CameraImage cameraImage) async {
    if (_model == null || _model!.labels == null || !mounted) {
      _isProcessing = false;
      return;
    }

    try {
      final imageBytes = await compute(_convertCameraImage, cameraImage);
      if (imageBytes == null) {
        _isProcessing = false;
        return;
      }

      List? recognitions =
      await _model!.getImagePredictionList(imageBytes); // raw logits

      if (recognitions != null && recognitions.isNotEmpty) {
        final logits =
        recognitions.map((e) => (e as num).toDouble()).toList();
        final probs = _softmax(logits);

        int maxIndex = 0;
        double maxProb = probs[0];
        for (int i = 1; i < probs.length; i++) {
          if (probs[i] > maxProb) {
            maxProb = probs[i];
            maxIndex = i;
          }
        }

        if (maxIndex < _model!.labels!.length) {
          String label = _model!.labels![maxIndex];
          double confidence = maxProb * 100;

          if (mounted) {
            setState(() {
              _prediction =
              "${label.replaceAll('_', ' ')}\nConfidence: ${confidence.toStringAsFixed(2)}%";
              _snakeInfo = _findSnakeInfo(label);
              _error = null;
            });
          }
        }
      }
    } catch (e, st) {
      _setError("Inference error:\n$e\n$st");
    } finally {
      _isProcessing = false;
    }
  }

  Map<String, dynamic>? _findSnakeInfo(String snakeName) {
    if (_snakeData == null || _snakeData!.isEmpty) return null;
    for (var row in _snakeData!) {
      if (row.isNotEmpty &&
          row[0] is String &&
          row[0].toString().trim().toLowerCase() ==
              snakeName.trim().toLowerCase()) {
        return {
          'name': row[0],
          'venom':
          row.length > 1 ? (row[1]?.toString() ?? 'N/A') : 'N/A',
          'details': row.length > 2
              ? (row[2]?.toString() ?? 'No details available.')
              : 'No details available.',
        };
      }
    }
    return null;
  }

  void _setError(String msg) {
    if (mounted) {
      setState(() {
        _error = msg;
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Snake Identifier - Camera')),
      body: _buildBody(context),
    );
  }

  Widget _buildBody(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return _ErrorBox(message: _error!);
    }
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Center(child: Text('Initializing camera...'));
    }

    return Stack(
      children: [
        Positioned.fill(
          child: AspectRatio(
            aspectRatio: _controller!.value.aspectRatio,
            child: CameraPreview(_controller!),
          ),
        ),
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: Container(
            padding: const EdgeInsets.all(16.0),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.6),
              borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(20),
                  topRight: Radius.circular(20)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  _prediction,
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
                if (_snakeInfo != null) ...[
                  const SizedBox(height: 10),
                  Text(
                    "Venom: ${_snakeInfo!['venom']}",
                    style: TextStyle(
                        color: _snakeInfo!['venom']
                            .toString()
                            .toLowerCase()
                            .contains('venomous')
                            ? Colors.redAccent
                            : Colors.greenAccent,
                        fontSize: 16),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    "${_snakeInfo!['details']}",
                    style: const TextStyle(
                        color: Colors.white, fontSize: 14),
                    textAlign: TextAlign.center,
                  ),
                ],
              ],
            ),
          ),
        ),
      ],
    );
  }
}

/* -------------------------------------------------------------------------- */
/*                            GALLERY SCREEN WIDGET                           */
/* -------------------------------------------------------------------------- */

class GalleryScreen extends StatefulWidget {
  const GalleryScreen({super.key});

  @override
  _GalleryScreenState createState() => _GalleryScreenState();
}

class _GalleryScreenState extends State<GalleryScreen> {
  ClassificationModel? _model;
  List<List<dynamic>>? _snakeData;
  bool _loading = true;
  String? _error;
  String _prediction = "Select an image from gallery";
  Map<String, dynamic>? _snakeInfo;
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _loadEverything();
  }

  Future<void> _loadEverything() async {
    if (!mounted) return;
    setState(() => _loading = true);
    try {
      final model = await PytorchLite.loadClassificationModel(
          'assets/snake_model.ptl', 224, 224,
          labelPath: 'assets/class_names.txt');
      final snakeDataString =
      await rootBundle.loadString('assets/Snake_Names_And_Venom.csv');
      final csvTable =
      const CsvToListConverter(eol: '\n').convert(snakeDataString);

      if (mounted) {
        setState(() {
          _model = model;
          _snakeData = csvTable;
          _loading = false;
          _error = null;
        });
      }
    } catch (e, st) {
      _setError("Failed to load model or CSV:\n$e\n$st");
    }
  }

  Future<void> _pickAndProcessImage() async {
    if (_model == null || _model!.labels == null || !mounted) return;

    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;

    setState(() => _loading = true);
    try {
      final imageBytes = await image.readAsBytes();
      final img.Image? decodedImage = img.decodeImage(imageBytes);
      if (decodedImage == null) {
        _setError("Failed to decode selected image.");
        return;
      }

      // Resize to model input
      final resized =
      img.copyResize(decodedImage, width: 224, height: 224);
      final processedBytes =
      Uint8List.fromList(img.encodeJpg(resized));

      List? recognitions =
      await _model!.getImagePredictionList(processedBytes);
      if (recognitions != null && recognitions.isNotEmpty) {
        final logits =
        recognitions.map((e) => (e as num).toDouble()).toList();
        final probs = _softmax(logits);

        int maxIndex = 0;
        double maxProb = probs[0];
        for (int i = 1; i < probs.length; i++) {
          if (probs[i] > maxProb) {
            maxProb = probs[i];
            maxIndex = i;
          }
        }

        if (maxIndex < _model!.labels!.length) {
          String label = _model!.labels![maxIndex];
          double confidence = maxProb * 100;
          if (mounted) {
            setState(() {
              _prediction =
              "${label.replaceAll('_', ' ')}\nConfidence: ${confidence.toStringAsFixed(2)}%";
              _snakeInfo = _findSnakeInfo(label);
              _loading = false;
              _error = null;
            });
          }
        }
      }
    } catch (e, st) {
      _setError("Error processing image:\n$e\n$st");
    }
  }

  Map<String, dynamic>? _findSnakeInfo(String snakeName) {
    if (_snakeData == null || _snakeData!.isEmpty) return null;
    for (var row in _snakeData!) {
      if (row.isNotEmpty &&
          row[0] is String &&
          row[0].toString().trim().toLowerCase() ==
              snakeName.trim().toLowerCase()) {
        return {
          'name': row[0],
          'venom':
          row.length > 1 ? (row[1]?.toString() ?? 'N/A') : 'N/A',
          'details': row.length > 2
              ? (row[2]?.toString() ?? 'No details available.')
              : 'No details available.',
        };
      }
    }
    return null;
  }

  void _setError(String msg) {
    if (mounted) {
      setState(() {
        _error = msg;
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Snake Identifier - Gallery')),
      body: Center(
        child: _loading
            ? const CircularProgressIndicator()
            : _error != null
            ? _ErrorBox(message: _error!)
            : Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: _pickAndProcessImage,
              child: const Text('Pick Image from Gallery'),
            ),
            const SizedBox(height: 20),
            Text(_prediction,
                style: const TextStyle(
                    fontSize: 20, fontWeight: FontWeight.bold)),
            if (_snakeInfo != null) ...[
              const SizedBox(height: 10),
              Text(
                "Venom: ${_snakeInfo!['venom']}",
                style: TextStyle(
                    color: _snakeInfo!['venom']
                        .toString()
                        .toLowerCase()
                        .contains('venomous')
                        ? Colors.redAccent
                        : Colors.greenAccent,
                    fontSize: 16),
              ),
              const SizedBox(height: 5),
              Text("${_snakeInfo!['details']}",
                  style: const TextStyle(fontSize: 14)),
            ],
          ],
        ),
      ),
    );
  }
}

/* ----------------------------- Error Box Widget -------------------------- */

class _ErrorBox extends StatelessWidget {
  final String message;
  const _ErrorBox({required this.message});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: SingleChildScrollView(
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.black87,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                "⚠️ Error (Dev Mode)",
                style: TextStyle(
                    color: Colors.orangeAccent,
                    fontSize: 18,
                    fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Text(
                message,
                style: const TextStyle(color: Colors.white, fontSize: 14),
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                onPressed: () {
                  Clipboard.setData(ClipboardData(text: message));
                  ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                      content: Text('Error message copied to clipboard')));
                },
                icon: const Icon(Icons.copy),
                label: const Text("Copy Error"),
              )
            ],
          ),
        ),
      ),
    );
  }
}

/* -------------------------- Image Conversion Utils ----------------------- */

Uint8List? _convertCameraImage(CameraImage image) {
  try {
    final img.Image? imgImage = _convertYUV420(image);
    if (imgImage == null) return null;

    final rotatedImage = img.copyRotate(imgImage, angle: 90);
    return Uint8List.fromList(img.encodeJpg(rotatedImage));
  } catch (e) {
    debugPrint("Error converting image: $e");
    return null;
  }
}

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
      final uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);

      final yValue = yPlane[yIndex];
      final uValue = uPlane[uvIndex];
      final vValue = vPlane[uvIndex];

      final r = (yValue + 1.402 * (vValue - 128)).toInt().clamp(0, 255);
      final g = (yValue -
          0.344136 * (uValue - 128) -
          0.714136 * (vValue - 128))
          .toInt()
          .clamp(0, 255);
      final b =
      (yValue + 1.772 * (uValue - 128)).toInt().clamp(0, 255);

      out.setPixelRgb(x, y, r, g, b);
    }
  }
  return out;
}

/* ------------------------------ Softmax Utils ---------------------------- */

List<double> _softmax(List<double> logits) {
  final maxLogit = logits.reduce((a, b) => a > b ? a : b);
  final expVals =
  logits.map((x) => math.exp(x - maxLogit)).toList();
  final sumExp = expVals.reduce((a, b) => a + b);
  return expVals.map((x) => x / sumExp).toList();
}
