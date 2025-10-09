// main.dart
import 'dart:async';
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
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const CameraScreen()),
                );
              },
              child: const Text('Open Camera'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const GalleryScreen()),
                );
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
/*                              CSV / VENOM HELPERS                           */
/* -------------------------------------------------------------------------- */

class _VenomDisplay {
  final String label;
  final Color color;
  const _VenomDisplay(this.label, this.color);
}

_VenomDisplay _venomFromRaw(String? raw) {
  final s = (raw ?? '').trim().toLowerCase();
  if (s.startsWith('high')) return _VenomDisplay('Highly Venomous', Colors.redAccent);
  if (s.startsWith('mod') || s.startsWith('mid') || s.contains('moder')) {
    return _VenomDisplay('Moderately Venomous', Colors.orangeAccent);
  }
  if (s.startsWith('mild')) return _VenomDisplay('Mildly Venomous', Colors.yellowAccent);
  if (s.startsWith('non')) return _VenomDisplay('Non-Venomous', Colors.greenAccent);
  return const _VenomDisplay('Unknown', Colors.white70);
}

class _SnakeRow {
  final String name; // common name (matches labels)
  final String scientific;
  final String venomRaw; // CSV value: Non/Mild/Mod/High

  const _SnakeRow({required this.name, required this.scientific, required this.venomRaw});
}

Map<String, _SnakeRow> _buildSnakeIndex(List<List<dynamic>> table) {
  final Map<String, _SnakeRow> index = {};
  if (table.isEmpty) return index;

  final headers = table.first.map((e) => e.toString().trim()).toList();
  final nameIdx = headers.indexOf('Snake_Name');
  final sciIdx = headers.indexOf('Scientific_Name');
  final venomIdx = headers.indexOf('Venom_Status');

  for (int i = 1; i < table.length; i++) {
    final row = table[i];
    final name = (nameIdx >= 0 && nameIdx < row.length) ? (row[nameIdx]?.toString() ?? '') : '';
    if (name.isEmpty) continue;
    final sci = (sciIdx >= 0 && sciIdx < row.length) ? (row[sciIdx]?.toString() ?? '') : '';
    final venom = (venomIdx >= 0 && venomIdx < row.length) ? (row[venomIdx]?.toString() ?? '') : '';

    final key = name.trim().toLowerCase();
    index[key] = _SnakeRow(name: name, scientific: sci, venomRaw: venom);
  }
  return index;
}

/* -------------------------------------------------------------------------- */
/*                               SUGGESTION MODEL                             */
/* -------------------------------------------------------------------------- */

class _Suggestion {
  final String label; // class label (as in model)
  final double prob; // 0..1
  final _SnakeRow? row; // null if not found in CSV
  const _Suggestion({required this.label, required this.prob, required this.row});

  _VenomDisplay get venom => _venomFromRaw(row?.venomRaw);
  String get displayName => label.replaceAll('_', ' ');
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
  Map<String, _SnakeRow> _snakeIndex = const {};

  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _loading = true;
  String? _error;
  String _prediction = 'Point your camera at a snake';
  _SnakeRow? _snakeInfo;
  bool _isProcessing = false;
  DateTime? _lastProcessed;

  int _currentCameraIndex = 0;
  bool _torchOn = false;
  bool _liveMode = true; // live prediction vs capture

  List<_Suggestion> _suggestions = const [];
  Uint8List? _lastCapturedBytes; // preview last captured frame

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
      _setError('Initialization failed:\n$e\n$st');
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
        42,
        labelPath: 'assets/class_names.txt',
        ensureMatchingNumberOfClasses: true,
        modelLocation: ModelLocation.asset,
        labelsLocation: LabelsLocation.asset,
      );

      final snakeDataString = await rootBundle.loadString('assets/Snake_Names_And_Venom.csv');
      final table = const CsvToListConverter(eol: '\n').convert(snakeDataString);
      final index = _buildSnakeIndex(table);

      if (!mounted) return;
      setState(() {
        _model = model;
        _snakeIndex = index;
        _loading = false;
        _error = null;
      });
    } catch (e, st) {
      _setError('Failed to load model or CSV:\n$e\n$st');
    }
  }

  Future<void> _initializeCamera() async {
    if (_cameras == null || _cameras!.isEmpty) {
      _setError('No cameras found on this device.');
      return;
    }

    _controller = CameraController(
      _cameras![_currentCameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() => _loading = false);

      if (_liveMode) {
        await _startStream();
      }
    } catch (e, st) {
      _setError('Could not initialize camera:\n$e\n$st');
    }
  }

  Future<void> _startStream() async {
    if (_controller == null || _controller!.value.isStreamingImages) return;
    await _controller!.startImageStream((CameraImage image) async {
      final shouldRun = !_isProcessing &&
          (_lastProcessed == null ||
              DateTime.now().difference(_lastProcessed!).inMilliseconds > 800);
      if (!shouldRun) return;

      _isProcessing = true;
      _lastProcessed = DateTime.now();
      await _runInferenceLive(image);
    });
  }

  Future<void> _stopStream() async {
    if (_controller == null || !_controller!.value.isStreamingImages) return;
    try {
      await _controller!.stopImageStream();
    } catch (_) {}
  }

  Future<void> _toggleTorch() async {
    if (_controller == null) return;
    try {
      _torchOn = !_torchOn;
      await _controller!.setFlashMode(_torchOn ? FlashMode.torch : FlashMode.off);
      if (mounted) setState(() {});
    } catch (_) {}
  }

  Future<void> _toggleMode() async {
    _liveMode = !_liveMode;
    if (_liveMode) {
      await _startStream();
    } else {
      await _stopStream();
    }
    if (mounted) setState(() {});
  }

  Future<void> _switchCamera() async {
    if (_cameras == null || _cameras!.length < 2) return;
    try {
      await _stopStream();
      await _controller?.dispose();
      _currentCameraIndex = (_currentCameraIndex + 1) % _cameras!.length;
      await _initializeCamera();
    } catch (e, st) {
      _setError('Failed to switch camera:\n$e\n$st');
    }
  }

  Future<void> _capturePhotoAndPredict() async {
    if (_controller == null) return;

    try {
      final wasStreaming = _controller!.value.isStreamingImages;
      if (wasStreaming) {
        await _stopStream();
        await Future<void>.delayed(const Duration(milliseconds: 80));
      }

      final XFile file = await _controller!.takePicture();
      final bytes = await file.readAsBytes();
      _lastCapturedBytes = bytes;

      await _runInferenceOnBytes(bytes);

      if (_liveMode && wasStreaming) {
        await _startStream();
      }

      if (mounted) setState(() {});
    } catch (e, st) {
      _setError('Capture failed:\n$e\n$st');
    }
  }

  Future<void> _runInferenceLive(CameraImage cameraImage) async {
    if (_model == null || _model!.labels == null || !mounted) {
      _isProcessing = false;
      return;
    }

    try {
      List<double>? logits;
      try {
        final rotation = _controller?.description.sensorOrientation ?? 0;
        logits = await _model!.getCameraImagePredictionList(
          cameraImage,
          rotation: rotation,
        );
      } catch (_) {
        final bytes = await compute(_convertCameraImage, cameraImage);
        if (bytes == null) return;
        final raw = await _model!.getImagePredictionList(bytes);
        if (raw != null) {
          logits = raw.map((e) => (e as num).toDouble()).toList();
        }
      }

      if (logits == null || logits.isEmpty) return;
      _applyLogits(logits, labels: _model!.labels!);
    } catch (e, st) {
      _setError('Inference error:\n$e\n$st');
    } finally {
      _isProcessing = false;
    }
  }

  Future<void> _runInferenceOnBytes(Uint8List bytes) async {
    if (_model == null || _model!.labels == null || !mounted) return;

    try {
      final img.Image? decoded = img.decodeImage(bytes);
      if (decoded == null) return;
      final resized = img.copyResize(decoded, width: 224, height: 224);
      final processed = Uint8List.fromList(img.encodeJpg(resized));

      final raw = await _model!.getImagePredictionList(processed);
      if (raw == null || raw.isEmpty) return;
      final logits = raw.map((e) => (e as num).toDouble()).toList();
      _applyLogits(logits, labels: _model!.labels!);
    } catch (e, st) {
      _setError('Inference error (capture):\n$e\n$st');
    }
  }

  void _applyLogits(List<double> logits, {required List<String> labels}) {
    final probs = _softmax(logits);

    int maxIndex = 0;
    double maxProb = probs[0];
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIndex = i;
      }
    }

    if (maxIndex < labels.length) {
      final label = labels[maxIndex];
      final confidence = maxProb * 100;
      final info = _findSnakeInfo(label);

      final topK = _topK(probs, labels, k: 3);
      final suggestions = topK
          .map((m) => _Suggestion(
        label: m['label'] as String,
        prob: m['prob'] as double,
        row: _findSnakeInfo(m['label'] as String),
      ))
          .toList();

      if (mounted) {
        setState(() {
          _prediction = '${label.replaceAll('_', ' ')}\nConfidence: ${confidence.toStringAsFixed(2)}%';
          _snakeInfo = info;
          _suggestions = suggestions;
          _error = null;
        });
      }
    }
  }

  _SnakeRow? _findSnakeInfo(String snakeLabel) {
    if (_snakeIndex.isEmpty) return null;
    final key1 = snakeLabel.trim().toLowerCase();
    final key2 = snakeLabel.replaceAll('_', ' ').trim().toLowerCase();
    return _snakeIndex[key1] ?? _snakeIndex[key2];
  }

  void _setError(String msg) {
    if (!mounted) return;
    setState(() {
      _error = msg;
      _loading = false;
    });
  }

  void _showSuggestionDetails(_Suggestion s) {
    final row = s.row;
    final venom = s.venom;
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.black87,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    s.displayName,
                    style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  IconButton(
                    onPressed: () => Navigator.of(ctx).pop(),
                    icon: const Icon(Icons.close, color: Colors.white70),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              if (row != null)
                Text(
                  'Scientific Name: ${row.scientific}',
                  style: const TextStyle(color: Colors.white70, fontStyle: FontStyle.italic),
                ),
              const SizedBox(height: 8),
              Text(
                'Venom: ${venom.label}',
                style: TextStyle(color: venom.color, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              Text(
                'Confidence: ${(s.prob * 100).toStringAsFixed(2)}%',
                style: const TextStyle(color: Colors.white70),
              ),
              const SizedBox(height: 12),
            ],
          ),
        );
      },
    );
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

        if (_lastCapturedBytes != null)
          SafeArea(
            child: Align(
              alignment: Alignment.topLeft,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.memory(
                    _lastCapturedBytes!,
                    width: 120,
                    height: 90,
                    fit: BoxFit.cover,
                  ),
                ),
              ),
            ),
          ),

        SafeArea(
          child: Align(
            alignment: Alignment.topRight,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  FloatingActionButton.small(
                    heroTag: 'mode',
                    onPressed: _toggleMode,
                    child: Icon(_liveMode ? Icons.videocam : Icons.photo_camera),
                    tooltip: _liveMode ? 'Live mode' : 'Photo mode',
                  ),
                  const SizedBox(height: 10),
                  FloatingActionButton.small(
                    heroTag: 'torch',
                    onPressed: _toggleTorch,
                    child: Icon(_torchOn ? Icons.flash_on : Icons.flash_off),
                  ),
                  const SizedBox(height: 10),
                  FloatingActionButton.small(
                    heroTag: 'switch',
                    onPressed: _switchCamera,
                    child: const Icon(Icons.cameraswitch),
                  ),
                ],
              ),
            ),
          ),
        ),

        Positioned(
          bottom: 90,
          left: 0,
          right: 0,
          child: _TopSuggestionsBar(
            suggestions: _suggestions,
            onTap: _showSuggestionDetails,
          ),
        ),

        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: _ResultCard(
            predictionText: _prediction,
            snakeInfo: _snakeInfo,
          ),
        ),

        Positioned(
          bottom: 150,
          left: 0,
          right: 0,
          child: Center(
            child: FloatingActionButton(
              heroTag: 'capture',
              onPressed: _capturePhotoAndPredict,
              child: const Icon(Icons.camera),
            ),
          ),
        ),

        if (_isProcessing)
          Positioned.fill(
            child: IgnorePointer(
              child: Container(
                color: Colors.transparent,
                alignment: Alignment.center,
                child: const DecoratedBox(
                  decoration: BoxDecoration(
                    color: Colors.black45,
                    borderRadius: BorderRadius.all(Radius.circular(12)),
                  ),
                  child: Padding(
                    padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    child: Text(
                      'Analyzing...',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
                ),
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
  Map<String, _SnakeRow> _snakeIndex = const {};

  bool _loading = true;
  String? _error;
  String _prediction = 'Select an image from gallery';
  _SnakeRow? _snakeInfo;
  final ImagePicker _picker = ImagePicker();
  List<_Suggestion> _suggestions = const [];
  Uint8List? _pickedImageBytes;

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
        'assets/snake_model.ptl',
        224,
        224,
        42,
        labelPath: 'assets/class_names.txt',
        ensureMatchingNumberOfClasses: true,
        modelLocation: ModelLocation.asset,
        labelsLocation: LabelsLocation.asset,
      );
      final snakeDataString = await rootBundle.loadString('assets/Snake_Names_And_Venom.csv');
      final table = const CsvToListConverter(eol: '\n').convert(snakeDataString);
      final index = _buildSnakeIndex(table);

      if (!mounted) return;
      setState(() {
        _model = model;
        _snakeIndex = index;
        _loading = false;
        _error = null;
      });
    } catch (e, st) {
      _setError('Failed to load model or CSV:\n$e\n$st');
    }
  }

  Future<void> _pickAndProcessImage() async {
    if (_model == null || _model!.labels == null || !mounted) return;

    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;

    setState(() => _loading = true);
    try {
      final imageBytes = await image.readAsBytes();
      _pickedImageBytes = imageBytes;

      final img.Image? decodedImage = img.decodeImage(imageBytes);
      if (decodedImage == null) {
        _setError('Failed to decode selected image.');
        return;
      }

      final resized = img.copyResize(decodedImage, width: 224, height: 224);
      final processedBytes = Uint8List.fromList(img.encodeJpg(resized));

      final rawList = await _model!.getImagePredictionList(processedBytes);
      if (rawList != null && rawList.isNotEmpty) {
        final logits = rawList.map((e) => (e as num).toDouble()).toList();
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
          final label = _model!.labels![maxIndex];
          final confidence = maxProb * 100;
          final info = _findSnakeInfo(label);

          final topK = _topK(probs, _model!.labels!, k: 3);
          final suggestions = topK
              .map((m) => _Suggestion(
            label: m['label'] as String,
            prob: m['prob'] as double,
            row: _findSnakeInfo(m['label'] as String),
          ))
              .toList();

          if (mounted) {
            setState(() {
              _prediction = '${label.replaceAll('_', ' ')}\nConfidence: ${confidence.toStringAsFixed(2)}%';
              _snakeInfo = info;
              _suggestions = suggestions;
              _loading = false;
              _error = null;
            });
          }
        }
      }
    } catch (e, st) {
      _setError('Error processing image:\n$e\n$st');
    }
  }

  _SnakeRow? _findSnakeInfo(String snakeLabel) {
    if (_snakeIndex.isEmpty) return null;
    final key1 = snakeLabel.trim().toLowerCase();
    final key2 = snakeLabel.replaceAll('_', ' ').trim().toLowerCase();
    return _snakeIndex[key1] ?? _snakeIndex[key2];
  }

  void _setError(String msg) {
    if (!mounted) return;
    setState(() {
      _error = msg;
      _loading = false;
    });
  }

  void _showSuggestionDetails(_Suggestion s) {
    final row = s.row;
    final venom = s.venom;
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.black87,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    s.displayName,
                    style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  IconButton(
                    onPressed: () => Navigator.of(ctx).pop(),
                    icon: const Icon(Icons.close, color: Colors.white70),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              if (row != null)
                Text(
                  'Scientific Name: ${row.scientific}',
                  style: const TextStyle(color: Colors.white70, fontStyle: FontStyle.italic),
                ),
              const SizedBox(height: 8),
              Text(
                'Venom: ${venom.label}',
                style: TextStyle(color: venom.color, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              Text(
                'Confidence: ${(s.prob * 100).toStringAsFixed(2)}%',
                style: const TextStyle(color: Colors.white70),
              ),
              const SizedBox(height: 12),
            ],
          ),
        );
      },
    );
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
            : SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: _pickAndProcessImage,
                child: const Text('Pick Image from Gallery'),
              ),
              const SizedBox(height: 16),
              if (_pickedImageBytes != null)
                ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: Image.memory(
                    _pickedImageBytes!,
                    width: double.infinity,
                    height: 240,
                    fit: BoxFit.cover,
                  ),
                ),
              const SizedBox(height: 16),
              _TopSuggestionsBar(
                suggestions: _suggestions,
                onTap: _showSuggestionDetails,
              ),
              const SizedBox(height: 10),
              _ResultCard(
                predictionText: _prediction,
                snakeInfo: _snakeInfo,
              ),
            ],
          ),
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
                '⚠️ Error (Dev Mode)',
                style: TextStyle(
                  color: Colors.orangeAccent,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
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
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Error message copied to clipboard')),
                  );
                },
                icon: const Icon(Icons.copy),
                label: const Text('Copy Error'),
              ),
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
    debugPrint('Error converting image: $e');
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
      final g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
          .toInt()
          .clamp(0, 255);
      final b = (yValue + 1.772 * (uValue - 128)).toInt().clamp(0, 255);

      out.setPixelRgb(x, y, r, g, b);
    }
  }
  return out;
}

/* ------------------------------ Softmax & Top-K -------------------------- */

List<double> _softmax(List<double> logits) {
  final maxLogit = logits.reduce((a, b) => a > b ? a : b);
  final expVals = logits.map((x) => math.exp(x - maxLogit)).toList();
  final sumExp = expVals.reduce((a, b) => a + b);
  return expVals.map((x) => x / sumExp).toList();
}

List<Map<String, dynamic>> _topK(
    List<double> probs,
    List<String> labels, {
      int k = 3,
    }) {
  final indexed = List.generate(probs.length, (i) => MapEntry(i, probs[i]));
  indexed.sort((a, b) => b.value.compareTo(a.value));
  final top = indexed.take(k).toList();
  return top
      .map((e) => {
    'label': labels[e.key],
    'prob': e.value,
  })
      .toList();
}

/* ------------------------------ Reusable Result Card --------------------- */

class _ResultCard extends StatelessWidget {
  final String predictionText;
  final _SnakeRow? snakeInfo;
  const _ResultCard({required this.predictionText, required this.snakeInfo});

  @override
  Widget build(BuildContext context) {
    final venom = _venomFromRaw(snakeInfo?.venomRaw);

    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(20),
          topRight: Radius.circular(20),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            predictionText,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          if (snakeInfo != null) ...[
            const SizedBox(height: 10),
            Text(
              'Scientific Name: ${snakeInfo!.scientific}',
              style: const TextStyle(
                color: Colors.white70,
                fontStyle: FontStyle.italic,
                fontSize: 14,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'Venom: ${venom.label}',
              style: TextStyle(
                color: venom.color,
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/* ------------------------------ Top Suggestions Bar ---------------------- */

class _TopSuggestionsBar extends StatelessWidget {
  final List<_Suggestion> suggestions;
  final void Function(_Suggestion) onTap;
  const _TopSuggestionsBar({required this.suggestions, required this.onTap});

  @override
  Widget build(BuildContext context) {
    if (suggestions.isEmpty) return const SizedBox.shrink();

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: suggestions.map((s) {
          final venom = s.venom;
          final bg = venom.color.withOpacity(0.18);
          final border = venom.color.withOpacity(0.5);
          return GestureDetector(
            onTap: () => onTap(s),
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 6),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              decoration: BoxDecoration(
                color: bg,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: border),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    s.displayName,
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        'Venom: ${venom.label}',
                        style: TextStyle(color: venom.color, fontSize: 12, fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(width: 10),
                      Text(
                        '${(s.prob * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}
