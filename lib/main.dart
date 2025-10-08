// lib/main.dart
import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:csv/csv.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SnakeIdentifierApp());
}

class SnakeIdentifierApp extends StatelessWidget {
  const SnakeIdentifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Snake Identifier',
      theme: ThemeData(
        brightness: Brightness.dark,
        primaryColor: Colors.teal,
        fontFamily: 'sans-serif',
        useMaterial3: false,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  ClassificationModel? _model;
  Map<String, dynamic>? _snakeData;
  List<String> _labels = const [];

  File? _image;
  List<Map<String, dynamic>>? _predictions;

  bool _isModelLoading = true;
  bool _isInferLoading = false;
  String? _error; // show full stack here

  final _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    // Defer heavy init to after the 1st frame so the UI renders (prevents "flash & close")
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadEverything();
    });
  }

  Future<void> _loadEverything() async {
    if (!mounted) return;
    setState(() {
      _isModelLoading = true;
      _error = null;
    });

    try {
      // Run model & data loading with a timeout so it can fail gracefully
      await Future.any([
        _loadModelAndData(),
        Future.delayed(const Duration(seconds: 30), () {
          throw TimeoutException('Model loading timed out (30s).');
        }),
      ]);
    } catch (e, st) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load model or data:\n$e\n$st';
      });
    } finally {
      if (mounted) setState(() => _isModelLoading = false);
    }
  }

  /// Reads labels first to determine numberOfClasses dynamically.
  Future<void> _loadModelAndData() async {
    // 1) Load labels (required by pytorch_lite)
    _labels = await _loadLabels('assets/class_names.txt');
    final numberOfClasses = _labels.isEmpty ? 1 : _labels.length;

    // 2) Load the model (pytorch_lite 4.3.2 signature)
    // loadClassificationModel(assetPath, inputW, inputH, numberOfClasses, {labelPath})
    try {
      _model = await PytorchLite.loadClassificationModel(
        'assets/snake_model.ptl',
        224,
        224,
        numberOfClasses,
        labelPath: 'assets/class_names.txt',
      );
    } on PlatformException catch (e, st) {
      // Native errors should be surfaced to the UI instead of crashing
      throw Exception('PlatformException from pytorch_lite: ${e.message}\n$st');
    } catch (e, st) {
      throw Exception('Error loading model: $e\n$st');
    }

    // 3) Load CSV metadata
    try {
      _snakeData = await _loadSnakeCsvData('assets/Snake_Names_And_Venom.csv');
    } catch (e, st) {
      // Not fatal for app lifecycle; still surface to UI
      throw Exception('Error loading CSV data: $e\n$st');
    }

    if (mounted) setState(() {});
  }

  Future<List<String>> _loadLabels(String path) async {
    try {
      final text = await rootBundle.loadString(path);
      // Split & trim; skip empties
      return text
          .split(RegExp(r'\r?\n'))
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList(growable: false);
    } on FlutterError catch (e) {
      // Asset not found
      throw Exception('Labels file not found at "$path". Did you list it in pubspec.yaml?\n$e');
    }
  }

  Future<Map<String, dynamic>> _loadSnakeCsvData(String path) async {
    try {
      final csvString = await rootBundle.loadString(path);
      final rows = const CsvToListConverter(eol: '\n').convert(csvString);
      final Map<String, dynamic> data = {};
      for (int i = 1; i < rows.length; i++) {
        final row = rows[i];
        if (row.length > 2) {
          data[row[0].toString()] = {
            'scientific_name': row[1].toString(),
            'venom_status': row[2].toString(),
          };
        }
      }
      return data;
    } on FlutterError catch (e) {
      throw Exception('CSV asset "$path" not found. Add it to pubspec.yaml.\n$e');
    } catch (e) {
      throw Exception('Failed parsing CSV "$path": $e');
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickImage(source: source);
      if (pickedFile == null) return;

      if (!mounted) return;
      setState(() {
        _image = File(pickedFile.path);
        _predictions = null;
        _isInferLoading = true;
        _error = null;
      });

      await _runInference();
    } catch (e, st) {
      if (!mounted) return;
      setState(() => _error = 'Failed to pick image:\n$e\n$st');
    }
  }

  Future<void> _runInference() async {
    if (!mounted) return;
    if (_image == null || _model == null || _snakeData == null) {
      setState(() => _isInferLoading = false);
      return;
    }

    try {
      // pytorch_lite classification returns a label (String)
      final bytes = await _image!.readAsBytes();
      final label = await _model!.getImagePrediction(bytes);

      final results = <Map<String, dynamic>>[];
      if (label != null && label.isNotEmpty) {
        final info = _snakeData![label] ?? {};
        results.add({
          'snake_name': label,
          'confidence': 100.0, // API doesn’t give a score; show 100% for now
          'details': info,
        });
      }

      if (!mounted) return;
      setState(() => _predictions = results.isNotEmpty ? results : null);
    } catch (e, st) {
      if (!mounted) return;
      setState(() => _error = 'Error during inference:\n$e\n$st');
    } finally {
      if (mounted) setState(() => _isInferLoading = false);
    }
  }

  void _retryLoad() {
    _image = null;
    _predictions = null;
    _model = null;
    _snakeData = null;
    _labels = const [];
    _loadEverything();
  }

  @override
  Widget build(BuildContext context) {
    final canPick = _model != null && _error == null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('🐍 Snake Identifier (On-Device)'),
        backgroundColor: Colors.teal[800],
        actions: [
          IconButton(
            tooltip: 'Retry loading model & data',
            onPressed: _isModelLoading ? null : _retryLoad,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                if (_isModelLoading) ...[
                  const CircularProgressIndicator(),
                  const SizedBox(height: 10),
                  const Text('Loading model, please wait...'),
                  const SizedBox(height: 12),
                ],

                if (_error != null && !_isModelLoading)
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.red.withOpacity(0.15),
                      border: Border.all(color: Colors.red),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: SingleChildScrollView(
                      scrollDirection: Axis.vertical,
                      child: Text(
                        _error!,
                        style: const TextStyle(
                          color: Colors.redAccent,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ),

                const SizedBox(height: 12),

                if (!_isModelLoading && _error == null && _model == null)
                  const Text(
                    '⚠️ Model failed to load. Check assets/ABI & try again.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.orange),
                  ),

                if (_model != null && _image == null && _error == null)
                  const Text(
                    'Upload an image to identify a snake.',
                    style: TextStyle(fontSize: 18),
                    textAlign: TextAlign.center,
                  ),

                if (_image != null)
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.file(_image!, height: 250),
                  ),

                const SizedBox(height: 16),

                if (_isInferLoading) const CircularProgressIndicator(),

                if (_predictions != null) _buildResults(),

                const SizedBox(height: 24),

                if (canPick) ...[
                  const Divider(),
                  const SizedBox(height: 10),
                  const Text(
                    'Select an Image',
                    style: TextStyle(fontSize: 16, color: Colors.grey),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton.icon(
                        onPressed: () => _pickImage(ImageSource.camera),
                        icon: const Icon(Icons.camera_alt),
                        label: const Text('Camera'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.teal,
                        ),
                      ),
                      ElevatedButton.icon(
                        onPressed: () => _pickImage(ImageSource.gallery),
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Gallery'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.cyan,
                        ),
                      ),
                    ],
                  ),
                ],

                const SizedBox(height: 16),
                _buildDisclaimer(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildResults() {
    if (_predictions == null || _predictions!.isEmpty) {
      return const Text('Could not identify a snake.');
    }
    final top = _predictions!.first;
    return Column(
      children: [
        Text('Most Likely Species:', style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 10),
        _buildPredictionCard(top, isTop: true),
      ],
    );
  }

  Widget _buildPredictionCard(Map<String, dynamic> pred, {bool isTop = false}) {
    final name = pred['snake_name']?.toString() ?? 'Unknown';
    final confRaw = pred['confidence'] ?? 0.0;
    final conf = (confRaw is num ? confRaw.toDouble() : 0.0).clamp(0.0, 100.0);
    final sci = pred['details']?['scientific_name']?.toString() ?? 'N/A';
    final venom = pred['details']?['venom_status']?.toString() ?? 'Unknown';

    final venomInfo = {
      'Non': {'label': 'Non-Venomous', 'color': Colors.green},
      'Mild': {'label': 'Mildly Venomous', 'color': Colors.yellow.shade700},
      'Mod': {'label': 'Moderately Venomous', 'color': Colors.orange},
      'High': {'label': 'Highly Venomous', 'color': Colors.red},
    };
    final current = venomInfo[venom] ?? {'label': 'Unknown', 'color': Colors.grey};

    return Card(
      elevation: 4,
      margin: const EdgeInsets.symmetric(vertical: 8),
      color: isTop ? Colors.teal.withOpacity(0.3) : Colors.grey[800],
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            CircularPercentIndicator(
              radius: 35,
              lineWidth: 8,
              animation: true,
              percent: conf / 100.0,
              center: Text(
                '${conf.toStringAsFixed(1)}%',
                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
              ),
              circularStrokeCap: CircularStrokeCap.round,
              progressColor: Colors.tealAccent,
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(name,
                      style:
                      const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                  Text(
                    sci,
                    style: const TextStyle(
                      fontStyle: FontStyle.italic,
                      color: Colors.grey,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: (current['color'] as Color).withOpacity(0.8),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      current['label'].toString(),
                      style: const TextStyle(
                        color: Colors.black,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.orange.withOpacity(0.2),
        border: Border.all(color: Colors.orange),
        borderRadius: BorderRadius.circular(8),
      ),
      child: const Text(
        '⚠️ DISCLAIMER: This app runs entirely on your device. Misidentification is possible. '
            'NEVER handle a snake based on this app\'s identification.',
        textAlign: TextAlign.center,
        style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
      ),
    );
  }
}
