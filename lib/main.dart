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
  File? _image;
  List<Map<String, dynamic>>? _predictions;

  bool _isModelLoading = true;
  bool _isInferLoading = false;
  String? _error; // show full stack here
  final _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    // Defer heavy init to after the 1st frame so the UI renders.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadEverything();
    });
  }

  Future<void> _loadEverything() async {
    setState(() {
      _isModelLoading = true;
      _error = null;
    });

    try {
      await Future.any([
        _loadModelAndData(),
        Future.delayed(const Duration(seconds: 30), () {
          throw TimeoutException('Model loading timed out (30s).');
        }),
      ]);
    } catch (e, st) {
      setState(() {
        _error = 'Failed to load model or data:\n$e\n$st';
      });
    } finally {
      if (mounted) setState(() => _isModelLoading = false);
    }
  }

  Future<void> _loadModelAndData() async {
    // IMPORTANT: set the real class count
    const numberOfClasses = 42;

    // pytorch_lite 4.3.2 signature:
    // loadClassificationModel(assetPath, inputW, inputH, numberOfClasses, {labelPath})
    _model = await PytorchLite.loadClassificationModel(
      'assets/snake_model.ptl',
      224,
      224,
      numberOfClasses,
      labelPath: 'assets/class_names.txt',
    );

    _snakeData = await _loadSnakeCsvData('assets/Snake_Names_And_Venom.csv');
  }

  Future<Map<String, dynamic>> _loadSnakeCsvData(String path) async {
    final csvString = await rootBundle.loadString(path);
    final rows = const CsvToListConverter(eol: '\n').convert(csvString);
    final Map<String, dynamic> data = {};
    for (int i = 1; i < rows.length; i++) {
      final row = rows[i];
      if (row.length > 2) {
        data[row[0]] = {'scientific_name': row[1], 'venom_status': row[2]};
      }
    }
    return data;
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickImage(source: source);
      if (pickedFile == null) return;
      setState(() {
        _image = File(pickedFile.path);
        _predictions = null;
        _isInferLoading = true;
        _error = null;
      });
      await _runInference();
    } catch (e, st) {
      setState(() => _error = 'Failed to pick image:\n$e\n$st');
    }
  }

  Future<void> _runInference() async {
    if (_image == null || _model == null || _snakeData == null) {
      setState(() => _isInferLoading = false);
      return;
    }

    try {
      // 4.3.2 returns a String label for classification
      final label = await _model!.getImagePrediction(await _image!.readAsBytes());

      final results = <Map<String, dynamic>>[];
      if (label != null && label.isNotEmpty) {
        final info = _snakeData![label] ?? {};
        results.add({
          'snake_name': label,
          'confidence': 100.0, // no score available from API
          'details': info,
        });
      }
      setState(() => _predictions = results.isNotEmpty ? results : null);
    } catch (e, st) {
      setState(() => _error = 'Error during inference:\n$e\n$st');
    } finally {
      setState(() => _isInferLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🐍 Snake Identifier (On-Device)'),
        backgroundColor: Colors.teal[800],
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                if (_isModelLoading)
                  const Column(
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 10),
                      Text('Loading model, please wait...'),
                    ],
                  ),
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
                    '⚠️ Model failed to load. Please restart the app or check assets.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.orange),
                  ),

                if (_model != null && _image == null && _error == null)
                  const Text('Upload an image to identify a snake.',
                      style: TextStyle(fontSize: 18)),

                if (_image != null)
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.file(_image!, height: 250),
                  ),
                const SizedBox(height: 16),

                if (_isInferLoading) const CircularProgressIndicator(),

                if (_predictions != null) _buildResults(),

                const SizedBox(height: 24),
                if (_model != null && _error == null) ...[
                  const Divider(),
                  const SizedBox(height: 10),
                  const Text('Select an Image',
                      style: TextStyle(fontSize: 16, color: Colors.grey)),
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
    final name = pred['snake_name'] ?? 'Unknown';
    final conf = (pred['confidence'] ?? 0.0) as double;
    final sci = pred['details']?['scientific_name'] ?? 'N/A';
    final venom = pred['details']?['venom_status'] ?? 'Unknown';

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
              percent: (conf.clamp(0, 100)) / 100,
              center: Text('${conf.toStringAsFixed(1)}%',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
              circularStrokeCap: CircularStrokeCap.round,
              progressColor: Colors.tealAccent,
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(name,
                      style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                  Text(sci,
                      style: const TextStyle(
                          fontStyle: FontStyle.italic, color: Colors.grey)),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: (current['color'] as Color).withOpacity(0.8),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      current['label'].toString(),
                      style: const TextStyle(
                          color: Colors.black, fontWeight: FontWeight.bold),
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
