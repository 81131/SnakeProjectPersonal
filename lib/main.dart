// lib/main.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:csv/csv.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';

void main() {
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
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _loadModelAndData();
  }

  // Load model and CSV data
  Future<void> _loadModelAndData() async {
    try {
      _model = await PytorchLite.loadClassificationModel(
          "assets/snake_model.ptl", 224, 224,
          labelPath: "assets/class_names.txt");
      _snakeData = await _loadSnakeCsvData("assets/Snake_Names_And_Venom.csv");
      setState(() {}); // Refresh UI after loading
    } catch (e) {
      debugPrint("Error loading model or data: $e");
    }
  }

  Future<Map<String, dynamic>> _loadSnakeCsvData(String path) async {
    final csvString = await rootBundle.loadString(path);
    final List<List<dynamic>> rows =
    const CsvToListConverter(eol: '\n').convert(csvString);
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
      if (pickedFile != null) {
        setState(() {
          _image = File(pickedFile.path);
          _isLoading = true;
          _predictions = null;
        });
        await _runInference();
      }
    } catch (e) {
      debugPrint("Failed to pick image: $e");
    }
  }
  Future<void> _runInference() async {
    if (_image == null || _model == null || _snakeData == null) return;

    // In your pytorch_lite version, this returns a String (label only)
    final prediction =
    await _model!.getImagePrediction(await _image!.readAsBytes());

    final results = <Map<String, dynamic>>[];

    if (prediction != null && prediction.isNotEmpty) {
      final snakeName = prediction; // already a String
      final snakeInfo = _snakeData![snakeName] ?? {};

      results.add({
        'snake_name': snakeName,
        'confidence': 100.0, // assume full confidence since score is unavailable
        'details': snakeInfo,
      });
    }

    setState(() {
      _predictions = results.isNotEmpty ? results : null;
      _isLoading = false;
    });
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
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: <Widget>[
                if (_model == null)
                  const Column(
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 10),
                      Text("Loading model, please wait..."),
                    ],
                  ),

                if (_model != null && _image == null)
                  const Text('Upload an image to identify a snake.',
                      style: TextStyle(fontSize: 18)),

                if (_image != null)
                  ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.file(_image!, height: 250)),
                const SizedBox(height: 20),

                if (_isLoading) const CircularProgressIndicator(),

                if (_predictions != null) _buildResults(),

                const SizedBox(height: 30),
                if (_model != null) ...[
                  const Divider(),
                  const SizedBox(height: 10),
                  const Text("Select an Image",
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
                            backgroundColor: Colors.teal),
                      ),
                      ElevatedButton.icon(
                        onPressed: () => _pickImage(ImageSource.gallery),
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Gallery'),
                        style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.cyan),
                      ),
                    ],
                  ),
                ],
                const SizedBox(height: 20),
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
      return const Text("Could not identify a snake.");
    }

    final topPrediction = _predictions![0];

    return Column(
      children: [
        Text("Most Likely Species:",
            style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 10),
        _buildPredictionCard(topPrediction, isTop: true),
      ],
    );
  }

  Widget _buildPredictionCard(Map<String, dynamic> prediction,
      {bool isTop = false}) {
    final String name = prediction['snake_name'] ?? 'Unknown';
    final double confidence = prediction['confidence'] ?? 0.0;
    final String scientificName =
        prediction['details']?['scientific_name'] ?? 'N/A';
    final String venomStatus =
        prediction['details']?['venom_status'] ?? 'Unknown';

    final venomInfo = {
      "Non": {"label": "Non-Venomous", "color": Colors.green},
      "Mild": {"label": "Mildly Venomous", "color": Colors.yellow.shade700},
      "Mod": {"label": "Moderately Venomous", "color": Colors.orange},
      "High": {"label": "Highly Venomous", "color": Colors.red},
    };

    final currentVenom =
        venomInfo[venomStatus] ?? {"label": "Unknown", "color": Colors.grey};

    return Card(
      elevation: 4,
      margin: const EdgeInsets.symmetric(vertical: 8),
      color: isTop ? Colors.teal.withOpacity(0.3) : Colors.grey[800],
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            CircularPercentIndicator(
              radius: 35.0,
              lineWidth: 8.0,
              animation: true,
              percent: confidence / 100,
              center: Text("${confidence.toStringAsFixed(1)}%",
                  style: const TextStyle(
                      fontWeight: FontWeight.bold, fontSize: 14.0)),
              circularStrokeCap: CircularStrokeCap.round,
              progressColor: Colors.tealAccent,
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(name,
                      style: const TextStyle(
                          fontSize: 20, fontWeight: FontWeight.bold)),
                  Text(scientificName,
                      style: const TextStyle(
                          fontStyle: FontStyle.italic, color: Colors.grey)),
                  const SizedBox(height: 8),
                  Container(
                    padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                        color:
                        (currentVenom["color"] as Color).withOpacity(0.8),
                        borderRadius: BorderRadius.circular(8)),
                    child: Text(currentVenom["label"].toString(),
                        style: const TextStyle(
                            color: Colors.black, fontWeight: FontWeight.bold)),
                  )
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
        "⚠️ DISCLAIMER: This app runs entirely on your device. Misidentification is possible. NEVER handle a snake based on this app's identification.",
        textAlign: TextAlign.center,
        style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
      ),
    );
  }
}
