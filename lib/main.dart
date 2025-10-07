// lib/main.dart
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pytorch_mobile/enums/dtype.dart';
import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:image/image.dart' as img;
import 'package:csv/csv.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';

// Constants for image processing, must match your model's training
const int imageSize = 224;
const double scaleSize = imageSize * 1.15;
final List<double> mean = [0.485, 0.456, 0.406];
final List<double> std = [0.229, 0.224, 0.225];

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
  Model? _model;
  List<String>? _classNames;
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

  // Load all necessary assets from the assets folder
  Future<void> _loadModelAndData() async {
    try {
      // Load the PyTorch Lite model
      _model = await PyTorchMobile.loadModel('assets/snake_model.ptl');
      // Load the class names
      final classNamesString = await rootBundle.loadString('assets/class_names.txt');
      _classNames = classNamesString.split('\n');
      // Load and parse the snake venom data from CSV
      _snakeData = await _loadSnakeCsvData('assets/Snake_Names_And_Venom.csv');
      setState(() {}); // Update UI to show that loading is complete
    } catch (e) {
      debugPrint("Error loading model or data: $e");
      // Handle error, maybe show a dialog
    }
  }

  Future<Map<String, dynamic>> _loadSnakeCsvData(String path) async {
    final csvString = await rootBundle.loadString(path);
    final List<List<dynamic>> rows =
    const CsvToListConverter(eol: '\n').convert(csvString);
    final Map<String, dynamic> data = {};
    // Skip header row (i=1)
    for (int i = 1; i < rows.length; i++) {
      final row = rows[i];
      if (row.isNotEmpty) {
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

  // The core logic: process image, run model, and handle output
  // The core logic: process image, run model, and handle output
  // The core logic: process image, run model, and handle output
  // The core logic: process image, run model, and handle output
  Future<void> _runInference() async {
    if (_image == null || _model == null || _classNames == null) return;

    // 1. Preprocess the image
    final inputTensor = await _preprocessImage(_image!);

    // 2. Run inference with the arguments packed into a List<dynamic>
    // This is the key change to fix the errors.
    final output = await _model!.getPrediction(
      inputTensor, imageSize as List<int>, imageSize as DType,
    );

    // 3. Post-process the output
    if (output == null) return;
    final probabilities = _softmax(output);
    final top3 = _getTop3(probabilities);

    // 4. Format results for display
    final results = <Map<String, dynamic>>[];
    for (var entry in top3) {
      final snakeName = _classNames![entry['index']];
      final snakeInfo = _snakeData![snakeName] ?? {};
      results.add({
        'snake_name': snakeName,
        'confidence': entry['confidence'] * 100,
        'details': snakeInfo,
      });
    }

    setState(() {
      _predictions = results;
      _isLoading = false;
    });
  }
  Future<Float32List> _preprocessImage(File imageFile) async {
    // Decode the image file to an image object
    img.Image? image = img.decodeImage(await imageFile.readAsBytes());
    if (image == null) throw Exception("Could not decode image");

    // 1. Resize
    img.Image resizedImage;
    if (image.width < image.height) {
      resizedImage = img.copyResize(image, width: scaleSize.round());
    } else {
      resizedImage = img.copyResize(image, height: scaleSize.round());
    }

    // 2. CenterCrop
    int offsetX = (resizedImage.width - imageSize) ~/ 2;
    int offsetY = (resizedImage.height - imageSize) ~/ 2;
    img.Image croppedImage = img.copyCrop(resizedImage, x: offsetX, y: offsetY, width: imageSize, height: imageSize);

    // 3. ToTensor & Normalize
    final imageBytes = croppedImage.getBytes(order: img.ChannelOrder.rgb);
    final imageAsList = Float32List(1 * 3 * imageSize * imageSize);
    int pixelIndex = 0;
    for (int i = 0; i < imageBytes.length; i += 3) {
      final r = imageBytes[i];
      final g = imageBytes[i + 1];
      final b = imageBytes[i + 2];

      imageAsList[pixelIndex] = (r / 255.0 - mean[0]) / std[0];
      imageAsList[imageSize * imageSize + pixelIndex] = (g / 255.0 - mean[1]) / std[1];
      imageAsList[2 * imageSize * imageSize + pixelIndex] = (b / 255.0 - mean[2]) / std[2];

      pixelIndex++;
    }

    return imageAsList;
  }

  // Softmax to convert model output (logits) to probabilities
  List<double> _softmax(List<dynamic> scores) {
    var scoresDouble = scores.map((s) => s as double).toList();
    var maxScore = scoresDouble.reduce(max);
    var exps = scoresDouble.map((s) => exp(s - maxScore)).toList();
    var sumExps = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sumExps).toList();
  }

  // Get top 3 predictions from probabilities
  List<Map<String, dynamic>> _getTop3(List<double> probabilities) {
    var indexedProbs = probabilities.asMap().entries.map((e) {
      return {'index': e.key, 'confidence': e.value};
    }).toList();
    indexedProbs.sort((a, b) => (b['confidence'] as double).compareTo(a['confidence'] as double));
    return indexedProbs.take(3).toList();
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
                // Show a message if model is still loading
                if (_model == null)
                  const Column(
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 10),
                      Text("Loading model, please wait..."),
                    ],
                  ),

                if (_model != null && _image == null)
                  const Text('Upload an image to identify a snake.', style: TextStyle(fontSize: 18)),

                if (_image != null) ClipRRect(borderRadius: BorderRadius.circular(12), child: Image.file(_image!, height: 250)),
                const SizedBox(height: 20),

                if (_isLoading) const CircularProgressIndicator(),

                if (_predictions != null) _buildResults(),

                const SizedBox(height: 30),
                const Divider(),
                if (_model != null) ...[
                  const Text("Select an Image", style: TextStyle(fontSize: 16, color: Colors.grey)),
                  const SizedBox(height: 10),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton.icon(
                        onPressed: () => _pickImage(ImageSource.camera),
                        icon: const Icon(Icons.camera_alt),
                        label: const Text('Camera'),
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.teal),
                      ),
                      ElevatedButton.icon(
                        onPressed: () => _pickImage(ImageSource.gallery),
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Gallery'),
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.cyan),
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

  // UI Widgets for displaying results (can be reused from previous example)
  Widget _buildResults() {
    if (_predictions == null || _predictions!.isEmpty) return const Text("No predictions found.");
    final topPrediction = _predictions![0];
    final otherPredictions = _predictions!.sublist(1);

    return Column(
      children: [
        Text("Most Likely Species:", style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 10),
        _buildPredictionCard(topPrediction, isTop: true),
        const SizedBox(height: 20),
        if (otherPredictions.isNotEmpty) ...[
          Text("Other Possibilities:", style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 10),
          ...otherPredictions.map((p) => _buildPredictionCard(p)).toList(),
        ]
      ],
    );
  }

  Widget _buildPredictionCard(Map<String, dynamic> prediction, {bool isTop = false}) {
    final String name = prediction['snake_name'] ?? 'Unknown';
    final double confidence = prediction['confidence'] ?? 0.0;
    final String scientificName = prediction['details']?['scientific_name'] ?? 'N/A';
    final String venomStatus = prediction['details']?['venom_status'] ?? 'Unknown';

    final venomInfo = {
      "Non": {"label": "Non-Venomous", "color": Colors.green},
      "Mild": {"label": "Mildly Venomous", "color": Colors.yellow},
      "Mod": {"label": "Moderately Venomous", "color": Colors.orange},
      "High": {"label": "Highly Venomous", "color": Colors.red},
    };

    final currentVenom = venomInfo[venomStatus] ?? {"label": "Unknown", "color": Colors.grey};

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
              center: Text("${confidence.toStringAsFixed(1)}%", style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14.0)),
              circularStrokeCap: CircularStrokeCap.round,
              progressColor: Colors.tealAccent,
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(name, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                  Text(scientificName, style: const TextStyle(fontStyle: FontStyle.italic, color: Colors.grey)),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                        color: (currentVenom["color"] as Color).withOpacity(0.8),
                        borderRadius: BorderRadius.circular(8)
                    ),
                    child: Text(currentVenom["label"].toString(), style: const TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
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