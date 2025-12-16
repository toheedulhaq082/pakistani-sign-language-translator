import 'dart:io';
import 'dart:convert';
import 'package:get/get.dart';
import 'package:http/http.dart' as http;
import 'package:frontend/features/ip/view_model/ip_controller.dart';

class DetectionController extends GetxController {
  var isRecording = false.obs;
  var predictedSign = "".obs;
  var confidence = 0.0.obs;

  /// NEW: loading state for spinner
  var isLoading = false.obs;

  final IPController ipController = Get.find<IPController>();

  /// Start recording
  void startRecording() {
    isRecording.value = true;
    predictedSign.value = "";
    confidence.value = 0.0;
    isLoading.value = false;
  }

  /// Stop recording and send to backend
  Future<void> stopRecording(String filePath) async {
    isRecording.value = false;
    await _sendVideoToBackend(filePath);
  }

  /// Upload video to backend and update prediction
  Future<void> _sendVideoToBackend(String filePath) async {
    final ip = ipController.ipAddress;
    if (ip.isEmpty) {
      predictedSign.value = "No IP configured";
      _autoClearPrediction();
      return;
    }

    final uri = Uri.parse("http://$ip:8000/predict/");

    var request = http.MultipartRequest("POST", uri);
    request.files.add(await http.MultipartFile.fromPath("file", filePath));

    isLoading.value = true; // show spinner

    try {
      var response = await request.send();
      final body = await response.stream.bytesToString();

      isLoading.value = false;

      if (response.statusCode == 200) {
        Map<String, dynamic> data = {};

        try {
          data = jsonDecode(body);
        } catch (_) {
          predictedSign.value = "Invalid JSON from server";
          _autoClearPrediction();
          return;
        }

        predictedSign.value = data["prediction"] ?? "Unknown";
        confidence.value = (data["confidence"] ?? 0.0).toDouble();

        _autoClearPrediction();
      } else {
        predictedSign.value = "Error: ${response.statusCode}";
        _autoClearPrediction();
      }
    } catch (e) {
      isLoading.value = false;
      predictedSign.value = "Error: $e";
      _autoClearPrediction();
    }
  }

  /// NEW: auto hide box after 3 seconds
  void _autoClearPrediction() {
    Future.delayed(const Duration(seconds: 3), () {
      predictedSign.value = "";
      confidence.value = 0.0;
    });
  }
}
