import 'dart:io';
import 'dart:convert';
import 'package:get/get.dart';
import 'package:http/http.dart' as http;
import 'package:frontend/features/ip/view_model/ip_controller.dart';

class DetectionController extends GetxController {
var isRecording = false.obs;
var predictedSign = "".obs;
var confidence = 0.0.obs;

final IPController ipController = Get.find<IPController>();

/// Start recording
void startRecording() {
isRecording.value = true;
predictedSign.value = "";
confidence.value = 0.0;
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
return;
}


final uri = Uri.parse("http://$ip:8000/predict/");
var request = http.MultipartRequest("POST", uri);
request.files.add(await http.MultipartFile.fromPath("file", filePath));

try {
  var response = await request.send();
  if (response.statusCode == 200) {
    final body = await response.stream.bytesToString();
    Map<String, dynamic> data = {};

    try {
      data = jsonDecode(body);
    } catch (_) {
      predictedSign.value = "Invalid JSON from server";
      return;
    }

    predictedSign.value = data["prediction"] ?? "Unknown";
    confidence.value = (data["confidence"] ?? 0.0).toDouble();
  } else {
    predictedSign.value = "Error: ${response.statusCode}";
  }
} catch (e) {
  predictedSign.value = "Error: $e";
}


}
}
