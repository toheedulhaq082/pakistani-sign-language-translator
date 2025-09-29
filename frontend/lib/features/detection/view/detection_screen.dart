import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:camerawesome/camerawesome_plugin.dart';
import '../view_model/detection_controller.dart';

class DetectionScreen extends StatelessWidget {
DetectionScreen({super.key});
final DetectionController detectionController =
Get.put(DetectionController());

@override
Widget build(BuildContext context) {
return Scaffold(
body: Stack(
children: [
CameraAwesomeBuilder.awesome(
saveConfig: SaveConfig.video(),
onMediaCaptureEvent: (event) async {
if (event.isVideo &&
event.status == MediaCaptureStatus.success) {
final filePath = event.captureRequest
.when(single: (single) => single.file?.path ?? "", multiple: (_) => "");
if (filePath.isNotEmpty) {
await detectionController.stopRecording(filePath);
}
}
},
),
Positioned(
top: 50,
left: 20,
right: 20,
child: Obx(() {
if (detectionController.predictedSign.value.isEmpty) {
return const SizedBox.shrink();
}
return Container(
padding: const EdgeInsets.all(12),
decoration: BoxDecoration(
color: Colors.black.withOpacity(0.6),
borderRadius: BorderRadius.circular(10),
),
child: Text(
"${detectionController.predictedSign.value} "
"(${(detectionController.confidence.value * 100).toStringAsFixed(1)}%)",
style: const TextStyle(
color: Colors.white,
fontSize: 18,
fontWeight: FontWeight.bold,
),
textAlign: TextAlign.center,
),
);
}),
),
],
),
);
}
}
