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
              if (event.isVideo && event.status == MediaCaptureStatus.success) {
                final filePath = event.captureRequest.when(
                  single: (single) => single.file?.path ?? "",
                  multiple: (_) => "",
                );
                if (filePath.isNotEmpty) {
                  await detectionController.stopRecording(filePath);
                }
              }
            },
          ),

          // ALWAYS VISIBLE prediction box (white background)
          Positioned(
            top: 40,
            left: 20,
            right: 20,
            child: Obx(() {
              final hasPrediction =
                  detectionController.predictedSign.value.isNotEmpty;

              return Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white, // <-- WHITE background
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.15),
                      blurRadius: 8,
                      offset: const Offset(0, 3),
                    )
                  ],
                ),
                child: Center(
                  child: hasPrediction
                      ? Text(
                          "${detectionController.predictedSign.value} ",
                          style: const TextStyle(
                            color: Colors.black,
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                          ),
                          textAlign: TextAlign.center,
                        )
                      : const Text(
                          "Waiting for prediction...",
                          style: TextStyle(
                            color: Colors.black54,
                            fontSize: 16,
                          ),
                          textAlign: TextAlign.center,
                        ),
                ),
              );
            }),
          ),

          // FULL SCREEN LOADING INDICATOR (center)
          Obx(() {
            if (!detectionController.isLoading.value) {
              return const SizedBox.shrink();
            }

            return Container(
              color: Colors.black.withOpacity(0.3), // dim background
              child: const Center(
                child: CircularProgressIndicator(
                  strokeWidth: 6,
                  color: Colors.white,
                ),
              ),
            );
          }),
        ],
      ),
    );
  }
}
