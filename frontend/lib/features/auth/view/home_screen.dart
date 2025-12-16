import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:frontend/features/auth/view_model/auth_controller.dart';
import 'package:frontend/features/detection/view/detection_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // Retrieve AuthController to handle logout
    final AuthController authController = Get.find();
    double screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      appBar: AppBar(
        title: const Text('SignFlow'),
        automaticallyImplyLeading: false, // Hide back button on the main screen
        actions: [
          IconButton(
            onPressed: () {
              // Call the logout method
              authController.logout();
            },
            icon: const Icon(Icons.logout),
            tooltip: 'Logout',
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Center(
            child: Container(
              constraints: const BoxConstraints(maxWidth: 450),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  SizedBox(height: screenHeight * 0.1),

                  // Logo Container (using a relevant icon)
                  Container(
                    height: 200,
                    width: 200,
                    decoration: BoxDecoration(
                      color: Colors.green,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Icon(
                      Icons.sign_language, // sign language icon
                      size: 100,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 20),

                  // App Name
                  const Text(
                    'SignFlow',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 36,
                      color: Colors.black87,
                      letterSpacing: 1.5,
                    ),
                  ),
                  const SizedBox(height: 10),
                  const Text(
                    'Pakistani Sign Language Translator',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey,
                    ),
                  ),
                  SizedBox(height: screenHeight * 0.1),

                  // Start Detection Button
                  SizedBox(
                    height: 60,
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      style: ButtonStyle(
                        backgroundColor: MaterialStateProperty.all<Color>(
                            Colors.green.shade700),
                        foregroundColor:
                            MaterialStateProperty.all<Color>(Colors.white),
                        shape:
                            MaterialStateProperty.all<RoundedRectangleBorder>(
                          RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(15),
                          ),
                        ),
                        elevation: MaterialStateProperty.all<double>(5),
                      ),
                      onPressed: () {
                        // Navigate to the Detection Screen
                        Get.to(() => DetectionScreen());
                      },
                      icon: const Icon(Icons.camera_alt, size: 24),
                      label: const Text(
                        'Start Detection',
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.w600),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
