import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:frontend/features/auth/view/login_screen.dart';
import 'package:frontend/features/auth/view_model/auth_controller.dart';
import 'package:frontend/features/detection/view_model/detection_controller.dart';
import 'package:frontend/features/ip/view_model/ip_controller.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  Get.put(IPController());
  Get.put(AuthController());

  
  Get.put(DetectionController());

  runApp(const MyApp());
}
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: 'SignFlow',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      debugShowCheckedModeBanner: false,
      home: const LoginScreen(),
    );
  }
}

