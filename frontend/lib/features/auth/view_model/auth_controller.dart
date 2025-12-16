import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:get/get.dart';
import 'package:frontend/features/auth/view/login_screen.dart';
import 'package:frontend/features/detection/view/detection_screen.dart';
import 'package:frontend/features/detection/view_model/detection_controller.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../../ip/view_model/ip_controller.dart';
import 'package:frontend/features/auth/view/home_screen.dart';

class AuthController extends GetxController {
  final IPController ipController = Get.find();

  RxMap<String, dynamic>? user = RxMap<String, dynamic>({});

  // -------------------------
  // Sign Up
  // -------------------------
  Future<void> signUp(String email, String password) async {
    try {
      final response = await http.post(
        Uri.parse('http://${ipController.ipAddress}:8000/auth/signup/'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"email": email, "password": password}),
      );

      if (response.statusCode == 201) {
        Fluttertoast.showToast(
          msg: "Signed up successfully. Please login.",
          backgroundColor: Colors.green,
        );
        Get.offAll(() => const LoginScreen());
      } else {
        var data = jsonDecode(response.body);
        Fluttertoast.showToast(
          msg: data.toString(),
          backgroundColor: Colors.red,
        );
      }
    } catch (e) {
      print('Signup Exception: $e');
      Fluttertoast.showToast(
        msg: "Unexpected error occurred",
        backgroundColor: Colors.red,
      );
    }
  }

  // -------------------------
  // Sign In
  // -------------------------
  Future<void> signIn(String email, String password) async {
    try {
      final response = await http.post(
        Uri.parse('http://${ipController.ipAddress}:8000/auth/signin/'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"email": email, "password": password}),
      );

      if (response.statusCode == 200) {
        // Save user info
        user!.value = jsonDecode(response.body);

        Fluttertoast.showToast(
          msg: "Signed in successfully.",
          backgroundColor: Colors.green,
        );

        // Register DetectionController
        Get.put(DetectionController());

        // Navigate to Home Screen
        Get.offAll(() => const HomeScreen());
      } else {
        var data = jsonDecode(response.body);
        Fluttertoast.showToast(
          msg: data.toString(),
          backgroundColor: Colors.red,
        );
      }
    } catch (e) {
      print('Signin Exception: $e');
      Fluttertoast.showToast(
        msg: "Unexpected error occurred",
        backgroundColor: Colors.red,
      );
    }
  }

  // -------------------------
  // Logout
  // -------------------------
  void logout() {
    user!.clear();
    Get.offAll(() => const LoginScreen());
  }
}
