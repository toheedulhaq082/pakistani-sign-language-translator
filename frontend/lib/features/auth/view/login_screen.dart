import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:get/get.dart';
import 'package:frontend/features/auth/view/signup_screen.dart';
import 'package:frontend/features/ip/view/ip_screen.dart';

import '../view_model/auth_controller.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final AuthController authController = Get.find();
  final emailController = TextEditingController();
  final passwordController = TextEditingController();
  RxBool loading = false.obs;

  @override
  Widget build(BuildContext context) {
    double screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      appBar: AppBar(
        actions: [
          IconButton(
            onPressed: () => Get.to(() => IPScreen()),
            icon: const Icon(Icons.settings),
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
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(height: screenHeight * 0.04),
                  Align(
                    alignment: Alignment.center,
                    child: Container(
                      height: 200,
                      width: 200,
                      decoration: BoxDecoration(
                        color: Colors.green,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: const Icon(
                        Icons.handshake,
                        size: 100,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                  const Text(
                    'Welcome back,',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 30,
                    ),
                  ),
                  const SizedBox(height: 10),
                  const Text('Email', style: TextStyle(fontWeight: FontWeight.bold)),
                  TextField(
                    controller: emailController,
                    decoration: InputDecoration(
                      prefixIcon: const Icon(Icons.alternate_email),
                      hintText: 'email',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  const Text('Password', style: TextStyle(fontWeight: FontWeight.bold)),
                  TextField(
                    controller: passwordController,
                    obscureText: true,
                    decoration: InputDecoration(
                      prefixIcon: const Icon(Icons.password),
                      hintText: 'password',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Obx(
                    () => loading.value
                        ? const Center(child: CircularProgressIndicator())
                        : SizedBox(
                            height: 50,
                            width: double.infinity,
                            child: ElevatedButton(
                              style: ButtonStyle(
                                backgroundColor: MaterialStateProperty.all<Color>(Colors.black),
                                foregroundColor: MaterialStateProperty.all<Color>(Colors.white),
                              ),
                              onPressed: () async {
                                loading.value = true;
                                if (emailController.text.trim().isEmpty ||
                                    passwordController.text.trim().isEmpty) {
                                  Fluttertoast.showToast(
                                    msg: "Please fill all fields",
                                    backgroundColor: Colors.blue,
                                  );
                                } else {
                                  await authController.signIn(
                                    emailController.text.trim().toLowerCase(),
                                    passwordController.text.trim(),
                                  );
                                }
                                loading.value = false;
                              },
                              child: const Text('Login'),
                            ),
                          ),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text('Don’t have an account? '),
                      TextButton(
                        onPressed: () => Get.to(() => const SignupScreen()),
                        child: const Text('Sign up'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  const Divider(),
                  SizedBox(height: screenHeight * 0.05),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
