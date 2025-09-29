import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../view_model/ip_controller.dart';

class IPScreen extends StatelessWidget {
  final IPController controller = Get.find();
  final TextEditingController ipController = TextEditingController();

  IPScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('IP Address Storage'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: ipController,
              decoration: const InputDecoration(labelText: 'Enter IP Address'),
              onChanged: (value) {
                // Update the controller's ip address on input change
                controller.ipAddress.value = value;
              },
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                controller.saveIP(ipController.text);
                ipController.clear();
              },
              child: const Text('Save IP Address'),
            ),
            const SizedBox(height: 20),
            Obx(() {
              return Text(
                'Stored IP Address: ${controller.ipAddress.value}',
                style: const TextStyle(fontSize: 16),
              );
            }),
          ],
        ),
      ),
    );
  }
}
//