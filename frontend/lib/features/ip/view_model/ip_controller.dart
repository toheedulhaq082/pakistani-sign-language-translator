import 'package:get/get.dart';
import 'package:shared_preferences/shared_preferences.dart';

class IPController extends GetxController {
  var ipAddress = ''.obs;

  @override
  void onInit() {
    super.onInit();
    loadIP();
  }

  Future<void> loadIP() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    ipAddress.value = prefs.getString('ipAddress') ?? '';
  }

  Future<void> saveIP(String ip) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString('ipAddress', ip);
    ipAddress.value = ip;
  }
}
