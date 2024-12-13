import 'package:animated_splash_screen/animated_splash_screen.dart';
import 'package:camera/camera.dart';
import 'package:image_speak/utils/splash.dart';
import 'package:image_speak/utils/translation_page.dart';
import 'pages/camera_page.dart';
import 'package:flutter/material.dart';
import 'pages/home.dart';
import 'pages/login_register_page.dart';
import 'pages/home_page.dart';
import 'package:flutter_gemini/flutter_gemini.dart';
import 'pages/consts.dart';
import 'pages/image_chat.dart';
import 'utils/splash.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MainApp(cameras: cameras));
}

class MainApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MainApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Splash(
        cameras: cameras,
      ),
    );
  }
}
