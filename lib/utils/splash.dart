import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_speak/pages/home.dart';

class Splash extends StatefulWidget {
  const Splash({super.key, required this.cameras});
  final List<CameraDescription> cameras;

  @override
  _SplashState createState() => _SplashState();
}

class _SplashState extends State<Splash> {
  @override
  void initState() {
    super.initState();
    _navigateToHome();
  }

  _navigateToHome() async {
    await Future.delayed(
        Duration(milliseconds: 750)); // Adjust delay duration as needed
    Navigator.pushReplacement(
        context, MaterialPageRoute(builder: (context) => HomePage(cameras: widget.cameras,)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D0D),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              child: Center(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Image.asset(
                    'lib/icons/icon.png',
                    width: 80,
                    height: 80,
                  ),
                ),
              ),
            ),
            SizedBox(height: 20),
            Text(
              'Image Speak',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Colors.white, // Text color
              ),
            ),
          ],
        ),
      ),
    );
  }
}
