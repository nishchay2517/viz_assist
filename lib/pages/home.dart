import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_speak/pages/camera_page.dart';
import '../utils/translation_page.dart';
import 'package:lottie/lottie.dart';
import '/utils/smart_device_box.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:translator/translator.dart';

speak(String text) async {
  final FlutterTts flutterTts = FlutterTts();
  await flutterTts.setLanguage("en-US");
  await flutterTts.setPitch(1.25);
  await flutterTts.speak(text);
}


class HomePage extends StatefulWidget {
  const HomePage({super.key, required this.cameras});
  final List<CameraDescription> cameras;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // padding constants
  final double horizontalPadding = 40;
  final double verticalPadding = 25;
  String output = "";
  String selectedLanguage = 'en';
  String? destinationLanguage = 'en'; // Initialize with en
  String destCopy = 'en';

  // List of language options
  final List<Map<String, String>> languages = [
    {'name': 'English ', 'code': 'en'},
    {'name': 'Hindi ', 'code': 'hi'},
    {'name': 'Japanese ', 'code': 'ja'},
  ];

  @override
  void initState() {
    speak("Welcome to ImageSpeak. Double tap anywhere to enable camera");
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onDoubleTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => CameraPage(cameras: widget.cameras, destinationLanguage: destCopy),
          ),
        );
      },
      child: Scaffold(
        backgroundColor: const Color(0xFF0D0D0D),
        body: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // app bar
              Padding(
                padding: EdgeInsets.symmetric(
                  horizontal: horizontalPadding,
                  vertical: verticalPadding,
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // menu icon
                    Image.asset(
                      'lib/icons/menu.png',
                      height: 45,
                    ),
                    // account icon
                    Theme(
                      data: ThemeData(
                        canvasColor: Colors.black,
                      ),
                      child: DropdownButton<String>(
                        value: destinationLanguage, // Set the currently selected language
                        focusColor: Colors.blueAccent,
                        icon: Icon(
                          Icons.translate,
                          color: const Color(0xFF4D96AF),
                        ),
                        items: languages.map((language) {
                          return DropdownMenuItem<String>(
                            value: language['code'],
                            child: Text(
                              language['name']!,
                              style: TextStyle(
                                color: const Color(0xFF4D96AF),
                              ),
                            ),
                          );
                        }).toList(),
                        onChanged: (String? value) {
                          setState(() {
                            destinationLanguage = value; // Update destinationLanguage with the selected value
                            destCopy = value!;
                          });
                          print(destinationLanguage);
                        },
                      ),
                    ),
                  ],
                ),
              ),

            const SizedBox(height: 20),

            // welcome home
            Padding(
              padding: EdgeInsets.symmetric(horizontal: horizontalPadding),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(
                    'Hello',
                    style: TextStyle(fontSize: 20, color: Colors.grey.shade700),
                  ),
                  Container(
                    child: AnimatedTextKit(
                      animatedTexts: [
                        TyperAnimatedText(
                          'Vivek Aggarwal',
                          textStyle: TextStyle(
                            fontFamily: GoogleFonts.bebasNeue().fontFamily,
                            fontSize: 62,
                            color: const Color(0xFF4D96AF),
                          ),
                          speed: const Duration(milliseconds: 150),
                        ),
                      ],
                      totalRepeatCount: 1,
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 25),

            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 40.0),
              child: Divider(
                thickness: 1,
                color: Color.fromARGB(255, 204, 204, 204),
              ),
            ),

            const SizedBox(height: 25),

            // smart devices grid
            Padding(
              padding: EdgeInsets.symmetric(horizontal: horizontalPadding),
              child: Text(
                "IMAGE SPEAK",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 30,
                  color: Colors.white,
                ),
              ),
            ),
            const SizedBox(height: 10),
            Padding(
              padding: EdgeInsets.symmetric(horizontal: horizontalPadding),
              child: Text(
                "Empowering the visually impaired with the language of images, providing access to a world of visual information through spoken captions",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 18,
                  color: Colors.grey.shade600,
                ),
              ),
            ),
            Padding(
              padding: EdgeInsets.only(
                  top: 65), // Adjust the top padding value as needed
              child: Container(
                alignment: Alignment.center,
                child: Lottie.asset(
                  "assets/animations/camera.json",
                  repeat: true,
                  height: 80,
                  width: 80,
                ),
              ),
            ),
            Padding(
              padding: EdgeInsets.symmetric(
                  vertical: verticalPadding, horizontal: horizontalPadding),
              child: Text(
                textAlign: TextAlign.center,
                "Double Tap Anywhere to Enable Camera",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 20,
                  color: Colors.grey.shade600,
                ),
              ),
            ),
          ],
        ),
      ),
      ),
    );
  }
}
