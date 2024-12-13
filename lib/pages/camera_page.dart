import 'dart:io';

import 'package:camera/camera.dart';
import 'package:external_path/external_path.dart';
import 'package:flutter/material.dart';
import 'package:gap/gap.dart';
import 'package:media_scanner/media_scanner.dart';
import 'package:path_provider/path_provider.dart';
import 'image_details.dart';
import 'package:shake/shake.dart';
import 'package:flutter_tts/flutter_tts.dart';

speak(String text) async {
  final FlutterTts flutterTts = FlutterTts();
  await flutterTts.setLanguage("en-US");
  await flutterTts.setPitch(1.25);
  await flutterTts.speak(text);
}

class CameraPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const CameraPage({super.key, required this.cameras, required this.destinationLanguage});
  final String destinationLanguage;

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  late CameraController cameraController;
  late Future<void> cameraValue;
  List<File> imagesList = [];
  bool isFlashOn = false;
  bool isRearCamera = true;

  Future<File> saveImage(XFile image) async {
    final downloadPath = await ExternalPath.getExternalStoragePublicDirectory(
        ExternalPath.DIRECTORY_DOWNLOADS);
    final fileName = '${DateTime.now().millisecondsSinceEpoch}.png';
    final file = File('$downloadPath/$fileName');

    try {
      await file.writeAsBytes(await image.readAsBytes());
    } catch (_) {}

    return file;
  }

  void takePicture() async {
    XFile? image;

    if (cameraController.value.isTakingPicture ||
        !cameraController.value.isInitialized) {
      return;
    }

    if (isFlashOn == false) {
      await cameraController.setFlashMode(FlashMode.off);
    } else {
      await cameraController.setFlashMode(FlashMode.torch);
    }
    image = await cameraController.takePicture();

    if (cameraController.value.flashMode == FlashMode.torch) {
      setState(() {
        cameraController.setFlashMode(FlashMode.off);
      });
    }

    final file = await saveImage(image);
    speak("Picture captured successfully!");
    setState(() {
      imagesList.add(file);
    });
    MediaScanner.loadMedia(path: file.path);
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => DetailsPage(
          imagePath: file.path,
          imageFile: file,
          destinationLanguage: widget.destinationLanguage,
        ),
      ),
    );
  }

  void startCamera(int camera) {
    cameraController = CameraController(
      widget.cameras[camera],
      ResolutionPreset.high,
      enableAudio: false,
    );
    cameraValue = cameraController.initialize();
  }

  @override
  void initState() {
    startCamera(0);
    speak("Double tap on the screen or press the Camera button to take a picture");
    super.initState();
  }

  @override
  void dispose() {
    cameraController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return GestureDetector(
      onDoubleTap: () {
        takePicture();
      },
      child: Scaffold(
      floatingActionButton: FloatingActionButton(
        backgroundColor: const Color.fromRGBO(255, 255, 255, .7),
        shape: const CircleBorder(),
        onPressed: takePicture,
        child: const Icon(
          Icons.photo_camera,
          size: 40,
          color: Colors.black87,
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      body: Stack(
        children: [
          FutureBuilder(
            future: cameraValue,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return SizedBox(
                  width: size.width,
                  height: size.height,
                  child: FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: 100,
                      child: CameraPreview(cameraController),
                    ),
                  ),
                );
              } else {
                return const Center(
                  child: CircularProgressIndicator(),
                );
              }
            },
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.only(right: 5, top: 10),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    GestureDetector(
                      onTap: () {
                        setState(() {
                          isFlashOn = !isFlashOn;
                        });
                      },
                      child: Container(
                        decoration: const BoxDecoration(
                          color: Color.fromARGB(50, 0, 0, 0),
                          shape: BoxShape.circle,
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(10),
                          child: isFlashOn
                              ? const Icon(
                            Icons.flash_on,
                            color: Colors.white,
                            size: 30,
                          )
                              : const Icon(
                            Icons.flash_off,
                            color: Colors.white,
                            size: 30,
                          ),
                        ),
                      ),
                    ),
                    const Gap(10),
                    GestureDetector(
                      onTap: () {
                        setState(() {
                          isRearCamera = !isRearCamera;
                        });
                        isRearCamera ? startCamera(0) : startCamera(1);
                      },
                      child: Container(
                        decoration: const BoxDecoration(
                          color: Color.fromARGB(50, 0, 0, 0),
                          shape: BoxShape.circle,
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(10),
                          child: isRearCamera
                              ? const Icon(
                            Icons.flip_camera_android,
                            color: Colors.white,
                            size: 30,
                          )
                              : const Icon(
                            Icons.flip_camera_android,
                            color: Colors.white,
                            size: 30,
                          ),
                        ),
                      ),
                    )
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
      ),
    );
  }
}