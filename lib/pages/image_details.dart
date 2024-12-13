import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image_speak/pages/image_chat.dart';
import 'package:image_speak/pages/voice_chat.dart';
import 'package:translator/translator.dart';
import 'chat.dart';
import 'package:http/http.dart' as http;
import 'package:loading_animation_widget/loading_animation_widget.dart';

speak(String text, String dest) async {
  final FlutterTts flutterTts = FlutterTts();

  await flutterTts.getEngines;

  List<dynamic> languages = await flutterTts.getLanguages;
  print(languages);
  var isLanguageAvailable = await flutterTts.isLanguageAvailable('ja-JP');
  if(isLanguageAvailable) {
    print("Language is available");
  } else {
    print("Language is not available");
  }

  // await flutterTts.isLanguageInstalled("ja-JP");
  await flutterTts.setPitch(1.25);
  if(dest == 'ja') {
    await flutterTts.setVoice({"name": "Karen", "locale": "ja-JP"});
    await flutterTts.setLanguage("ja-JP");
  }
  else if(dest == 'hi') {
    await flutterTts.setVoice({"name": "Karen", "locale": "hi-IN"});
    await flutterTts.setLanguage("hi-IN");
  }
  else {
    await flutterTts.setLanguage("en-US");
  }
  await flutterTts.speak(text); // नमस्ते
}

Future<String> translate(String src, String dest, String input) async {
  GoogleTranslator translator = GoogleTranslator();
  String output = "";
  var translation = await translator.translate(input, from: src, to: dest);
    output = translation.text.toString();
    print("object");
  if (src == '--' || dest == '--') {
      output = 'Fail to translate';
  }
  return output;
}

Future<String> performTranslation(String dest,String inp) async {
  return await translate('en', dest, inp);
}

class DetailsPage extends StatefulWidget {
  // requires imagePath
  const DetailsPage(
      {Key? key, required this.imagePath, required this.imageFile, required this.destinationLanguage})
      : super(key: key);
  final String imagePath;
  final File imageFile;
  final String destinationLanguage;

  @override
  State<DetailsPage> createState() => _DetailsPageState();
}

Future<String> encodeImage(String imagePath) async {
  File imageFile = File(imagePath);
  List<int> imageBytes = await imageFile.readAsBytes();
  String encodedString = base64Encode(imageBytes);
  return encodedString;
}

class RequestResult {
  final String text;
  final String title;

  RequestResult(this.text, this.title);
}

Future<RequestResult> makeRequest(path, File file, dest) async {
  var request = http.MultipartRequest(
      'POST', Uri.parse('https://79a4-34-125-213-165.ngrok-free.app/caption'));
  request.files.add(http.MultipartFile.fromBytes('file', file.readAsBytesSync(),
      filename: file.path.split('/').last));
  var streamedResponse = await request.send();
  var res = await http.Response.fromStream(streamedResponse);
  var responseBody = json.decode(res.body);
  String title = responseBody['title'];
  if (res.statusCode == 200) {
    print("Uploaded!");
    print("Title: $title");
    if(dest != 'en')
      title = await performTranslation(dest, title);
    print("Translated Title: $title");
    speak(title, dest);
    return RequestResult('', title);
  } else {
    print("Failed to upload");
    // print error
    print("Server response: $res");
  }
  return RequestResult('text', 'title');
}

class _DetailsPageState extends State<DetailsPage> {
  RequestResult result =
      RequestResult(" Loading Description...", " Loading Title...");

  @override
  void initState() {
    super.initState();
    _initializeRequest();
  }

  Future<void> _initializeRequest() async {
    // Call your async function here
    RequestResult requestResult =
        await makeRequest(widget.imagePath, widget.imageFile, widget.destinationLanguage);
    setState(() {
      result = requestResult;
      String ans = result.title;
      print("TITLE API RESULT: $ans");
    });
  }

  @override
  Widget build(BuildContext context) {
    // get the image path
    final imagePath = widget.imagePath;
    final TextEditingController textController = TextEditingController();

    return GestureDetector(
      onPanUpdate: (details) {
        // Swiping in right direction.
        if (details.delta.dx > 0) {
          Navigator.pop(context);
        }

        // Swiping in left direction.
        if (details.delta.dx < 0) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ChatPage(imageFile: widget.imageFile, initialMessage: "", destinationLanguage: widget.destinationLanguage),
            ),
          );
        }
      },

      child: Scaffold(
        backgroundColor: const Color(0xff121012),
        body: Column(
          children: [
            Expanded(
              child: Stack(
                children: [
                  SingleChildScrollView(
                    child: Column(
                      children: [
                        Container(
                          foregroundDecoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(0),
                            gradient: LinearGradient(
                              begin: Alignment.bottomCenter,
                              end: Alignment.topCenter,
                              colors: [
                                const Color(0xff121012).withOpacity(0.8),
                                Colors.transparent,
                              ],
                            ),
                          ),
                          width: MediaQuery.of(context).size.width,
                          height: MediaQuery.of(context).size.height * 0.75,
                          decoration: BoxDecoration(
                            image: DecorationImage(
                              image: FileImage(File(imagePath)),
                              fit: BoxFit.cover,
                            ),
                          ),
                        ),
                        if (result.title == " Loading Title...")
                          Container(
                            height: MediaQuery.of(context).size.height * 0.50,
                            alignment: Alignment.topCenter,
                            child: LoadingAnimationWidget.threeRotatingDots(
                              color: const Color(0xFF4D96AF),
                              size: 80,
                            ),
                          )
                        else
                          Container(
                            margin: const EdgeInsets.symmetric(horizontal: 20),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const SizedBox(
                                  height: 20,
                                ),
                                Row(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        SizedBox(
                                          width: 300,
                                          height:
                                              100, // Adjust this height according to your layout needs
                                          child: Expanded(
                                            child: Text(
                                              // result.title.substring(1),
                                              result.title,
                                              style: const TextStyle(
                                                color: Colors.white,
                                                fontSize: 20,
                                                fontWeight: FontWeight.w600,
                                              ),
                                            ),
                                          ),
                                        ),
                                        const SizedBox(
                                          height: 10,
                                        ),
                                      ],
                                    ),
                                    Row(
                                      children: [
                                        const Text(
                                          "", // "Speech"
                                          style: TextStyle(
                                            color: Colors.white,
                                            fontSize: 15,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                        const SizedBox(
                                          width: 0,
                                        ),
                                        InkWell(
                                          onTap: () => speak(result.title, widget.destinationLanguage),
                                          child: const Icon(
                                            Icons.volume_up,
                                            color: Colors.white70,
                                            size: 25,
                                          ),
                                        )
                                      ],
                                    )
                                  ],
                                ),
                              ],
                            ),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
