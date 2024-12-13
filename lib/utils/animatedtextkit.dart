import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:flutter/material.dart';

class MyAnimatedTextApp extends StatefulWidget {
  const MyAnimatedTextApp({super.key});

  @override
  State<MyAnimatedTextApp> createState() => _MyAnimatedTextAppState();
}

class _MyAnimatedTextAppState extends State<MyAnimatedTextApp> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        flexibleSpace: Container(
          decoration: const BoxDecoration(color: Colors.red),
        ),
        title: const Text(
          'Animated Text App',
          style: TextStyle(
              color: Color.fromARGB(255, 249, 249, 249),
              fontWeight: FontWeight.bold,
              fontSize: 25),
        ),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // SizedBox(
            //   height: 300,
            //   child:
            // AnimatedTextKit(animatedTexts: [
            //   RotateAnimatedText(
            //     'AWESOME',
            //     textStyle: const TextStyle(
            //         color: Colors.red,
            //         fontSize: 30,
            //         fontWeight: FontWeight.bold),
            //     duration: Duration(seconds: 5),
            //     rotateOut: false,
            //     transitionHeight: 150,
            //     alignment: Alignment.center,
            //   ),
            //   // RotateAnimatedText(
            //   //   'OPTISMISTIC',
            //   //   textStyle: const TextStyle(
            //   //       color: Colors.red,
            //   //       fontSize: 30,
            //   //       fontWeight: FontWeight.bold),
            //   // ),
            //   // RotateAnimatedText(
            //   //   'DIFFERENT',
            //   //   textStyle: const TextStyle(
            //   //       color: Colors.red,
            //   //       fontSize: 30,
            //   //       fontWeight: FontWeight.bold),
            //   // )
            //   // FadeAnimatedText('do it!',
            //   //     textStyle: const TextStyle(
            //   //         color: Colors.deepOrange,
            //   //         fontSize: 30,
            //   //         fontWeight: FontWeight.bold),
            //   //     fadeInEnd: 1,
            //   //     fadeOutBegin: 2),
            //   // FadeAnimatedText(
            //   //   'do it! Right!',
            //   //   textStyle: const TextStyle(
            //   //       color: Colors.deepOrange,
            //   //       fontSize: 30,
            //   //       fontWeight: FontWeight.bold),
            //   // ),
            //   // FadeAnimatedText(
            //   //   'do it Right NOW!',
            //   //   textStyle: const TextStyle(
            //   //       color: Colors.deepOrange,
            //   //       fontSize: 30,
            //   //       fontWeight: FontWeight.bold),
            //   // )
            //   // TyperAnimatedText('It is not enough to do your best',
            //   //  textStyle: const TextStyle(
            //   //           color: Colors.teal,
            //   //           fontSize: 30,
            //   //           fontWeight: FontWeight.bold),
            //   //           textAlign: TextAlign.center,
            //   //           speed: Duration(milliseconds: 100),
            //   //           curve: Curves.fastOutSlowIn)
            //   // TypewriterAnimatedText('Discipline is the best tool', textStyle: const TextStyle(
            //   //           color: Colors.indigo,
            //   //           fontSize: 30,
            //   //           fontWeight: FontWeight.bold),
            //   //           textAlign: TextAlign.center,
            //   //           cursor: '#')
            //   // ScaleAnimatedText('Think', textStyle: const TextStyle(
            //   //           color: Colors.deepPurple,
            //   //           fontSize: 50,
            //   //           fontWeight: FontWeight.bold),
            //   //           scalingFactor: 0.01)
            //   // ColorizeAnimatedText('COLORIZE', textStyle:  const TextStyle(
            //   //           color: Colors.deepPurple,
            //   //           fontSize: 50,
            //   //           fontWeight: FontWeight.bold), colors: [
            //   //             Colors.purple,Colors.blue,Colors.yellow,
            //   //             Colors.red
            //   //           ],
            //   //           speed: Duration(seconds: 5))
            //   // WavyAnimatedText('Look at the Wave',textStyle: const TextStyle(
            //   //           color: Colors.blueGrey,
            //   //           fontSize: 30,
            //   //           fontWeight: FontWeight.bold),
            //   //           speed: Duration(milliseconds: 200))
            //   FlickerAnimatedText('FLICKER',textStyle: const TextStyle(
            //             color: Colors.pink,
            //             fontSize: 30,
            //             fontWeight: FontWeight.bold),
            //             entryEnd: 0.1)
            // ],
            // // isRepeatingAnimation: false,
            // // repeatForever: true,
            // totalRepeatCount: 3,
            // displayFullTextOnTap: true,
            // stopPauseOnTap: true,),
            // // )
            TextLiquidFill(text: 'LIQUIDY',
              waveColor: Colors.red,
              textStyle: TextStyle(fontSize: 80,fontWeight: FontWeight.bold),
              boxBackgroundColor: Colors.amber,
              boxHeight: 200,boxWidth: 400,
              loadDuration: Duration(seconds: 9),
              waveDuration: Duration(seconds: 1),
              loadUntil: 0.5,)
          ],
        ),
      ),
    );
  }
}