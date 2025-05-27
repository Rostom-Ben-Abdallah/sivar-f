// lib/main.dart

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as status;
import 'package:audioplayers/audioplayers.dart';
import 'package:vector_math/vector_math_64.dart' show Vector3;
import 'package:ar_flutter_plugin_updated/ar_flutter_plugin.dart';
import 'package:ar_flutter_plugin_updated/datatypes/config_planedetection.dart';
import 'package:ar_flutter_plugin_updated/datatypes/node_types.dart';
import 'package:ar_flutter_plugin_updated/managers/ar_session_manager.dart';
import 'package:ar_flutter_plugin_updated/managers/ar_object_manager.dart';
import 'package:ar_flutter_plugin_updated/managers/ar_anchor_manager.dart';
import 'package:ar_flutter_plugin_updated/managers/ar_location_manager.dart';
import 'package:ar_flutter_plugin_updated/models/ar_node.dart';

void main() => runApp(const SafeVisionApp());

class SafeVisionApp extends StatelessWidget {
  const SafeVisionApp({Key? key}) : super(key: key);
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'SafeVision AR',
        debugShowCheckedModeBanner: false,
        theme: ThemeData.dark(useMaterial3: true),
        home: const ArAlertPage(),
      );
}

class ArAlertPage extends StatefulWidget {
  const ArAlertPage({Key? key}) : super(key: key);
  @override
  State<ArAlertPage> createState() => _ArAlertPageState();
}

class _ArAlertPageState extends State<ArAlertPage> {
  WebSocketChannel? _ws;
  late AudioPlayer _audio;
  late ARObjectManager _objectManager;

  bool _online = false;
  String _statusMessage = 'Disconnected';

  @override
  void initState() {
    super.initState();
    _audio = AudioPlayer();
    _connectWebSocket();
  }

  void _connectWebSocket() {
    _ws?.sink.close(status.goingAway);
    setState(() {
      _online = false;
      _statusMessage = 'Connecting…';
    });

    try {
      _ws = WebSocketChannel.connect(Uri.parse('ws://192.168.1.12:8765'));
      setState(() {
        _online = true;
        _statusMessage = 'Connected';
      });
      _ws!.stream.listen(
        _handleWsMessage,
        onError: (err) => setState(() {
          _online = false;
          _statusMessage = 'Error';
        }),
        onDone: () => setState(() {
          _online = false;
          _statusMessage = 'Disconnected';
        }),
      );
    } catch (e) {
      setState(() {
        _online = false;
        _statusMessage = 'Connect failed';
      });
    }
  }

  @override
  void dispose() {
    _ws?.sink.close(status.goingAway);
    _audio.dispose();
    super.dispose();
  }

  void _handleWsMessage(dynamic raw) {
    final msg = jsonDecode(raw as String) as Map<String, dynamic>;
    if (msg['type'] == 'alerts') {
      final alerts = msg['alerts'] as List<dynamic>;
      for (var a in alerts) {
        final label = a['label'] as String;
        final coords = (a['coordinates'] as List<dynamic>?) ?? [0.0, 0.0, -1.0];
        final position = Vector3(
          (coords[0] as num).toDouble(),
          (coords[1] as num).toDouble(),
          (coords[2] as num).toDouble(),
        );
        _spawnAlertWithAxes(label, position);
        _showAlertNotification(label, position);
      }
    }
  }

  Future<void> _spawnAlertWithAxes(String label, Vector3 position) async {
    // 1) RGB axes
    final axesNode = ARNode(
      type: NodeType.webGLB,
      uri: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/'
          'master/2.0/AxisLines/glTF-Binary/AxisLines.glb',
      scale: Vector3(0.2, 0.2, 0.2),
      position: position,
    );
    await _objectManager.addNode(axesNode).catchError((e) {
      debugPrint('Axes load error: $e');
    });

    // 2) Your alert model
    String uri;
    if (label.toUpperCase().contains('SMOKING')) {
      uri = 'assets/ss/cigarette.glb';
    } else if (label.toUpperCase().contains('FALL')) {
      uri = 'assets/ss/falling.glb';
    } else {
      uri = 'assets/ss/location_01.glb';
    }
    final alertNode = ARNode(
      type: NodeType.localGLTF2,
      uri: uri,
      scale: Vector3(0.3, 0.3, 0.3),
      position: position,
    );
    await _objectManager.addNode(alertNode).catchError((e) {
      debugPrint('Alert model load error: $e');
    });

    // 3) Beep
    _audio.play(AssetSource('audio/beep-04.mp3'), volume: 0.4);
  }

  void _showAlertNotification(String label, Vector3 pos) {
    final coordsText =
        '(${pos.x.toStringAsFixed(2)}, ${pos.y.toStringAsFixed(2)}, ${pos.z.toStringAsFixed(2)})';
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('$label alert at $coordsText'),
        duration: const Duration(seconds: 3),
      ),
    );
  }

  void _onARViewCreated(
    ARSessionManager sessionManager,
    ARObjectManager objectManager,
    ARAnchorManager anchorManager,
    ARLocationManager locationManager,
  ) {
    _objectManager = objectManager;
    sessionManager.onInitialize(
      showFeaturePoints: false,
      showWorldOrigin: true,
      showPlanes: false,
      handleTaps: false,
    );
    sessionManager.onPlaneOrPointTap = (_) {};
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SafeVision AR')),
      body: Stack(
        children: [
          ARView(
            onARViewCreated: _onARViewCreated,
            planeDetectionConfig: PlaneDetectionConfig.none,
          ),
          Positioned(
            top: 16,
            left: 16,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'WS: $_statusMessage',
                  style: TextStyle(color: _online ? Colors.green : Colors.red),
                ),
                const SizedBox(height: 8),
                ElevatedButton(
                  onPressed: _connectWebSocket,
                  child: const Text('Retry WS'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
