// =============================================================================
// HAR App - Reconhecimento de Atividade Humana
// Curso: Aplicações de Aprendizado de Máquina em Sistemas Embarcados
// Aluno: Marcos Cabanas Esteves
//
// Este aplicativo coleta dados do acelerômetro do celular a 20 Hz, agrupa as
// amostras em janelas de 200 leituras (10 s) e usa um modelo TensorFlow Lite
// para classificar a atividade do usuário (Walking, Jogging, Standing, etc.).
//
// Compatibilidade mínima:
// - Android 5.0 (Lollipop, API 21) ou superior (celulares lançados em 2013 / versão do SO compatível em 2014)
// - iOS 11.0 ou superior (a partir do iPhone 5s, lançado em 2013 / versão compatível do SO em 2017)
// =============================================================================

import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:share_plus/share_plus.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:vibration/vibration.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

/// Ponto de entrada do aplicativo.
///
/// Inicializa o Flutter, trava a orientação em retrato 
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  runApp(const ActivityApp());
}

/// Widget raiz: configura o tema Material 3 e define a tela inicial.
class ActivityApp extends StatelessWidget {
  const ActivityApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
      title: 'Identificador de atividade',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
          useMaterial3: true,
        ),
        home: const ActivityScreen(),
      );
}

/// Tela principal do app: exibe a predição em tempo real e os controles de coleta, calibração e exportação.
class ActivityScreen extends StatefulWidget {
  const ActivityScreen({super.key});

  @override
  State<ActivityScreen> createState() => _ActivityScreenState();
}

class _ActivityScreenState extends State<ActivityScreen> {
  // ---------------------------------------------------------------------------
  // Constantes de configuração
  // ---------------------------------------------------------------------------

  /// Versão exibida no canto superior direito da AppBar.
  static const String appVersion = '2.08';

  /// Chave usada para persistir o nome do usuário no cache local.
  static const String _userNamePrefsKey = 'user_name';

  /// Tamanho da janela de inferência: 200 amostras = 10 s a 20 Hz.
  static const int windowSize = 200;

  /// Quantidade de amostras preservadas entre janelas consecutivas (50%).
  static const int overlapSize = windowSize ~/ 2;

  /// Número padrão de janelas analisadas em cada execução do botão Iniciar.
  static const int defaultTotalWindows = 1;

  /// Rótulos das classes na MESMA ordem usada no treino do modelo.
  static const List<String> labels = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking',
  ];

  /// Contagem regressiva (em segundos) antes de iniciar a captura.
  static const int capturePrepSeconds = 10;

  /// Chave usada para persistir a preferência de uso do servidor remoto.
  static const String _useRemoteServerPrefsKey = 'use_remote_server';

  /// URL base do backend na nuvem.
  static const String _backendUrl =
      'https://har-backend-1079589245145.southamerica-east1.run.app';

  // ---------------------------------------------------------------------------
  // Estado de coleta de dados
  // ---------------------------------------------------------------------------

  /// Identificador incremental de cada execução do botão Iniciar (usado nos CSVs exportados para diferenciar coletas).
  int _currentSequence = 0;

  /// Histórico bruto de todas as leituras do acelerômetro coletadas na sessão.
  final List<_AccelRecord> _accelRecords = [];

  /// Interpretador TFLite carregado a partir de assets/model.tflite.
  Interpreter? _interpreter;

  /// Fila circular com as últimas [windowSize] amostras (entrada do modelo).
  final Queue<List<double>> _buffer = Queue();

  // Resultado mostrado na tela.
  String _prediction = '—';
  double _confidence = 0.0; // Agora representa "confiança" baseada em entropia
  String _status = 'Carregando modelo...';

  // Flags de fluxo (controlam habilitação dos botões e estado da UI).
  bool _isCollecting = false;
  bool _isStartingSequence = false;

  // Contagem regressiva visível no centro da tela durante a preparação.
  String _countdownText = '';
  Timer? _countdownTimer;

  // Controle do progresso entre janelas (ex.: "janela 2/3").
  int _currentWindow = 1;
  int _selectedTotalWindows = defaultTotalWindows;

  // Histórico das predições por janela (usado em votação majoritária).
  final List<int> _windowPredictions = <int>[];
  final List<double> _windowConfidences = <double>[];
  final List<_WindowAnalysisRecord> _windowAnalysisRecords =
      <_WindowAnalysisRecord>[];

  // Soma e média das probabilidades por classe (somadas ao longo das janelas).
  final List<double> _classProbSums = List<double>.filled(labels.length, 0.0);
  final List<double> _classProbAverages =
      List<double>.filled(labels.length, 0.0);

  /// Marca o instante da última amostra aceita; usado para garantir 20 Hz.
  DateTime _lastSample = DateTime.now();
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;
  bool _receivedFirstSample = false;
  bool _inferenceFailed = false;
  bool _realTimeMode = false;
  bool _useRemoteServer = false;
  String _userName = '';

  // ---------------------------------------------------------------------------
  // Capacidades de feedback tátil do dispositivo
  // ---------------------------------------------------------------------------
  bool _feedbackCapabilitiesLoaded = false;
  bool _hasVibratorSupport = false;
  bool _hasCustomVibrationSupport = false;
  bool _hasShownNoVibrationHint = false;

  // ===========================================================================
  // SECÇÃO 1 — Ajustes de Calibração
  // ===========================================================================

  /// Converte uma amostra (x, y, z) do acelerômetro do iPhone para a convenção
  /// de eixos esperada pelo modelo treinado com o dataset WISDM.
  ///
  /// Mapeamento fixo validado em testes de calibração: preserva X e inverte Y e Z.
  List<double> _mapSampleToModelAxes(List<double> raw) {
    return [raw[0], -raw[1], -raw[2]];
  }

  // ===========================================================================
  // SECÇÃO 2 — Feedback tátil e sonoro
  // ===========================================================================

  /// Nome usado nos envios para servidor; aplica fallback quando vazio.
  String get _userNameForServer {
    final trimmed = _userName.trim();
    return trimmed.isEmpty ? 'nao informado' : trimmed;
  }

  /// Abre a janela de configurações para nome do usuário e modo Tempo Real.
  Future<void> _openSettingsDialog() async {
    final nameController = TextEditingController(text: _userName);
    var localUserName = _userName;
    var localRealTimeMode = _realTimeMode;
    var localUseRemoteServer = _useRemoteServer;

    final saved = await showDialog<bool>(
      context: context,
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setModalState) {
            return AlertDialog(
              title: const Text('Settings'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    TextField(
                      controller: nameController,
                      decoration: const InputDecoration(
                        labelText: 'Nome do Usuário',
                        hintText: 'nao informado',
                        border: OutlineInputBorder(),
                      ),
                      onChanged: (value) {
                        localUserName = value;
                      },
                    ),
                    const SizedBox(height: 12),
                    SwitchListTile(
                      contentPadding: EdgeInsets.zero,
                      title: const Text('Tempo Real'),
                      value: localRealTimeMode,
                      onChanged: (value) {
                        setModalState(() {
                          localRealTimeMode = value;
                        });
                      },
                    ),
                    SwitchListTile(
                      contentPadding: EdgeInsets.zero,
                      title: const Text('Usar Servidor Remoto'),
                      subtitle: const Text(
                        'Envia dados de aferição para a nuvem',
                      ),
                      value: localUseRemoteServer,
                      onChanged: (value) {
                        setModalState(() {
                          localUseRemoteServer = value;
                        });
                      },
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(dialogContext).pop(false),
                  child: const Text('Cancelar'),
                ),
                FilledButton(
                  onPressed: () => Navigator.of(dialogContext).pop(true),
                  child: const Text('Salvar'),
                ),
              ],
            );
          },
        );
      },
    );

    nameController.dispose();

    if (saved != true || !mounted) return;
    final normalizedUserName = localUserName.trim();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_userNamePrefsKey, normalizedUserName);
    await prefs.setBool(_useRemoteServerPrefsKey, localUseRemoteServer);
    setState(() {
      _userName = normalizedUserName;
      _realTimeMode = localRealTimeMode;
      _useRemoteServer = localUseRemoteServer;
      _status =
          'Configuracao atualizada. Usuario: $_userNameForServer. Pressione "Iniciar" para comecar.';
    });
  }

  /// Toca o sinal de "início de coleta" (1 vibração curta + 1 beep).
  Future<void> _notifyStart() async {
    await _playFeedback(isStart: true);
  }

  /// Toca o sinal de "fim da análise" (2 vibrações + 2 beeps).
  Future<void> _notifyEnd() async {
    await _playFeedback(isStart: false);
  }

  /// Roteia o feedback adequado conforme o evento (início ou fim).
  ///
  /// Combina haptics do sistema, vibração nativa do plugin e som de alerta
  /// para maximizar a chance do usuário perceber o sinal.
  Future<void> _playFeedback({required bool isStart}) async {
    await _ensureFeedbackCapabilities();

    if (isStart) {
      await HapticFeedback.mediumImpact();
      await HapticFeedback.vibrate();
      await _vibratePulses(const [220]);
      await _playAlert(times: 1);
      return;
    }

    await HapticFeedback.heavyImpact();
    await HapticFeedback.vibrate();
    await _vibratePulses(const [260, 320]);
    await _playAlert(times: 2);
  }

  /// Executa uma sequência de pulsos de vibração (em ms).
  ///
  /// Faz um fallback para haptics do sistema quando o dispositivo não tem
  /// vibrador ou não aceita durações customizadas.
  Future<void> _vibratePulses(List<int> durationsMs) async {
    if (!_hasVibratorSupport) {
      // Fallback: usa apenas haptics do sistema quando não há vibrador.
      for (var i = 0; i < durationsMs.length; i++) {
        await HapticFeedback.selectionClick();
        await HapticFeedback.vibrate();
        if (i < durationsMs.length - 1) {
          await Future<void>.delayed(const Duration(milliseconds: 90));
        }
      }

      if (!_hasShownNoVibrationHint && mounted) {
        _hasShownNoVibrationHint = true;
        setState(() {
          _status =
              'Aviso: este dispositivo nao reportou suporte a vibracao. Usando apenas feedback haptico.';
        });
      }
      return;
    }

    try {
      if (_hasCustomVibrationSupport) {
        for (var i = 0; i < durationsMs.length; i++) {
          await Vibration.vibrate(duration: durationsMs[i]);
          if (i < durationsMs.length - 1) {
            await Future<void>.delayed(const Duration(milliseconds: 90));
          }
        }
      } else {
        // Alguns dispositivos não aceitam duração customizada.
        await Vibration.vibrate();
        if (durationsMs.length > 1) {
          await Future<void>.delayed(const Duration(milliseconds: 140));
          await Vibration.vibrate();
        }
      }
    } catch (e) {
      // Se falhar no plugin, mantém fallback haptico para não perder feedback.
      for (var i = 0; i < durationsMs.length; i++) {
        await HapticFeedback.vibrate();
        if (i < durationsMs.length - 1) {
          await Future<void>.delayed(const Duration(milliseconds: 90));
        }
      }
      if (mounted) {
        setState(() {
          _status = 'Falha na vibracao nativa ($e). Usando fallback haptico.';
        });
      }
    }
  }

  /// Consulta uma única vez se o dispositivo tem vibrador e se aceita
  /// durações customizadas, armazenando o resultado em cache.
  Future<void> _ensureFeedbackCapabilities() async {
    if (_feedbackCapabilitiesLoaded) return;
    try {
      final hasVibrator = await Vibration.hasVibrator();
      final hasCustom = await Vibration.hasCustomVibrationsSupport();
      _hasVibratorSupport = hasVibrator;
      _hasCustomVibrationSupport = hasCustom;
    } catch (_) {
      _hasVibratorSupport = false;
      _hasCustomVibrationSupport = false;
    } finally {
      _feedbackCapabilitiesLoaded = true;
    }
  }

  /// Toca o som de alerta do sistema [times] vezes, com pequena pausa entre eles.
  Future<void> _playAlert({required int times}) async {
    for (var i = 0; i < times; i++) {
      try {
        await SystemSound.play(SystemSoundType.alert);
      } catch (_) {
        // Ignora falhas do som e segue com a coleta.
      }
      if (i < times - 1) {
        await Future<void>.delayed(const Duration(milliseconds: 140));
      }
    }
  }

  // ===========================================================================
  // SECÇÃO 3 — Ciclo de vida e carregamento do modelo
  // ===========================================================================

  @override
  void initState() {
    super.initState();
    unawaited(_ensureFeedbackCapabilities());
    _loadModel();
    unawaited(_loadUserSettings());
  }

  /// Carrega configurações persistidas do app, como o nome do usuário.
  Future<void> _loadUserSettings() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final savedUserName = prefs.getString(_userNamePrefsKey) ?? '';
      final savedUseRemoteServer =
          prefs.getBool(_useRemoteServerPrefsKey) ?? false;
      if (!mounted) return;
      setState(() {
        _userName = savedUserName;
        _useRemoteServer = savedUseRemoteServer;
      });
    } catch (_) {
      // Se o cache falhar, o app continua com o fallback "nao informado".
    }
  }

  /// Carrega o modelo TFLite a partir dos assets e prepara os tensores.
  ///
  /// Faz `resizeInputTensor` se o modelo tiver batch dinâmico (-1) e atualiza
  /// a UI com a forma de entrada/saída detectada — útil para depuração.
  Future<void> _loadModel() async {
    try {
      // O caminho deve bater exatamente com o asset declarado no pubspec.yaml.
      final interpreter = await Interpreter.fromAsset('assets/model.tflite');

      // Alguns modelos exportam batch dinamico (-1) e precisam de resize antes
      // do allocate para evitar erro de precondicao em tempo de inferencia.
      final inputShapeBefore =
          List<int>.from(interpreter.getInputTensor(0).shape);
      if (inputShapeBefore.contains(-1)) {
        if (inputShapeBefore.length == 2) {
          interpreter.resizeInputTensor(0, [1, windowSize * 3]);
        } else if (inputShapeBefore.length == 3) {
          interpreter.resizeInputTensor(0, [1, windowSize, 3]);
        } else if (inputShapeBefore.length == 4) {
          interpreter.resizeInputTensor(0, [1, windowSize, 3, 1]);
        }
      }

      interpreter.allocateTensors();
      _interpreter = interpreter;
      final inputShape = List<int>.from(interpreter.getInputTensor(0).shape);
      final outputShape = List<int>.from(interpreter.getOutputTensor(0).shape);
      setState(() {
        _status =
            'Modelo pronto. Entrada=$inputShape Saida=$outputShape. Pressione "Iniciar" para começar.';
      });
    } catch (e) {
      setState(() {
        _status = 'Erro ao carregar modelo: $e';
      });
    }
  }

  // ===========================================================================
  // SECÇÃO 4 — Fluxo de coleta (botão Iniciar)
  // ===========================================================================

  /// Executado ao tocar no botão "Iniciar".
  ///
  /// Reseta os buffers e contadores, mostra a contagem regressiva e, ao
  /// terminar, ativa o stream do acelerômetro chamando [_startSensor].
  void _start() {
    if (_isCollecting || _isStartingSequence) {
      return;
    }

    _accelerometerSub?.cancel();

    // Cada toque em "Iniciar" é uma nova sequência (refletida no CSV).
    _currentSequence++;
    _countdownTimer?.cancel();
    setState(() {
      _isStartingSequence = true;
      _prediction = '—';
      _confidence = 0.0;
      _buffer.clear();
      _receivedFirstSample = false;
      _windowPredictions.clear();
      _windowConfidences.clear();
      for (var i = 0; i < labels.length; i++) {
        _classProbSums[i] = 0.0;
        _classProbAverages[i] = 0.0;
      }
      _currentWindow = 1;
      _countdownText = '$capturePrepSeconds';
      _status = 'Prepare o celular e aguarde...';
    });

    // Contagem regressiva de capturePrepSeconds até 1, depois "INICIANDO!".
    var countdownValue = capturePrepSeconds;
    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }

      countdownValue--;
      if (countdownValue > 0) {
        setState(() {
          _countdownText = '$countdownValue';
        });
        return;
      }

      timer.cancel();
      setState(() {
        _countdownText = 'INICIANDO!';
        _status = 'INICIANDO!';
      });

      // Pequena pausa para o usuário ler "INICIANDO!" antes do sensor ligar.
      Future<void>.delayed(const Duration(milliseconds: 900), () {
        if (!mounted || !_isStartingSequence) return;
        setState(() {
          _isStartingSequence = false;
          _isCollecting = true;
          _countdownText = '';
          _status =
              'Coletando janela $_currentWindow/$_selectedTotalWindows...';
        });
        unawaited(_notifyStart());
        _lastSample = DateTime.fromMillisecondsSinceEpoch(0);
        _startSensor();
      });
    });
  }

  // ===========================================================================
  // SECÇÃO 5 — Coleta de amostras e inferência
  // ===========================================================================

  /// Liga o stream do acelerômetro a 20 Hz e processa cada amostra recebida.
  ///
  /// Para cada evento:
  ///  1) Aplica coletas de 50 ms (=20 Hz) para casar com o WISDM;
  ///  2) Empilha no buffer os valores que estamos trabalhando para a janela atual;
  ///  3) Quando o buffer atinge [windowSize], dispara [_runInference].
  ///
  ///  Detalhe do protótipo: mantenho a tela ligada com Wakelock e aviso
  ///  caso nenhum evento chegue em 5 s (provavelmente permissão de movimento negada).
  void _startSensor() {
    WakelockPlus.enable(); // Mantém a tela ligada durante a coleta.

    // IMPORTANTE: aqui forcei 20 Hz (50 ms) para casar com a taxa do dataset WISDM. 
    // Sem isso o iOS entrega eventos em ~5 Hz por padrão, distorcendo o padrão temporal
    // de atividades dinâmicas (Walking/Downstairs/Upstairs).
    _accelerometerSub = accelerometerEventStream(
      samplingPeriod: const Duration(milliseconds: 50),
    ).listen((event) {
      if (!_isCollecting) return;

      final now = DateTime.now();
      if (now.difference(_lastSample).inMilliseconds < 50) return;
      _lastSample = now;

      if (!_receivedFirstSample && mounted) {
        _receivedFirstSample = true;
        setState(() {
          _status =
              'Sensores ativos. Coletando janela $_currentWindow/$_selectedTotalWindows...';
        });
      }

      // Adiciona amostra mapeada ao buffer da janela.
      final rawSample = [event.x, event.y, event.z];
      final mappedSample = _mapSampleToModelAxes(rawSample);
      _buffer.addLast(mappedSample);

      // Mantém histórico bruto para exportação em CSV.
      _accelRecords.add(_AccelRecord(
        sequence: _currentSequence,
        dateTime: DateTime.now(),
        x: event.x,
        y: event.y,
        z: event.z,
      ));
      if (_buffer.length > windowSize) {
        _buffer.removeFirst();
      }

      // Atualiza a UI somente em marcos relevantes para não sobrecarregar.
      if (mounted &&
          (_buffer.length == 1 ||
              _buffer.length == windowSize ||
              _buffer.length % 20 == 0)) {
        setState(() {});
      }

      // Janela completa → roda inferência.
      if (_buffer.length == windowSize) {
        _runInference();
      }

      // Tempo Real: abordagem diferenciada para este casa. Aqui ao invés de
      // inferência ao término da janela, usamos inferência progressiva a cada 
      // 20 novas amostras.
      if (_realTimeMode &&
          _buffer.length >= 20 &&
          _buffer.length < windowSize &&
          _buffer.length % 20 == 0) {
        _runInference(isTiled: true, windowData: _buildTiledWindow());
      }
    }, onError: (error) {
      if (!mounted) return;
      setState(() {
        _status = 'Erro no sensor: $error';
      });
    });

    // Se nenhum evento chegar em alguns segundos, mostra dica de permissão.
    Future<void>.delayed(const Duration(seconds: 5), () {
      if (!mounted || _receivedFirstSample) return;
      setState(() {
        _status =
            'Sem dados do sensor. Verifique Ajustes > Privacidade e Seguranca > Movimento e Condicionamento Fisico.';
      });
    });
  }

  /// Executa a inferência TFLite sobre o buffer atual e atualiza a UI.
  ///
  /// Suporta três formatos comuns de entrada do modelo:
  /// `[1, 200, 3]`, `[1, 600]` e `[1, 200, 3, 1]`. Quando todas as janelas
  /// configuradas tiverem sido processadas, faz votação majoritária entre
  /// elas (com desempate por confiança média) para definir a classe final.
  void _runInference({bool isTiled = false, List<List<double>>? windowData}) {
    final interpreter = _interpreter;
    if (interpreter == null || _inferenceFailed) return;
    if (!isTiled && _buffer.length != windowSize) return;

    final sourceWindow = windowData ?? _buffer.toList();

    final inputTensor = interpreter.getInputTensor(0);
    final outputTensor = interpreter.getOutputTensor(0);
    final inputShape = List<int>.from(inputTensor.shape);

    if (inputTensor.type != TensorType.float32) {
      if (!mounted) return;
      setState(() {
        _inferenceFailed = true;
        _status =
            'Erro na inferencia: tipo de entrada nao suportado (${inputTensor.type})';
      });
      return;
    }

    // Constrói o tensor de entrada conforme o shape esperado pelo modelo.
    Object? input;
    if (inputShape.length == 3 &&
        inputShape[0] == 1 &&
        inputShape[1] == windowSize &&
        inputShape[2] == 3) {
      // [1, 200, 3]
      input = [sourceWindow.map((s) => List<double>.from(s)).toList()];
    } else if (inputShape.length == 2 &&
        inputShape[0] == 1 &&
        inputShape[1] == windowSize * 3) {
      // [1, 600] — achata as amostras
      final flat = <double>[];
      for (final s in sourceWindow) {
        flat.addAll(s);
      }
      input = [flat];
    } else if (inputShape.length == 4 &&
        inputShape[0] == 1 &&
        inputShape[1] == windowSize &&
        inputShape[2] == 3 &&
        inputShape[3] == 1) {
      // [1, 200, 3, 1]
      input = [
        sourceWindow
            .map((s) => s.map((v) => <double>[v]).toList())
            .toList(),
      ];
    } else {
      if (!mounted) return;
      setState(() {
        _inferenceFailed = true;
        _status = 'Erro na inferencia: shape de entrada nao suportado ($inputShape)';
      });
      return;
    }

    // Prepara o buffer de saída conforme o shape do tensor de saída.
    final outputShape = List<int>.from(outputTensor.shape);
    final numClasses =
        outputShape.isNotEmpty ? outputShape.last : labels.length;
    final output = [List<double>.filled(numClasses, 0.0)];

    try {
      interpreter.run(input, output);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _inferenceFailed = true;
        _status = 'Erro na inferencia: $e';
      });
      return;
    }

    final probs = _flattenToDoubleList(output);
    if (probs.isEmpty) return;
    final safeProbs = List<double>.generate(
      labels.length,
      (i) => i < probs.length ? probs[i] : 0.0,
    );
    final maxVal = probs.reduce((a, b) => a > b ? a : b);
    final maxIdx = probs.indexWhere((p) => p == maxVal);

    // Calcula entropia normalizada (0=certeza, 1=confusão)
    final entropy = _normalizedEntropy(safeProbs);
    final confidence = 1.0 - entropy;

    if (!mounted) return;

    if (isTiled) {
      setState(() {
        final safeIdx = maxIdx >= 0 && maxIdx < labels.length ? maxIdx : 0;
        _prediction = labels[safeIdx];
        _confidence = confidence;
        for (var i = 0; i < labels.length; i++) {
          _classProbAverages[i] = i < safeProbs.length ? safeProbs[i] : 0.0;
        }
        _status =
            'Tempo Real: ${_buffer.length}/$windowSize amostras (${(_buffer.length * 100 ~/ windowSize)}%)...';
      });
      return;
    }

    final windowNumber = _windowPredictions.length + 1;
    var analysisFinished = false;

    setState(() {
      // Registra o resultado desta janela.
      final safeIdx = maxIdx >= 0 && maxIdx < labels.length ? maxIdx : 0;
      _windowPredictions.add(safeIdx);
      _windowConfidences.add(safeProbs[safeIdx]);
      _windowAnalysisRecords.add(
        _WindowAnalysisRecord(
          sequence: _currentSequence,
          windowNumber: windowNumber,
          predictedClassIndex: safeIdx,
          classProbabilities: List<double>.from(safeProbs),
          windowSamples: sourceWindow
              .map((s) => List<double>.from(s))
              .toList(),
          confidence: confidence,
        ),
      );

      // Atualiza médias de probabilidade por classe (mostradas na tela).
      final completedWindows = _windowPredictions.length;
      for (var i = 0; i < labels.length; i++) {
        _classProbSums[i] += safeProbs[i];
        _classProbAverages[i] = _classProbSums[i] / completedWindows;
      }

      // Atualiza confiança baseada em entropia
      _confidence = confidence;

      // Ainda há janelas a coletar: prepara a próxima com 50% de overlap.
      if (_currentWindow < _selectedTotalWindows) {
        _currentWindow++;
        while (_buffer.length > overlapSize) {
          _buffer.removeFirst();
        }
        _status =
            'Janela ${_currentWindow - 1}/$_selectedTotalWindows concluida. Coletando janela $_currentWindow/$_selectedTotalWindows com sobreposicao de 50%...';
        return;
      }

      // Todas as janelas processadas → votação majoritária.
      final votes = <int, int>{};
      for (final idx in _windowPredictions) {
        votes[idx] = (votes[idx] ?? 0) + 1;
      }

      int bestIdx = _windowPredictions.first;
      for (final entry in votes.entries) {
        final currentBestVotes = votes[bestIdx] ?? 0;
        if (entry.value > currentBestVotes) {
          bestIdx = entry.key;
          continue;
        }
        // Empate: desempata pela maior confiança média.
        if (entry.value == currentBestVotes && entry.key != bestIdx) {
          final confA = _averageConfidenceFor(entry.key);
          final confB = _averageConfidenceFor(bestIdx);
          if (confA > confB) {
            bestIdx = entry.key;
          }
        }
      }

      _prediction = labels[bestIdx];
      _isCollecting = false;
      _status =
          'Análise concluída com $_selectedTotalWindows janelas. Pressione "Iniciar" para nova análise.';
      analysisFinished = true;
    });

    if (analysisFinished) {
      _accelerometerSub?.cancel();
      WakelockPlus.disable();
      unawaited(_finishAnalysis());
    }
  }

  /// Confiança média das janelas que predisseram [labelIdx]; 0 se nenhuma.
  double _averageConfidenceFor(int labelIdx) {
    var sum = 0.0;
    var count = 0;
    for (var i = 0; i < _windowPredictions.length; i++) {
      if (_windowPredictions[i] == labelIdx) {
        sum += _windowConfidences[i];
        count++;
      }
    }
    return count == 0 ? 0.0 : sum / count;
  }

  // ===========================================================================
  // SECÇÃO 6 — Geração de CSV e exportação
  // ===========================================================================

  /// Constrói a string CSV com TODOS os registros brutos do acelerômetro.
  String _generateCsv() {
    final buffer = StringBuffer();
    buffer.writeln('Sequence,Data,Eixo X,Eixo Y,Eixo Z');
    for (final r in _accelRecords) {
      buffer.writeln(
        '${r.sequence},${r.dateTime.toIso8601String()},${r.x},${r.y},${r.z}',
      );
    }
    return buffer.toString();
  }

  /// Constrói a string CSV com a análise por janela (top-2, margem, entropia,
  /// e probabilidades de cada classe).
  String _generateAnalysisCsv() {
    final buffer = StringBuffer();
    final probabilityHeaders =
        labels.map((label) => 'Probabilidade $label').join(',');
    buffer.writeln(
      'Sequence,Janela,Classe Vencedora,Probabilidade Vencedora,Classe 2a,Probabilidade 2a,Margem Top1 Top2,Razao Top1 Top2,Entropia Normalizada,$probabilityHeaders',
    );

    for (final record in _windowAnalysisRecords) {
      // Garante uma linha de probabilidades com tamanho fixo = labels.length.
      final normalizedProbs = List<double>.generate(
        labels.length,
        (index) => index < record.classProbabilities.length
            ? record.classProbabilities[index]
            : 0.0,
      );

      // Ordena índices por probabilidade decrescente para obter top-1 e top-2.
      final rankedIndices = List<int>.generate(labels.length, (index) => index)
        ..sort((a, b) => normalizedProbs[b].compareTo(normalizedProbs[a]));

      final winnerIdx = rankedIndices.first;
      final secondIdx =
          rankedIndices.length > 1 ? rankedIndices[1] : rankedIndices.first;
      final winnerProb = normalizedProbs[winnerIdx];
      final secondProb = normalizedProbs[secondIdx];
      final marginTop1Top2 = winnerProb - secondProb;
      final ratioTop1Top2 = winnerProb / (secondProb + 1e-9);
      final entropy = _normalizedEntropy(normalizedProbs);

      final probs = labels.asMap().entries.map((entry) {
        final idx = entry.key;
        final prob = normalizedProbs[idx];
        return prob.toStringAsFixed(6);
      }).join(',');

      buffer.writeln(
        '${record.sequence},${record.windowNumber},${labels[winnerIdx]},${winnerProb.toStringAsFixed(6)},${labels[secondIdx]},${secondProb.toStringAsFixed(6)},${marginTop1Top2.toStringAsFixed(6)},${ratioTop1Top2.toStringAsFixed(6)},${entropy.toStringAsFixed(6)},$probs',
      );
    }
    return buffer.toString();
  }

  /// Calcula a entropia de Shannon normalizada (0..1) da distribuição [probs].
  /// 0 indica certeza total; 1 indica distribuição uniforme (máxima incerteza).
  double _normalizedEntropy(List<double> probs) {
    if (probs.isEmpty) return 0.0;
    const epsilon = 1e-12;
    var entropy = 0.0;
    for (final p in probs) {
      final safeP = p.clamp(epsilon, 1.0).toDouble();
      entropy += -safeP * math.log(safeP);
    }
    final maxEntropy = math.log(probs.length);
    if (maxEntropy <= 0.0) return 0.0;
    return (entropy / maxEntropy).clamp(0.0, 1.0);
  }

  /// Exporta os registros brutos do acelerômetro via diálogo de
  /// compartilhamento do sistema (não grava nada no dispositivo).
  Future<void> _saveRecordsToFile() async {
    try {
      final csv = _generateCsv();
      final tempFile = XFile.fromData(
        Uint8List.fromList(csv.codeUnits),
        name: 'acelerometro_${DateTime.now().millisecondsSinceEpoch}.csv',
        mimeType: 'text/csv',
      );
      await SharePlus.instance.share(
        ShareParams(
          files: [tempFile],
          text: 'Registros do acelerômetro',
          subject: 'Exportação CSV de atividade',
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Erro ao compartilhar: $e')),
      );
    }
  }

  /// Exporta a análise por janela (CSV) via diálogo de compartilhamento.
  Future<void> _saveAnalysisToFile() async {
    try {
      final csv = _generateAnalysisCsv();
      final tempFile = XFile.fromData(
        Uint8List.fromList(csv.codeUnits),
        name: 'janela_${DateTime.now().millisecondsSinceEpoch}.csv',
        mimeType: 'text/csv',
      );
      await SharePlus.instance.share(
        ShareParams(
          files: [tempFile],
          text: 'Análise por janela',
          subject: 'Exportação CSV de análise por janela',
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Erro ao compartilhar análise: $e')),
      );
    }
  }

  // ===========================================================================
  // SECÇÃO 7 — Helpers de pós-processamento
  // ===========================================================================

  /// Constrói uma janela de [windowSize] amostras a partir do buffer atual,
  /// repetindo as amostras ciclicamente quando ainda não há dados suficientes.
  List<List<double>> _buildTiledWindow() {
    final real = _buffer.toList();
    final tiled = <List<double>>[];
    var i = 0;
    while (tiled.length < windowSize) {
      tiled.add(List<double>.from(real[i % real.length]));
      i++;
    }
    return tiled;
  }

  /// Achata recursivamente um valor (número, lista de listas…) em uma lista
  /// plana de doubles. Útil para extrair as probabilidades do tensor de saída.
  List<double> _flattenToDoubleList(Object value) {
    if (value is num) return [value.toDouble()];
    if (value is List) {
      final out = <double>[];
      for (final item in value) {
        out.addAll(_flattenToDoubleList(item));
      }
      return out;
    }
    return const [];
  }

  /// Retorna os índices das classes ordenados por probabilidade média DESC.
  /// Usado para mostrar o ranking colorido na UI.
  List<int> _sortedClassIndicesByConfidence() {
    final indices = List<int>.generate(labels.length, (i) => i);
    indices.sort(
      (a, b) => _classProbAverages[b].compareTo(_classProbAverages[a]),
    );
    return indices;
  }

  // ===========================================================================
  // SECÇÃO 7.5 — Confirmação pós-aferição e envio ao servidor
  // ===========================================================================

  /// Toca o sinal de fim. Se servidor remoto estiver habilitado, exibe
  /// o diálogo de confirmação da atividade e envia os dados para a nuvem.
  Future<void> _finishAnalysis() async {
    await _notifyEnd();
    if (!mounted) return;

    if (!_useRemoteServer || _realTimeMode) {
      return;
    }

    final realClass = await _showRealClassDialog();
    if (!mounted) return;
    if (realClass == null) return;

    final isCorrect = realClass == _prediction;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          isCorrect
              ? '✓ Acertou! APP identificou corretamente.'
              : '✗ Errou. Atividade real: $realClass',
        ),
        backgroundColor: isCorrect ? Colors.green : Colors.redAccent,
        duration: const Duration(seconds: 3),
      ),
    );

    unawaited(_sendWindowsToBackend(realClass));
  }

  /// Exibe um diálogo pedindo ao usuário que confirme qual atividade estava
  /// realizando. Retorna a classe escolhida, ou null se o usuário pular.
  /// O diálogo é posicionado mais abaixo para não sobrepor dados importantes.
  Future<String?> _showRealClassDialog() {
    String selected =
        labels.contains(_prediction) ? _prediction : labels.first;

    return showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (dialogContext) {
        return Align(
          alignment: Alignment.bottomCenter,
          child: Padding(
            padding: const EdgeInsets.only(bottom: 80),
            child: StatefulBuilder(
              builder: (context, setModalState) {
                return AlertDialog(
                  title: const Text('Confirmar atividade'),
                  content: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'APP identificou: $_prediction',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 16),
                      const Text('Qual atividade você estava realizando?'),
                      const SizedBox(height: 8),
                      DropdownButton<String>(
                        value: selected,
                        isExpanded: true,
                        onChanged: (value) {
                          if (value != null) {
                            setModalState(() => selected = value);
                          }
                        },
                        items: labels
                            .map(
                              (l) => DropdownMenuItem(
                                value: l,
                                child: Text(l),
                              ),
                            )
                            .toList(),
                      ),
                    ],
                  ),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.of(dialogContext).pop(null),
                      child: const Text('Pular'),
                    ),
                    FilledButton(
                      onPressed: () => Navigator.of(dialogContext).pop(selected),
                      child: const Text('Confirmar'),
                    ),
                  ],
                );
              },
            ),
          ),
        );
      },
    );
  }

  /// Envia os dados de cada janela da aferição atual para o servidor remoto.
  Future<void> _sendWindowsToBackend(String realClass) async {
    final currentRecords = _windowAnalysisRecords
        .where((r) => r.sequence == _currentSequence)
        .toList();

    var sentCount = 0;
    for (final record in currentRecords) {
      final predictedClass = labels[record.predictedClassIndex];
      final topProb = record.classProbabilities[record.predictedClassIndex];
      final samples = record.windowSamples
          .map((s) => {'x': s[0], 'y': s[1], 'z': s[2]})
          .toList();

      final payload = {
        'device': 'flutter_ios',
        'user_name': _userNameForServer,
        'samples': samples,
        'confidence': record.confidence,
        'top_class_probability': topProb,
        'predicted_class': predictedClass,
        'real_class': realClass,
      };

      try {
        final response = await http.post(
          Uri.parse('$_backendUrl/v1/coletas'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload),
        );
        if (response.statusCode == 200 || response.statusCode == 201) {
          sentCount++;
        }
      } catch (_) {
        // Falha silenciosa — não interrompe o uso normal do app.
      }
    }

    if (!mounted) return;
    final total = currentRecords.length;
    final message = sentCount == total
        ? '☁ $sentCount janela(s) enviada(s) ao servidor.'
        : '⚠ Apenas $sentCount/$total janelas enviadas.';
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: sentCount == total ? null : Colors.orange,
      ),
    );
  }

  @override
  void dispose() {
    // Cancela timers, streams, libera o wakelock e fecha o interpretador.
    _countdownTimer?.cancel();
    _accelerometerSub?.cancel();
    WakelockPlus.disable();
    _interpreter?.close();
    super.dispose();
  }

  // ===========================================================================
  // SECÇÃO 8 — Construção da UI
  // ===========================================================================

  /// Painel inferior com seletor de número de janelas, botão de início,
  /// exportação e contadores informativos.
  Widget _buildControlButtons() {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(24, 10, 24, 20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text('Janelas:'),
                const SizedBox(width: 12),
                DropdownButton<int>(
                  value: _selectedTotalWindows,
                  onChanged: (_isCollecting || _isStartingSequence)
                      ? null
                      : (value) {
                          if (value == null) return;
                          setState(() {
                            _selectedTotalWindows = value;
                            _currentWindow = 1;
                            _buffer.clear();
                            _windowPredictions.clear();
                            _windowConfidences.clear();
                            for (var i = 0; i < labels.length; i++) {
                              _classProbSums[i] = 0.0;
                              _classProbAverages[i] = 0.0;
                            }
                            _prediction = '—';
                            _confidence = 0.0;
                            _status =
                                'Configuração atualizada. Pressione "Iniciar" para começar.';
                          });
                        },
                  items: List.generate(
                    10,
                    (index) => DropdownMenuItem<int>(
                      value: index + 1,
                      child: Text('${index + 1}'),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            Center(
              child: SizedBox(
                width: 220,
                child: ElevatedButton.icon(
                  onPressed: (_isCollecting || _isStartingSequence)
                      ? null
                      : _start,
                  icon: const Icon(Icons.play_arrow, size: 30),
                  label: const Text(
                    'Iniciar',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.w700),
                  ),
                  style: ElevatedButton.styleFrom(
                    minimumSize: const Size(220, 68),
                    backgroundColor: Colors.green,
                    disabledBackgroundColor: Colors.grey,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton.icon(
                  onPressed: _accelRecords.isEmpty ? null : _saveRecordsToFile,
                  icon: const Icon(Icons.download),
                  label: const Text('Exportar Dados'),
                ),
                const SizedBox(width: 10),
                ElevatedButton.icon(
                  onPressed: _windowAnalysisRecords.isEmpty ? null : _saveAnalysisToFile,
                  icon: const Icon(Icons.table_view),
                  label: const Text('Exportar Análise'),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              'Registros coletados: ${_accelRecords.length}',
              style: const TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 2),
            Text(
              'Janelas analisadas: ${_windowAnalysisRecords.length}',
              style: const TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 2),
            Text(
              'Usuario para servidor: $_userNameForServer',
              style: const TextStyle(color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }

  /// Pequena animação de "fogos de artifício" exibida no momento exato em
  /// que a coleta inicia (mensagem "INICIANDO!").
  Widget _buildFireworksOverlay() {
    const colors = [
      Colors.redAccent,
      Colors.amber,
      Colors.lightBlueAccent,
      Colors.greenAccent,
      Colors.pinkAccent,
      Colors.orangeAccent,
    ];

    return IgnorePointer(
      child: Stack(
        alignment: Alignment.center,
        children: List.generate(14, (index) {
          final angleBase = index * (math.pi / 7);
          final distance = 70.0 + (index % 4) * 18.0;
          final dx = math.cos(angleBase) * distance;
          final dy = math.sin(angleBase) * distance;

          return TweenAnimationBuilder<double>(
            tween: Tween<double>(begin: 0, end: 1),
            duration: Duration(milliseconds: 650 + (index % 4) * 90),
            curve: Curves.easeOutCubic,
            builder: (context, value, child) {
              final opacity = (1 - value).clamp(0.0, 1.0);
              return Transform.translate(
                offset: Offset(dx * value, dy * value),
                child: Opacity(opacity: opacity, child: child),
              );
            },
            child: Icon(
              Icons.auto_awesome,
              color: colors[index % colors.length],
              size: 18 + (index % 3) * 4,
            ),
          );
        }),
      ),
    );
  }

  /// Monta a estrutura completa da tela: AppBar, cabeçalho do projeto, área
  /// central com a predição e o ranking de classes, e o painel de controle.
  @override
  Widget build(BuildContext context) {
    final sortedIndices = _sortedClassIndicesByConfidence();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Identificador de atividade'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            onPressed: (_isCollecting || _isStartingSequence)
                ? null
                : _openSettingsDialog,
            tooltip: 'Settings',
            icon: const Icon(Icons.settings),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Center(
              child: Text(
                'v$appVersion',
                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          Container(
            width: double.infinity,
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  'Curso: Aplicações de Aprendizado de Máquina em Sistema Embarcados',
                  style: TextStyle(fontSize: 14, fontStyle: FontStyle.italic),
                ),
                SizedBox(height: 2),
                Text(
                  'Projeto: Reconhecimento de Atividade Humana (HAR) com Flutter',
                  style: TextStyle(fontSize: 14, fontStyle: FontStyle.italic),
                ),
                SizedBox(height: 2),
                Text(
                  'Aluno: Marcos Cabanas Esteves',
                  style: TextStyle(fontSize: 13, fontStyle: FontStyle.italic, color: Colors.grey),
                ),
              ],
            ),
          ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 20, 16, 12),
              child: Stack(
                alignment: Alignment.center,
                children: [
                  if (_isStartingSequence && _countdownText == 'INICIANDO!')
                    _buildFireworksOverlay(),
                  Column(
                    children: [
                      Text(
                        _isStartingSequence ? _countdownText : _prediction,
                        style: TextStyle(
                          fontSize: _isStartingSequence ? 60 : 48,
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 16),
                      if (!_isStartingSequence)
                        Text(
                          'Confiança: ${(_confidence * 100).toStringAsFixed(1)}%',
                          style: const TextStyle(fontSize: 24),
                        ),
                      const SizedBox(height: 28),
                      Text(
                        'Buffer: ${_buffer.length}/$windowSize amostras   |   Janela: $_currentWindow/$_selectedTotalWindows',
                        style: const TextStyle(color: Colors.grey),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 14),
                      Text(
                        _status,
                        style: const TextStyle(color: Colors.grey, fontSize: 12),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 12),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: List.generate(sortedIndices.length, (index) {
                          final classIdx = sortedIndices[index];
                          return Text(
                            '${labels[classIdx]}: ${(_classProbAverages[classIdx] * 100).toStringAsFixed(1)}%',
                            style: const TextStyle(color: Colors.grey, fontSize: 12),
                          );
                        }),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          _buildControlButtons(),
        ],
      ),
    );
  }
}

/// Registro bruto de uma leitura do acelerômetro (uma linha do CSV exportado).
class _AccelRecord {
  final int sequence;
  final DateTime dateTime;
  final double x;
  final double y;
  final double z;

  const _AccelRecord({
    required this.sequence,
    required this.dateTime,
    required this.x,
    required this.y,
    required this.z,
  });
}

/// Resultado completo da inferência de UMA janela: classe predita e o vetor
/// de probabilidades por classe (usado no CSV de análise).
class _WindowAnalysisRecord {
  final int sequence;
  final int windowNumber;
  final int predictedClassIndex;
  final List<double> classProbabilities;
  final List<List<double>> windowSamples;
  final double confidence;

  _WindowAnalysisRecord({
    required this.sequence,
    required this.windowNumber,
    required this.predictedClassIndex,
    required this.classProbabilities,
    required this.windowSamples,
    required this.confidence,
  });
}
