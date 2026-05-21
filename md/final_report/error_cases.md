# 错误案例分析 (50 samples × {K=1,3,5})

## 总览

| Method | K | total | n_FP | n_FN | n_parse_fail |
|---|---:|---:|---:|---:|---:|
| rag-bucketed | 1 | 50 | 15 | 1 | 0 |
| rag-bucketed | 3 | 50 | 16 | 0 | 1 |
| rag-bucketed | 5 | 50 | 15 | 0 | 0 |
| raw | 1 | 50 | 11 | 4 | 0 |
| raw | 3 | 50 | 18 | 1 | 0 |
| raw | 5 | 50 | 16 | 0 | 0 |
| semantic-neutral-fixed | 1 | 50 | 15 | 1 | 1 |
| semantic-neutral-fixed | 3 | 50 | 6 | 1 | 19 |
| semantic-neutral-fixed | 5 | 50 | 0 | 18 | 5 |
| semantic-risky-old | 1 | 50 | 21 | 0 | 1 |
| semantic-risky-old | 3 | 50 | 18 | 1 | 0 |
| semantic-risky-old | 5 | 50 | 18 | 0 | 0 |

## 代表案例：method=semantic-risky-old, K=5, kind=FP
- idx: 0
- true: B, pred: S
- active_feature_count: 45
- evidence: android.os.IBinder, android.os.Binder, bindService, onServiceConnected, transact, Ljava.lang.Class.forName, Ljava.lang.Class.getMethods, Ljava.lang.Class.getDeclaredField, Ljava.lang.Class.getField, Ljava.lang.Class.getCanonicalName
- explanation: The sample exhibits multiple indicators of malicious behavior, including extensive use of reflection (Ljava.lang.Class.*, Ljava.lang.Object.getClass), dynamic loading (ClassLoader, System.loadLibrary), and complex system interactions (Binder, ServiceConnection, transact, onBind, onServiceConnected). Furthermore, it collects sensitive device information (TelephonyManager.getNetworkOperator, android.content.pm.PackageInfo) and performs network communication (HttpGet.init, HttpPost.init, HttpUriRequest) while requesting high-privilege permissions like SYSTEM_ALERT_WINDOW, suggesting potential spyware or advanced malware functionality.
- invalid_evidence_inactive: ['Ljava.lang.Class.getField', 'Ljava.lang.Class.getCanonicalName']

## 代表案例：method=semantic-neutral-fixed, K=5, kind=FN
- idx: 7
- true: S, pred: B
- active_feature_count: 33
- evidence: 
- explanation: The presence of multiple network-related API calls (e.g., ClassLoader, URL, Socket) and the use of system-level functions suggest malicious activity, which is characteristic of a botnet or malware payload. The combination of these indicators points towards a malicious intent.

## 代表案例：method=rag-bucketed, K=5, kind=FP
- idx: 9
- true: B, pred: S
- active_feature_count: 39
- evidence: Binder, ClassLoader, HttpGet.init, HttpPost.init, HttpUriRequest, IBinder, Landroid.content.Context.registerReceiver, Landroid.content.Context.unregisterReceiver, Ljava.lang.Class.cast, Ljava.lang.Class.forName
- explanation: The sample exhibits multiple indicators of malicious behavior, including the use of reflection and dynamic loading (ClassLoader, System.loadLibrary, Ljava.lang.Class.* methods) combined with device identification (TelephonyManager.getDeviceId) and network communication (HttpGet.init, HttpPost.init). The combination of these features suggests unauthorized data collection or execution, aligning with patterns seen in malware. Rule [RULE_DEVICE_IDENTIFIER_COLLECTION] is relevant due to the presence of TelephonyManager.getDeviceId.
