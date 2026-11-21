# mosaic-pipeline

動画内の指定した対象を自動で追跡してモザイクをかけるツールです。

- **SAM 2**（プロンプト可能セグメンテーション）で対象を切り取り、フレームをまたいで追跡
- 対象が画面外に出たら **YOLO（person検出）** で再捕捉
- 遮蔽や形の変化には途中フレームでの**再指定（リプロンプト）**で対応
- クリックで点を打てる **Gradio GUI** 付き

## 動作環境

- Windows 10/11
- Python 3.12、Git、FFmpeg
- NVIDIA GPU（CUDA）推奨。RTX 3080 で約 179 ms/frame（CPUだと大幅に遅くなります）

## クイックスタート

```powershell
.\setup_windows.ps1   # 環境構築（初回のみ）
.\start_app.bat       # GUIアプリを起動
```

CLIで使う場合や詳しい手順は `README_PC.md`、アプリの使い方は `README_APP.md` を見てください。

## 注意

- モデルファイル（SAM 2 / YOLO の重み）は容量が大きいためリポジトリには含めていません。`setup_windows.ps1` が自動でダウンロードします。
- 動画ファイル自体も含めていません。各自の環境でご用意ください。
