# 動画モザイク処理 ― Windows PC 版 かんたん手順書

動画の中の「指定した物体・部分」だけに自動でモザイクをかけるツールです。
最初のフレームで「ここ」とクリック座標（点）か四角（枠）を指定すると、AI（SAM 2）がその対象を追いかけて、毎フレームにモザイクをかけます。音声はそのままコピーされます。

NVIDIA のグラフィックボード（GPU）がある PC では、自動的に GPU を使って高速に処理します。GPU がなくても動きますが、かなり遅くなります（後述の「速度の目安」を参照）。

---

## 0. 用意するもの

- Windows 10 / 11（64bit）
- インターネット接続（初回セットアップで合計 約 3〜5 GB ダウンロードします）
- 空きディスク 約 10 GB
- （あれば）NVIDIA の GPU と、最新の NVIDIA ドライバー

> **フォルダーの置き場所について**
> このフォルダー（`mosaic.py` が入っているフォルダー）は、`C:\mosaic-pipeline` のような**短くて単純な場所**に置くのがおすすめです。
> 「ドキュメント」「デスクトップ」など **OneDrive で同期されている場所は避けてください**（処理中に大量の一時画像をこのフォルダー内に作るため、同期で非常に遅くなります）。

---

## 1. 事前準備（最初の 1 回だけ）

### 1-1. Python をインストール

1. https://www.python.org/downloads/windows/ を開き、**Python 3.12**（または 3.11）の「Windows installer (64-bit)」をダウンロードします。
2. インストーラーの最初の画面で、下にある **「Add python.exe to PATH」に必ずチェック** を入れてから「Install Now」を押します。
   - ※ Microsoft Store 版の Python ではなく、python.org 版を使ってください。

### 1-2. Git をインストール

https://git-scm.com/download/win からインストーラーをダウンロードし、設定はすべて初期値のまま「Next」で進めて構いません。

### 1-3. ffmpeg をインストール

スタートメニューで「PowerShell」と入力して **Windows PowerShell** を開き、次の 1 行を貼り付けて Enter を押します。

```powershell
winget install Gyan.FFmpeg
```

インストールが終わったら、**PowerShell をいったん閉じて、開き直してください**（これをしないと ffmpeg が見つかりません）。開き直した PowerShell で次を実行し、バージョン情報が表示されれば OK です。

```powershell
ffmpeg -version
```

> Git も winget でインストールできます：`winget install Git.Git`

### 1-4. （GPU がある場合）NVIDIA ドライバーを最新に

https://www.nvidia.com/Download/index.aspx から最新ドライバーを入れておくと確実です。PowerShell で `nvidia-smi` と入力して、GPU の名前と「CUDA Version」が表示されれば OK です。
（CUDA Toolkit などの開発ツールを別途入れる必要はありません。）

---

## 2. セットアップスクリプトを実行する

1. エクスプローラーでこのフォルダー（`mosaic.py` がある場所）を開きます。
2. 上のアドレスバーをクリックし、`powershell` と入力して Enter を押すと、このフォルダーで PowerShell が開きます。
3. 次の 1 行を貼り付けて Enter を押します。

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

スクリプトが自動で次のことを行います（10〜30 分ほどかかります）。

- 専用の Python 環境（`.venv` フォルダー）を作る
- GPU があるかを調べて、GPU 用または CPU 用の PyTorch を入れる
- ultralytics（YOLO）、opencv-python を入れる
- SAM 2 を `third_party\sam2` にダウンロードしてインストールする
- AI モデル（`sam2.1_hiera_tiny.pt`、`sam2.1_hiera_t.yaml`、`yolo11n.pt`）を `models\` にダウンロードする
- ffmpeg が使えるかを確認する

最後に `Setup finished.` と、次に実行するコマンドが表示されれば完了です。

- 途中で「CUDA extension build failed -> retrying with SAM2_BUILD_CUDA=0」と黄色で表示されることがありますが、**問題ありません**（任意の追加機能が入らないだけで、結果にはほぼ影響しません）。
- 失敗したときや途中で止まったときは、**もう一度同じコマンドを実行**してください。済んだ部分は飛ばして続きから進みます。
- GPU があっても CPU 版にしたい場合：`powershell -ExecutionPolicy Bypass -File setup_windows.ps1 -Cpu`

### GPU が使えるかの確認

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

`True` なら GPU が使われます。`False` なら CPU で動きます（GPU があるのに False の場合は「6. 困ったとき」を参照）。

> 以降のコマンドはすべて、このフォルダーで開いた PowerShell で実行します。
> `.\.venv\Scripts\python.exe` は「このツール専用の Python」という意味です。

---

## 3. テスト動画で動作確認する

まず、付属のテスト動画（ボールが動いて画面外に出て、また戻ってくる 4 秒の動画）で試します。

```powershell
.\.venv\Scripts\python.exe make_test_clip.py test\test_clip.mp4
.\.venv\Scripts\python.exe mosaic.py test\test_clip.mp4 test\out_test.mp4 --frame 0 --point "96,180" --detect-reacquire --detect-classes "sports ball" --save-masks test\out_test.masks.npz
```

- 最初に `[mosaic] device: cuda`（GPU）または `device: cpu` と表示されます。
- 途中で `tracking lost @32` → `tracking resumed @70` のように表示されれば、「画面外に出た対象を見失い、戻ってきたときに再発見した」という意味で、正常です。
- 終わると `test\out_test.mp4` ができます。再生して、ボールにだけモザイクがかかっていることを確認してください。

（任意）正解データとの自動照合：

```powershell
.\.venv\Scripts\python.exe verify_output.py test\test_clip.mp4 test\out_test.mp4 test\test_clip.gt.npz test\out_test.masks.npz --report test\out_test.verify.json --sheet test\out_test.sheet.png
```

`test\out_test.sheet.png` に「元動画 | 処理後」の比較画像が保存されます。

---

## 4. 自分の動画を処理する

### 4-1. まず座標を調べる

モザイクをかけたい対象を「どのフレームの、どの位置か」で指定します。座標は **動画の元の解像度でのピクセル位置**（左上が 0,0、右へ X、下へ Y）です。

1. 対象がはっきり写っているフレームを画像として書き出します（例：0 フレーム目）。

   ```powershell
   ffmpeg -i "D:\video\input.mp4" -vf "select=eq(n\,0)" -frames:v 1 frame0.png
   ```

   別のフレームを使うときは `n\,0` の `0` をフレーム番号に変えます（フレーム番号 ＝ 秒数 × フレームレート。例：30fps の動画の 4 秒目 → 120）。
2. `frame0.png` を **ペイント** で開きます（表示倍率 100% のまま）。対象の上にマウスを置くと、画面左下に `640, 320px` のように座標が表示されます。これを使います。
   - 「点」で指定する場合：対象の中心あたりの 1 点（必要なら複数点）
   - 「枠」で指定する場合：対象を囲む四角の 左上 (X1,Y1) と 右下 (X2,Y2)

### 4-2. 実行する

```powershell
.\.venv\Scripts\python.exe mosaic.py "D:\video\input.mp4" "D:\video\output.mp4" --frame 0 --point "640,320"
```

- 入力・出力のパスに空白や日本語が入る場合は `" "` で囲んでください。
- 数字の組（`"640,320"` など）も `" "` で囲んでおくと安全です。
- **最初は `--max-seconds 3` を付けて、最初の 3 秒だけで試す** のがおすすめです。うまくいったら外して全体を処理します。

### 4-3. よく使うオプション

| オプション | 意味 | 例 |
|---|---|---|
| `--frame N` | 最初の指定をするフレーム番号（既定 0）。途中から指定した場合、前後両方向に追跡します | `--frame 120` |
| `--point "X,Y"` | 対象の上の点。複数回書けます | `--point "640,320" --point "660,380"` |
| `--neg-point "X,Y"` | 「ここは対象ではない」点（はみ出しを防ぐ） | `--neg-point "700,500"` |
| `--box "X1,Y1,X2,Y2"` | 対象を囲む四角。点より安定することが多いです | `--box "600,280,700,400"` |
| `--reprompt "F:X,Y"` | フレーム F で点を追加指定。**対象の形・向きが大きく変わる場面**や、追跡がずれ始めた場面で使います。`"F:neg:X,Y"`（除外点）、`"F:X1,Y1,X2,Y2"`（枠）も可。複数回書けます | `--reprompt "240:410,150"` |
| `--detect-reacquire` | 対象が画面外に出たり隠れたりして見失ったとき、人物検出（YOLO）で探し直して自動で追跡を再開します | `--detect-reacquire` |
| `--detect-classes` | 探し直しに使う検出の種類（既定 `person`＝人物）。人の体の一部が対象なら `person` のままで OK | `--detect-classes "person"` |
| `--block N` | モザイクのマス目の大きさ（ピクセル、既定 12）。大きいほど粗いモザイクになります | `--block 20` |
| `--dilate N` | マスクを外側に広げる幅（ピクセル、既定 6）。輪郭がはみ出して見える場合は大きくします | `--dilate 12` |
| `--max-seconds S` | 最初の S 秒だけ処理（お試し用） | `--max-seconds 3` |
| `--device cpu/cuda` | 計算に使う装置。普段は指定不要（GPU があれば自動で cuda） | `--device cpu` |

例（30 フレーム目で枠指定、240 フレーム目で形が変わるので点を追加、見失い対策あり、粗めのモザイク）：

```powershell
.\.venv\Scripts\python.exe mosaic.py in.mp4 out.mp4 --frame 30 --box "100,80,220,200" --reprompt "240:410,150" --detect-reacquire --block 16
```

うまくいかないときのコツ：

- モザイクが対象の一部にしかかからない → 点を増やす、または `--box` を使う
- 背景まではみ出す → `--neg-point` ではみ出した場所を指定する
- 途中でずれる・消える → ずれ始めたフレームを 4-1 の方法で書き出して座標を調べ、`--reprompt` で指定する

---

## 5. 速度の目安（CPU と GPU）

| 環境 | 1 フレームあたり | 1 分の動画（30fps＝1,800 フレーム） |
|---|---|---|
| CPU のみ（開発用の仮想マシンでの実測、640×360） | 約 13〜20 秒 | 約 7〜10 時間 |
| 一般的なデスクトップ CPU | 数秒程度（目安） | 数時間 |
| NVIDIA GPU（RTX クラス） | 0.1 秒前後〜（目安） | 数分〜十数分 |

- GPU の数値は目安で、まだ実機では測っていません。解像度が高いほど、また対象を見失って探し直す回数が多いほど遅くなります。
- CPU で処理する場合は、まず短い動画（数秒）で試してください。
- 処理の最後に `tracking done: ... (xxx ms/frame)` と 1 フレームあたりの時間が表示されます。
- GPU（RTX 30 シリーズ以降）では `--bf16` を付けると、さらに速くなる場合があります（結果がわずかに変わることがあります）。**CPU では `--bf16` を付けないでください**（多くの CPU では逆に遅くなります）。

---

## 6. 困ったとき

| 症状 | 原因と対処 |
|---|---|
| `このシステムではスクリプトの実行が無効になっている…` と出る | `.\setup_windows.ps1` ではなく、`powershell -ExecutionPolicy Bypass -File setup_windows.ps1` の形で実行してください |
| `Python 3.11 or 3.12 not found` | python.org 版 Python 3.12 を「Add python.exe to PATH」にチェックを入れてインストールし、**PowerShell を開き直して**から再実行 |
| `Git not found` | Git をインストール（`winget install Git.Git`）し、PowerShell を開き直して再実行 |
| ダウンロードやインストールの途中でエラー・停止 | 通信の一時的な問題のことが多いので、同じコマンドをもう一度実行（続きから進みます） |
| `CUDA extension build failed` と黄色で出る | 問題ありません。自動で拡張機能なしの方法でインストールし直します |
| GPU があるのに `torch.cuda.is_available()` が `False` / `device: cpu` になる | ① NVIDIA ドライバーを最新にして PC を再起動 ② `nvidia-smi` が動くか確認 ③ `.venv` フォルダーを削除してセットアップを再実行。それでもだめなら `-CudaIndex cu126` を付けて再実行 |
| `--device cuda requested but torch.cuda.is_available() is False` | 上と同じ（GPU 版 PyTorch が使えていません）。`--device` を外せば CPU で動きます |
| `ffmpeg not found on PATH` / `ffprobe not found on PATH` | `winget install Gyan.FFmpeg` を実行し、**PowerShell を開き直す**。`ffmpeg -version` で確認 |
| `initial prompt produced an empty mask` | 座標が対象から外れています。4-1 の方法で座標を確認し直す（動画の元の解像度の座標か確認） |
| GPU のメモリ不足（`CUDA out of memory`） | 他のアプリ（ゲーム等）を閉じる。改善しなければ `--device cpu` |
| 処理がとても遅い | `device: cpu` になっていないか確認。CPU の場合は `--max-seconds` で短く区切るか、解像度を下げた動画で処理 |
