# Line Digitizer

这个工具专门处理彩色 line 图，比如 LSV、CV、XRD 曲线图。它不需要 ChartRecover 的模型权重，核心逻辑是：

1. 你给出绘图区的像素边界。
2. 你给出 x/y 轴真实范围。
3. 你给出每条曲线的颜色。
4. 脚本按颜色把曲线像素提出来。
5. 脚本把像素坐标换成真实坐标，并输出 CSV。

## 安装

在项目根目录运行：

```bash
cd "/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/Data_extraction"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.venv"
python -m pip install -r line_digitizer/requirements.txt
```

## 典型用法

假设你的图叫 `lsv.png`：

```bash
python line_digitizer/line_digitizer.py init lsv.png \
  --out lsv_config.json \
  --x-min 1.2 --x-max 1.8 \
  --y-min -2 --y-max 50 \
  --x-step 0.002 \
  --series "blue:#232ddc:38" \
  --series "red:#dc2337:42" \
  --series "black:#1e1e1e:30"
```

然后打开 `lsv_config.json`，重点检查：

- `plot_area.left/right/top/bottom`：绘图区边框的像素位置。
- `axes.x.pixel_min/pixel_max`：x 轴最小/最大 tick 的像素位置。
- `axes.y.pixel_min/pixel_max`：y 轴最小/最大 tick 的像素位置。注意这不一定等于外框上下边。
- `ignore_regions`：图例、面板字母、标题等不想提取的区域。
- `series.rgb` 和 `tolerance`：每条线的颜色和容差。
- `keep_largest_components`：黑色/深色曲线常设为 `1`，可以过滤掉同色文字和图例；彩色曲线一般先不设，避免曲线低信号段被切掉。

## 自动检测 tick 线

更推荐的流程是先让脚本检测 tick 线，然后你手动填 tick 数值：

```bash
python line_digitizer/line_digitizer.py detect-ticks lsv.png \
  --config lsv_config.json \
  --out lsv_config_ticks.json \
  --preview lsv_ticks_preview.png
```

它会在 `lsv_config_ticks.json` 里写入类似内容：

```json
{
  "axes": {
    "x": {
      "scale": "linear",
      "calibration": [
        {"pixel": 106, "value": null},
        {"pixel": 221, "value": null},
        {"pixel": 336, "value": null},
        {"pixel": 451, "value": null},
        {"pixel": 566, "value": null},
        {"pixel": 682, "value": null}
      ]
    },
    "y": {
      "scale": "linear",
      "calibration": [
        {"pixel": 488, "value": null},
        {"pixel": 407, "value": null},
        {"pixel": 327, "value": null},
        {"pixel": 8, "value": null}
      ]
    }
  }
}
```

你只需要把 `value: null` 改成真实 tick 值：

```json
"x": {
  "scale": "linear",
  "calibration": [
    {"pixel": 106, "value": 1.0},
    {"pixel": 221, "value": 1.2},
    {"pixel": 336, "value": 1.4},
    {"pixel": 451, "value": 1.6},
    {"pixel": 566, "value": 1.8},
    {"pixel": 682, "value": 2.0}
  ]
}
```

如果检测到 minor tick，也可以删掉不需要的项，只保留你确定的 major tick。`lsv_ticks_preview.png` 会把检测到的 x/y tick 编号画出来，方便检查。

检测结果可能包含外框边线或 minor tick。原则是：保留你能确定数值的 tick，删掉不确定的候选。

可调参数：

```bash
--search-px 18          # 在轴线附近多宽的区域内找 tick
--dark-threshold 170    # 多暗才算轴线/tick 像素
--min-tick-len 5        # tick 至少多长
--max-cluster-width 14  # tick 候选线最多多粗；如果漏检，可以调大
```

如果坐标轴范围由 tick 决定，不要拿整个外框当范围。比如某张 LSV 图：

```text
x 轴：1.0 到 2.0 V
y 轴：下方 0 tick 到上方 24 tick
```

就应该这样写：

```json
{
  "axes": {
    "x": {"min": 1.0, "max": 2.0, "scale": "linear", "pixel_min": 106, "pixel_max": 682},
    "y": {"min": 0, "max": 24, "scale": "linear", "pixel_min": 488, "pixel_max": 8}
  }
}
```

这里 `pixel_min: 488` 是 y=0 tick 的 y 像素位置，`pixel_max: 8` 是 y=24 tick 的 y 像素位置。y 轴像素从上往下变大，所以 `pixel_min` 通常比 `pixel_max` 大，这是正常的。

也可以用多个 tick 做校准：

```json
{
  "axes": {
    "y": {
      "scale": "linear",
      "calibration": [
        {"pixel": 488, "value": 0},
        {"pixel": 407, "value": 4},
        {"pixel": 327, "value": 8},
        {"pixel": 8, "value": 24}
      ]
    }
  }
}
```

有 `calibration` 时，脚本会优先用这些 tick 点拟合坐标轴，`min/max/pixel_min/pixel_max` 可以不写。

提取：

```bash
python line_digitizer/line_digitizer.py extract lsv.png \
  --config lsv_config.json \
  --out lsv_points.csv \
  --preview lsv_preview.png \
  --debug-dir debug_masks
```

输出的 CSV 格式：

```csv
series,x,y,pixel_x,pixel_y
blue,1.200,0.12,250.0,1118.4
blue,1.202,0.13,253.9,1118.2
red,1.200,0.40,250.0,1112.1
```

`preview` 图会把提取到的点画回原图，用来检查是否误提取了文字、图例或坐标轴。

## 批量 UI

处理几十到上百张图时，不建议手改 JSON。启动本地 UI：

```bash
streamlit run line_digitizer/batch_ui.py
```

默认会读取：

```text
/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/Data_extraction/LSVs/Original
```

实际使用时，在左侧改成你的图片文件夹，例如：

```text
/path/to/lsv_images
```

UI 会为每张图片保存独立文件：

```text
line_digitizer/batch_outputs/
  configs/      # 每张图的配置
  csv/          # 导出的曲线数据
  previews/     # tick 预览和曲线 overlay
  debug_masks/  # 每条曲线的颜色分割 mask
```

推荐操作顺序：

```text
1. 选择一张图
2. 在左侧输入 x/y min、x/y max，以及 x/y major tick count；也可以先点 Auto read axis ranges 让 OCR 给建议值
3. 在 Plot area 模式确认橙色框覆盖绘图区，必要时点 Re-detect plot area
4. 点击 Auto detect ticks，界面会自动切到 Ticks
5. 看绿色 tick 是否压在真实主刻度上
6. 如有整体偏移，在 Ticks 模式拖动图上的 tick overlay，然后点 Apply movement
7. 如果整体移动还不够准，打开 Advanced: click exact ticks 手动点击主刻度
8. 需要时在 Advanced: curve colors 里检测或调整颜色
9. 点击 Confirm ticks + preview CSV
10. 界面会自动切到 Points，看提取点是否压在原曲线上
11. 点 Next
```

`Auto read axis ranges` 用本地 Tesseract OCR 读取坐标轴 tick label 数字，只会生成 suggested values，不会自动覆盖当前设置。确认建议值正确后，点 `Apply x`、`Apply y` 或 `Apply all`，再继续 `Auto detect ticks`。

OCR 依赖两部分：

```bash
python -m pip install pytesseract
conda install -p ./.venv -c conda-forge tesseract -y
```

如果你不用 conda，也可以安装系统包：

```bash
sudo apt install tesseract-ocr
```

OCR 不可用时 UI 仍然能正常打开，只是 `Auto read axis ranges` 会提示安装方法。OCR 结果里 `pass/review/fail` 只表示识别质量；因为数据必须 100% 人工确认，所以它永远只是建议值。

批量自动处理：

```text
1. 左侧确认 Image folder 和 Output folder
2. 左侧确认 x/y min、x/y max、x/y major tick count
3. 点击 Auto process folder
4. 查看 Output folder 下的 batch_qc.csv
5. 用 QC filter 切到 Needs verification
6. 逐张人工检查 tick overlay，尤其是 y=0 和 y=max
7. 确认无误后点击 Confirm ticks + preview CSV
```

`Auto process folder` 会对文件夹内所有图片自动做 plot area、tick、QC、颜色、CSV 和 overlay 输出。已有 config 会尽量保留你之前选好的曲线颜色；新图会自动检测颜色候选。

`batch_qc.csv` 字段包括：

```text
image,status,auto_status,verified,usable,verified_at,y_label_assisted,score,x_tick_count,y_tick_count,max_residual_px,warnings,points,csv_path,overlay_path,tick_overlay_path,config_path
```

状态含义：

```text
needs_verification  自动候选已生成，但还不能用于分析
verified            你已人工检查 tick overlay，usable=true
fail                自动候选明显失败，需要用 Advanced: click exact ticks 修正
```

`auto_status` 和 `score` 只用于排序和提醒，不表示数据已经可用。只有 `usable=true` 的行才能进入后续分析；如果 y 轴没有人工核查，不要使用该 CSV。

tick overlay 颜色含义：绿色是最终用于校准的主刻度，灰色是检测到但未采用的候选 tick；如果某个轴失败，高风险校准 tick 会显示为红色。右上角会显示当前图片的 verification status、auto score 和主要 warning。

主图上方只有三种预览模式：

```text
Plot area  只看绘图区框
Ticks      看坐标校准 tick，也可以拖动整组 x/y tick 做微调
Points     看已经导出的曲线点 overlay
```

主预览会居中显示并自动缩放完整图片，Ticks 模式不会再只显示局部画布。Points 模式的 overlay 点半径默认是 6 px，方便肉眼检查提取点是否压在原曲线上。

`Confirm ticks + preview CSV` 是人工确认动作。它会保存校准、标记 `usable=true`、导出 CSV、生成 point overlay，并切到 `Points` 模式。

如果 y 轴 tick 线一直识别不准，可以在 `Advanced tick detection` 里打开默认开启的 `Use y tick labels to assist y-axis`。它不是 OCR，不会自动读出数字内容；它利用你已经输入的 `y min/y max/y major tick count`，在 y 轴左侧找数字标签所在的水平行。如果找到的 label 行数量正好等于 y major tick count，就用这些行来辅助 y 轴校准。tick overlay 里短的青色横线表示检测到的 y label 行。

这个辅助功能能避开很多短 tick 线、minor tick、虚线参考线和 y 轴外框干扰，但它仍然必须人工确认。少数图片中，数字标签的视觉中心可能和真实 tick 线有 1-2 px 偏移；如果 overlay 中青色/绿色位置没有压中真实主刻度，关掉这个开关，或者用 `Advanced: click exact ticks` 点击 y 轴主刻度。

左侧的 `CSV x interval` 是输出采样间隔，不是坐标轴 tick 间隔。填 `0` 表示按图像像素列输出原始点；填 `0.002` 这类值表示把曲线插值成固定 x 间隔，方便后续比较不同图。

正常情况下不用手输 `left/right/top/bottom` 这类像素边界。它们只放在 `Advanced pixel crop` 里，作用是当自动绘图区识别明显错了时，手动改图像裁剪范围。坐标轴的真实数值仍然以左侧的 x/y min/max 和 tick 表格里的 value 为准。

如果 `Auto detect ticks` 失败：

```text
1. 先确认左侧 x/y major tick count 填的是主刻度数量，不要把 minor tick 算进去
2. 切到 Plot area，确认橙色框覆盖的是绘图区
3. 打开 Advanced tick detection，适当调大 search px，或降低/提高 dark threshold 后重试
4. 如果只是整组 y tick 上下偏移或 x tick 左右偏移，在 Ticks 模式拖动 overlay 后点 Apply movement
5. 还不行就在 Advanced: fallback actions 里点 Use evenly spaced tick refs
6. 如果等距校准也不准，打开 Advanced: click exact ticks，在图上直接点主刻度
```

`Use evenly spaced tick refs` 适合 tick 本来就是等间距、且绘图区边界和首尾主刻度比较接近的图。LSV 图如果 y=0 或 y=max 不在外框边界上，先用 `Advanced pixel crop` 把 crop top/bottom 调到真实首尾主刻度附近，再用这个兜底按钮。

`Advanced: click exact ticks` 不需要你知道像素点。选择 `x axis` 或 `y axis`，按主刻度顺序在图上点击，点满左侧设置的 major tick count 后，点击 `Apply clicked x` 或 `Apply clicked y`。UI 会自动把点击位置转成 pixel ref、写入 tick 表格，并生成 tick overlay。点错可以 `Undo last pick` 或 `Clear current axis`。

当前 tick 检测不需要 machine learning。UI 会先用传统图像处理找轴边附近的短竖线/短横线，再根据你输入的主刻度数量，选出最接近等间距、强度更高、跨度更合理的一组 tick。这个方法对 minor tick、外框线和黑色文字的干扰更稳，而且能在 CPU 上直接跑。

坐标轴范围读取也不需要 GPT 或 deep learning。现在用本地 Tesseract OCR 读 tick label，再用位置单调性和等间距检查过滤干扰。优点是 CPU 可跑、不上传图片、成本低；缺点是低分辨率或文字被遮挡时仍然需要人工确认。

只有在下面这种情况才值得考虑 deep learning：

```text
- 图片来源非常杂，坐标轴样式差异特别大
- tick 被曲线、文字或低分辨率压得很严重
- Tesseract 对你的图片来源经常误读 tick label 数字
- 你愿意准备一批人工标注的 tick/axis 数据来训练模型
```

准确性判断主要看三件事：

- `tick overlay`：校准 tick 是否压在真实主刻度上，特别是 y=0 和 y=max 这两个点。
- `point overlay`：提取点是否压在原曲线上，是否误提取了图例、文字或坐标轴。
- `QC`：多个 tick 拟合后的 pixel residual 是否很小。通常 `1-2 px` 比较理想。
- `csv/`：抽查关键点，比如 `J=10 mA/cm2` 对应的 potential 是否和图上读数一致。

## LSV 图的注意点

- 如果红线冲出 y 轴上限，只能提取图框内的部分。
- 黑色曲线容易和坐标轴、文字混在一起，需要用 `ignore_regions` 排除图例和文字。
- 原图分辨率越高，提取越准。
- 这个工具恢复的是图片里的近似曲线，不等于仪器导出的原始数据。
