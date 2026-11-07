# 中国A股股票K线图动态展示
# powered with claude
这是个很适合用 Python 做的需求，核心思路是：逐帧绘制 K 线图 → 合成视频。下面给你一个完整可运行的方案。

整体流程
获取股票日K数据（开高低收、成交量）
用 mplfinance + matplotlib.animation 逐帧绘制：第 N 帧只显示前 N 个交易日
用 ffmpeg 导出为 mp4。

1. 安装依赖
bash
pip install akshare mplfinance matplotlib pandas
还需要安装 ffmpeg（导出 mp4 必需）
macOS: brew install ffmpeg
Windows: 官网下载后加入 PATH，或 choco install ffmpeg
Linux: sudo apt install ffmpeg

2. 完整代码
moutai_cinematic.py

3. 几个常用调整
想要的效果	怎么改
播放更慢/更快	改 FPS（越小越慢），或每隔几天取一帧：frames=range(0, TOTAL, 2)
从空白开始、图越来越长	WINDOW = None（此时一开始只有几根，建议 end 至少从 5 起）
固定Y轴不跳动	在 update 里加 ax_price.set_ylim(df.Low.min()*0.95, df.High.max()*1.05)
美股/港股数据	用 yfinance：yf.download("AAPL", start=..., end=...)
最后停留几秒	在 frames 末尾重复最后一帧，例如 frames=list(range(TOTAL)) + [TOTAL-1]*60
中文显示乱码	加 plt.rcParams["font.sans-serif"] = ["SimHei"]（Mac 用 "PingFang SC" 或 "Arial Unicode MS"）
4. 不想写代码的替代方案
Flourish / Datawrapper：有现成的时间序列动画，但对K线支持有限
TradingView 回放 + 录屏：用它的 Bar Replay 功能，配合 OBS 录屏，最简单但画面质量和可控性较差
Remotion（JS）：如果你想要更精美的动效（渐变、缩放、标注），可以用 Remotion 配合 ECharts 的 K 线图，上限更高但上手更难
小提示
数据量大（上千帧）时导出会比较慢，可先用 dpi=60、较少帧测试效果，再高清导出。
如果想加配音/背景音乐，导出后用 ffmpeg -i kline.mp4 -i music.mp3 -shortest out.mp4 合并即可。

如果你告诉我具体是哪只股票、哪个时间段，以及想要的风格（比如加均线、标注重大事件、竖屏短视频尺寸 9:16 等），我可以帮你直接把代码调好，甚至在这里帮你跑出来视频。

#后期修改：
已改好。现在图表会实时向左滚动，当前交易日的折线头固定在画面水平正中央。events.json 不变，替换 moutai_cinematic.py 后用原命令运行即可。

我用测试数据抽帧检查了版式，并跑通了多进程渲染和 mp4 输出，但没有整片播放看动态效果。

画面变化：

居中：折线头始终在 x=960，过去的走势向左流走，右半边留空，表示尚未发生的时间。
此刻线：头部位置有一条淡淡的竖向发光线。
月份刻度：画面里有竖向虚线刻度和"2月、3月"标签，随图表一起滚动，右侧能看到即将到来的月份。
左边缘：折线和光幕在左边缘渐隐。
纵轴：只跟随画面里可见的这段走势自动缩放，不再按全年范围压缩。
事件：走到事件那天，滚动停下，信息卡和光圈出现在头部旁边，结束后继续向左移动。
片尾：画面会拉远，整年走势铺满全屏，再显示统计和标注。

可调参数：

--px 16：每个交易日占的像素数，默认 16，画面里大约能看到最近 60 个交易日。调到 24 看得更细，调到 10 能看到更长的时间段。
--speed 8：向左滚动的快慢，数值越大滚动越快。
--fit：改回上一版"整段走势逐渐压缩、不滚动"的样式。

你说的"K线"，我仍按折线图实现。如果你想要真正的红绿蜡烛柱，随时告诉我，我可以改成蜡烛图版本。
