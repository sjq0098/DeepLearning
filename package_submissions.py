# -*- coding: utf-8 -*-
"""
按课程要求打包四个小实验的提交物：
  每个实验 = 报告(PDF) + 代码，打包成 zip（学号+姓名）。
  自动排除：数据集(data/)、checkpoint(*.pth)、缓存、压缩包、Overleaf 样式、作业模板(docx/pptx)。

用法：
  1) 在 Overleaf 重新编译四份报告，把最终 PDF 放到下面 PDF_CANDIDATES 能找到的位置
     （仓库根目录或对应 exp 文件夹均可）。
  2) python package_submissions.py
  3) 生成的 zip 在 submission/ 下，按"提交方式另行通知"再上交即可。
"""
import os, glob, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDENT = "2313119申健强"
OUT = os.path.join(ROOT, "submission")
os.makedirs(OUT, exist_ok=True)

# 每个实验：代码目录、图目录、报告 PDF 候选路径
EXPS = {
    "实验1_CNN":        dict(codedir="exp1",         imgdir="exp1/images",
                            pdf=["深度学习实验1.pdf", "exp1/深度学习实验1.pdf",
                                 "深度学习实验一CNN.pdf", "exp1/深度学习实验一CNN.pdf"]),
    "实验2_RNN":        dict(codedir="exp2",         imgdir="exp2/images",
                            pdf=["深度学习实验2.pdf", "exp2/深度学习实验2.pdf"]),
    "实验3_Attention":  dict(codedir="exp3/seq2seq", imgdir="exp3/images",
                            pdf=["深度学习实验3.pdf", "exp3/深度学习实验3.pdf"]),
    "实验4_GAN":        dict(codedir="exp4",         imgdir="exp4/images",
                            pdf=["深度学习实验4.pdf", "exp4/深度学习实验4.pdf"]),
}

# 代码：只收顶层 .py / .ipynb / .json（不递归进 data/）
CODE_EXT = (".py", ".ipynb", ".json")
MAX_MB = 40   # 跳过超大文件（如带大量输出的 notebook 可自行清输出后再打）

# 仅用于生成报告插图/做分析的脚本——不属于"实验代码"，提交时排除
EXCLUDE_SCRIPTS = {
    "exp1_figs.py", "exp1_gradcam.py",
    "exp2_figs.py", "exp2_tokens.py",
    "exp3_figs.py", "exp3_tokens.py", "exp3_compare.py",
    "exp4_extra.py", "exp4_quality.py", "exp4_combine.py",
    "mc_figs.py", "check_answer_collapse.py",
    "package_submissions.py",
}
# 保留的"额外实验"训练脚本(报告中有相应实验): exp3_dot.py, mc_train.py
INCLUDE_IMAGES = False   # 图已在报告 PDF 中，默认不再单独打包；改 True 可一并附上


def find_pdf(cands):
    for c in cands:
        p = os.path.join(ROOT, c)
        if os.path.exists(p):
            return p
    return None


def collect_code(codedir):
    files = []
    for f in sorted(os.listdir(os.path.join(ROOT, codedir))):
        full = os.path.join(ROOT, codedir, f)
        if not os.path.isfile(full):
            continue
        if not f.lower().endswith(CODE_EXT):
            continue
        if f in EXCLUDE_SCRIPTS:
            continue
        sz = os.path.getsize(full)
        if sz > MAX_MB * 1024 * 1024:
            print(f"   ! 跳过超大文件 {f} ({sz/1e6:.1f}MB) —— 建议清除 notebook 输出后再加入")
            continue
        files.append(full)
    return files


def main():
    print(f"学生：{STUDENT}\n输出目录：{OUT}\n")
    for tag, cfg in EXPS.items():
        zip_name = f"{STUDENT}_{tag}.zip"
        zip_path = os.path.join(OUT, zip_name)
        code = collect_code(cfg["codedir"])
        imgs = sorted(glob.glob(os.path.join(ROOT, cfg["imgdir"], "*.png"))) if INCLUDE_IMAGES else []
        pdf = find_pdf(cfg["pdf"])
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in code:
                z.write(f, arcname=f"{tag}/code/{os.path.basename(f)}")
            for f in imgs:
                z.write(f, arcname=f"{tag}/images/{os.path.basename(f)}")
            if pdf:
                z.write(pdf, arcname=f"{tag}/报告_{os.path.basename(pdf)}")
        mb = os.path.getsize(zip_path) / 1e6
        print(f"[{tag}]  -> {zip_name}  ({mb:.1f}MB)")
        print(f"   报告PDF : {'OK ' + os.path.basename(pdf) if pdf else '!! 未找到，请先放入最终 PDF'}")
        print(f"   代码    : {len(code)} 个文件   图: {len(imgs)} 张\n")
    print("完成。注意：报告需 ≤8 页、内容工整；具体提交方式以老师“另行通知”为准。")


if __name__ == "__main__":
    main()
