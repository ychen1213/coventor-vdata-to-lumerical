"""Small standard-library Windows front end for independent VDATA conversion."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


ROOT = Path(__file__).resolve().parent


def conversion_python():
    executable = Path(sys.executable)
    console = executable.with_name("python.exe")
    return str(console if executable.name.lower() == "pythonw.exe" and console.is_file() else executable)


def make_command(input_path, output_path, material_map=None, verify=True, resume=False, reduction=0.):
    script = "convert_reduced.py" if reduction > 0 else "convert_vdata.py"
    command = [conversion_python(), str(ROOT / script), str(input_path), "--output", str(output_path)]
    if reduction > 0:
        command += ["--reduction", str(reduction)]
    if material_map:
        command += ["--material-map", str(material_map)]
    if verify:
        command += ["--verify-fdtd"]
    if resume:
        command += ["--resume"]
    return command


class ConverterApp:
    def __init__(self, window, initial_input=None):
        self.window = window
        self.window.title("Coventor VDATA → STL / FDTD")
        self.window.geometry("900x650")
        self.window.minsize(740, 550)
        self.events = queue.Queue()
        self.running = False
        self.started_at = None
        self.last_output = None
        self.log_file = None
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.map_var = tk.StringVar()
        self.verify_var = tk.BooleanVar(value=True)
        self.resume_var = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value="不减面")
        self.custom_reduction_var = tk.StringVar(value="45")
        self.status_var = tk.StringVar(value="选择独立的 Coventor VDATA 文件，然后开始转换。")
        self._build_widgets()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.after(120, self.poll_events)
        if initial_input:
            self.set_input(initial_input)

    def _build_widgets(self):
        frame = ttk.Frame(self.window, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(9, weight=1)
        ttk.Label(frame, text="独立 VDATA 文件转换", font=("Microsoft YaHei UI", 16)).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(frame, text="读取 .vdata 本身；不需要原始 ZAM、VWAF、GDS 或工艺工程。").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 14))
        fields = [("输入 VDATA", self.input_var, self.choose_input),
                  ("输出目录", self.output_var, self.choose_output),
                  ("材料映射 JSON（可选）", self.map_var, self.choose_mapping)]
        self.input_controls = []
        for row, (label, variable, callback) in enumerate(fields, start=2):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
            entry = ttk.Entry(frame, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", pady=5)
            button = ttk.Button(frame, text="浏览…", command=callback)
            button.grid(row=row, column=2, padx=(10, 0), pady=5)
            self.input_controls.extend([entry, button])
        options = ttk.Frame(frame)
        options.grid(row=5, column=0, columnspan=3, sticky="w", pady=(9, 5))
        modes = ttk.Frame(options)
        modes.pack(anchor="w", pady=(0, 6))
        ttk.Label(modes, text="转换模式：").pack(side="left")
        mode = ttk.Combobox(modes, textvariable=self.mode_var, state="readonly", width=16,
                            values=("不减面", "减面45%", "减面65%", "自定义比例"))
        mode.pack(side="left")
        ttk.Label(modes, text="  自定义减面百分比：").pack(side="left")
        custom = ttk.Entry(modes, textvariable=self.custom_reduction_var, width=7)
        custom.pack(side="left")
        ttk.Label(modes, text="%（仅自定义模式；当前QEM限单材料）").pack(side="left")
        self.input_controls.extend([mode, custom])
        verify = ttk.Checkbutton(options, text="验证并生成 FDTD 工程（需要 Lumerical）", variable=self.verify_var)
        verify.pack(anchor="w")
        resume = ttk.Checkbutton(options, text="复用本目录已完成的几何（--resume；补充材料映射时使用）", variable=self.resume_var)
        resume.pack(anchor="w", pady=(5, 0))
        self.input_controls.extend([verify, resume])
        actions = ttk.Frame(frame)
        actions.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 8))
        self.start_button = ttk.Button(actions, text="开始转换", command=self.start)
        self.start_button.pack(side="left")
        self.open_button = ttk.Button(actions, text="打开输出目录", command=self.open_output, state="disabled")
        self.open_button.pack(side="left", padx=(10, 0))
        self.mapping_button = ttk.Button(actions, text="编辑材料映射模板", command=self.open_mapping, state="disabled")
        self.mapping_button.pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="使用说明", command=self.open_help).pack(side="right")
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(frame, textvariable=self.status_var, wraplength=830).grid(row=8, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.log = ScrolledText(frame, height=15, wrap="word", state="disabled", font=("Consolas", 10))
        self.log.grid(row=9, column=0, columnspan=3, sticky="nsew")
        ttk.Label(frame, text="导出曲面仍是近似；本工具不运行光学时域求解。未知材料需要显式映射。").grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def set_input(self, path):
        path = Path(path).expanduser().resolve()
        self.input_var.set(str(path))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_var.set(str(path.with_name(path.stem + "_converted_" + stamp)))
        self.resume_var.set(False)

    def choose_input(self):
        path = filedialog.askopenfilename(title="选择独立 VDATA 文件", filetypes=[("Coventor VDATA", "*.vdata"), ("所有文件", "*.*")])
        if path:
            self.set_input(path)

    def choose_output(self):
        path = filedialog.askdirectory(title="选择输出目录：首次转换应为空目录，或填写新的目录名")
        if path:
            self.output_var.set(path)

    def choose_mapping(self):
        path = filedialog.askopenfilename(title="选择光学材料映射 JSON", filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
        if path:
            self.map_var.set(path)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.running:
            return
        try:
            if not self.input_var.get().strip():
                raise ValueError("请先选择输入 VDATA。")
            source = Path(self.input_var.get().strip()).expanduser().resolve()
            if not source.is_file():
                raise ValueError("输入文件不存在。")
            if not self.output_var.get().strip():
                raise ValueError("请指定输出目录。")
            output = Path(self.output_var.get().strip()).expanduser().resolve()
            if output.exists() and not output.is_dir():
                raise ValueError("输出必须是目录。")
            if output.exists() and any(output.iterdir()) and not self.resume_var.get():
                raise ValueError("输出目录已有文件。请选择新目录；补充材料映射时可勾选复用已完成的几何。")
            mapping = self.map_var.get().strip()
            if mapping and not Path(mapping).is_file():
                raise ValueError("材料映射 JSON 不存在。")
            reduction = {"不减面": 0., "减面45%": .45, "减面65%": .65}.get(self.mode_var.get())
            if reduction is None:
                reduction = float(self.custom_reduction_var.get()) / 100.
                if not math.isfinite(reduction) or not 0 < reduction < 1:
                    raise ValueError("自定义减面百分比必须大于0且小于100。")
            command = make_command(source, output, mapping, self.verify_var.get(), self.resume_var.get(), reduction)
        except Exception as exc:
            messagebox.showerror("无法开始", str(exc), parent=self.window)
            return
        self.last_output = output
        self.running = True
        self.started_at = time.monotonic()
        self.status_var.set("正在转换：独立读取 VDATA、提取材料曲面。")
        self.start_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.mapping_button.configure(state="disabled")
        for widget in self.input_controls:
            widget.configure(state="disabled")
        self.progress.start(12)
        self.append_log("\n命令：" + subprocess.list2cmdline(command) + "\n")
        threading.Thread(target=self._run, args=(command,), daemon=True).start()

    def _run(self, command):
        # A file, rather than an inherited pipe, lets an already-running
        # conversion finish even when the user closes this front end.
        try:
            descriptor, path = tempfile.mkstemp(prefix="coventor_vdata_gui_", suffix=".log")
            os.close(descriptor)
            self.events.put(("log_path", path))
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            with open(path, "wb") as capture:
                process = subprocess.Popen(command, stdout=capture, stderr=subprocess.STDOUT,
                                           cwd=str(ROOT), env=env, shell=False, creationflags=flags)
                with open(path, "r", encoding="utf-8", errors="replace") as reader:
                    while process.poll() is None:
                        chunk = reader.read()
                        if chunk:
                            self.events.put(("log", chunk))
                        time.sleep(.2)
                    chunk = reader.read()
                    if chunk:
                        self.events.put(("log", chunk))
                self.events.put(("finished", process.returncode))
        except Exception as exc:
            self.events.put(("log", "启动失败：" + str(exc) + "\n"))
            self.events.put(("finished", 1))

    def poll_events(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.append_log(value)
            elif kind == "log_path":
                self.log_file = value
                self.append_log("前台日志：" + value + "\n")
            elif kind == "finished":
                self.finish(value)
        if self.running and self.started_at:
            elapsed = int(time.monotonic() - self.started_at)
            self.status_var.set("转换正在运行，已用时 %d 秒。各阶段日志和精度检查写入输出目录。" % elapsed)
        self.window.after(120, self.poll_events)

    def finish(self, returncode):
        self.running = False
        self.progress.stop()
        self.start_button.configure(state="normal")
        for widget in self.input_controls:
            widget.configure(state="readonly" if isinstance(widget, ttk.Combobox) else "normal")
        output = self.last_output
        report_path = output / "conversion_report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        except Exception:
            report = {}
        fdtd = report.get("fdtd_report", {})
        fdtd_status = fdtd.get("fdtd_status")
        result_directory = Path(report.get("result_directory", str(output)))
        if returncode == 0 and report.get("status") == "empty_geometry":
            self.status_var.set("文件有效，但没有实体材料，无需生成 STL / FDTD 对象。")
            self.append_log("\n输入仅含背景 Air / void；材料数量为 0，没有生成 STL 或新的 FDTD 工程。\n")
        elif returncode in (0, 3) and fdtd_status == "needs_material_mapping":
            self.status_var.set("几何已转换；需要指定光学材料，尚未生成可用的 FDTD 工程。")
            self.resume_var.set(True)
            template = result_directory / "fdtd/material_map.template.json"
            missing = fdtd.get("unknown_materials", []) + fdtd.get("verification", {}).get("unavailable_optical_materials", [])
            self.append_log("\n未知或不可用材料：" + ", ".join(missing) + "\n")
            self.append_log("编辑 " + str(template) + "，将 null 改为本机 FDTD 材料库的准确名称。\n")
            self.append_log("选择编辑后的 JSON，保留本输出目录和“复用已完成的几何”，再开始即可。\n")
        elif returncode == 0 and report.get("status") == "complete" and fdtd_status == "verified":
            self.status_var.set("转换完成：STL 与 FDTD 工程已生成，材料、坐标及保存重开检查通过。")
            self.append_log("\nFDTD 工程：" + str(result_directory / "fdtd/imported.fsp") + "\n")
        elif returncode == 0 and report.get("status") == "complete" and fdtd_status in ("ready", "ready_to_import", "generated"):
            self.status_var.set("几何与导入脚本已生成；未执行 FDTD 验证，没有新生成的 .fsp 工程。")
        elif returncode in (0, 3) and report.get("status") == "geometry_converted":
            self.status_var.set("几何已转换；FDTD 验证失败，请查看 fdtd/fdtd_report.json。")
            self.resume_var.set(True)
            self.append_log("\nFDTD 状态：" + str(fdtd_status) + "\n")
        else:
            self.status_var.set("转换未完成。请查看日志与 conversion_report.json；未标记为成功。")
            if report.get("error"):
                self.append_log("\n错误：" + str(report["error"]) + "\n")
        if output.is_dir():
            self.open_button.configure(state="normal")
        if self.mapping_template() is not None:
            self.mapping_button.configure(state="normal")
        self.append_log("\n退出码：%s；报告：%s\n" % (returncode, report_path))

    def open_output(self):
        if self.last_output and self.last_output.is_dir():
            os.startfile(str(self.last_output))

    def open_mapping(self):
        path = self.mapping_template()
        if path is not None:
            self.map_var.set(str(path))
            subprocess.Popen(["notepad.exe", str(path)], shell=False)

    def mapping_template(self):
        if self.last_output:
            for relative in ("reduced/fdtd/material_map.template.json", "fdtd/material_map.template.json"):
                path = self.last_output / relative
                if path.is_file():
                    return path
        return None

    def open_help(self):
        os.startfile(str(ROOT / "README.md"))

    def on_close(self):
        if self.running:
            message = "转换会继续在后台运行。结果将写入：\n%s\n\n关闭窗口后请查看 conversion_report.json 判断完成状态。是否关闭窗口？" % self.last_output
            if not messagebox.askyesno("转换仍在运行", message, parent=self.window):
                return
        self.window.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="A dragged VDATA file")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    window = tk.Tk()
    if args.self_test:
        window.withdraw()
    app = ConverterApp(window, args.input)
    if args.self_test:
        window.update_idletasks()
        window.update()
        assert app.start_button.winfo_exists()
        assert "convert_vdata.py" in make_command("input.vdata", "output")[1]
        assert "--verify-fdtd" in make_command("input.vdata", "output")
        assert "convert_reduced.py" in make_command("input.vdata", "output", reduction=.45)[1]
        reduced_command = make_command("input.vdata", "output", reduction=.65)
        assert reduced_command[reduced_command.index("--reduction") + 1] == "0.65"
        window.destroy()
        print("GUI construction and command-list self-test passed (window hidden).")
        return 0
    window.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
