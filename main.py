import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import random
import math


class ImageHistogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Contrast Enhancement of Greyscale Images using Genetic Algorithm and Histogram Equalization")
        self.root.geometry("1100x820")
        self.root.minsize(950, 700)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.original_img = None
        self.enhanced_img = None
        self.current_image_path = None

        # ==================== Control Area ====================
        control_frame = tk.Frame(root, pady=8)
        control_frame.pack(fill="x", padx=10)

        # --- Row 1: Buttons + Status ---
        row1 = tk.Frame(control_frame)
        row1.pack(fill="x", pady=(0, 6))

        self.browse_btn = tk.Button(
            row1, text="Browse Image...", command=self.browse_image,
            font=("Segoe UI", 10), width=15
        )
        self.browse_btn.pack(side="left", padx=(0, 8))

        self.enhance_btn = tk.Button(
            row1, text="Enhance Contrast (GA)", command=self.enhance_image,
            font=("Segoe UI", 10), width=20, state="disabled"
        )
        self.enhance_btn.pack(side="left", padx=8)

        self.status_label = tk.Label(
            row1, text="No image loaded", font=("Segoe UI", 10), fg="gray"
        )
        self.status_label.pack(side="left", padx=12)

        # --- Row 2: GA Parameters ---
        row2 = tk.Frame(control_frame)
        row2.pack(fill="x", pady=4)

        tk.Label(row2, text="Population:", font=("Segoe UI", 9)).pack(side="left")
        self.entry_ps = tk.Entry(row2, width=6, font=("Segoe UI", 9))
        self.entry_ps.insert(0, "40")
        self.entry_ps.pack(side="left", padx=(2, 12))

        tk.Label(row2, text="Crossover (Pc):", font=("Segoe UI", 9)).pack(side="left")
        self.entry_pc = tk.Entry(row2, width=6, font=("Segoe UI", 9))
        self.entry_pc.insert(0, "0.8")
        self.entry_pc.pack(side="left", padx=(2, 12))

        tk.Label(row2, text="Mutation (Pm):", font=("Segoe UI", 9)).pack(side="left")
        self.entry_pm = tk.Entry(row2, width=6, font=("Segoe UI", 9))
        self.entry_pm.insert(0, "0.1")
        self.entry_pm.pack(side="left", padx=(2, 12))

        tk.Label(row2, text="Generations/Iterations:", font=("Segoe UI", 9)).pack(side="left")
        self.entry_gen = tk.Entry(row2, width=6, font=("Segoe UI", 9))
        self.entry_gen.insert(0, "100")
        self.entry_gen.pack(side="left", padx=(2, 12))

        # --- Row 3: Progress bar + live fitness ---
        row3 = tk.Frame(control_frame)
        row3.pack(fill="x", pady=(6, 0))

        self.progress = ttk.Progressbar(row3, orient="horizontal", mode="determinate", length=400)
        self.progress.pack(side="left", padx=(0, 12))

        self.fitness_label = tk.Label(
            row3, text="Best Fitness: -", font=("Segoe UI", 9), fg="#333"
        )
        self.fitness_label.pack(side="left")

        # --- Row 4: Save buttons ---
        row4 = tk.Frame(control_frame)
        row4.pack(fill="x", pady=(10, 0))

        tk.Label(row4, text="Save:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))

        self.btn_save_orig_hist = tk.Button(
            row4, text="Original Histogram", command=self.save_original_histogram,
            font=("Segoe UI", 9), width=16, state="disabled"
        )
        self.btn_save_orig_hist.pack(side="left", padx=4)

        self.btn_save_enh_img = tk.Button(
            row4, text="Enhanced Image", command=self.save_enhanced_image,
            font=("Segoe UI", 9), width=14, state="disabled"
        )
        self.btn_save_enh_img.pack(side="left", padx=4)

        self.btn_save_enh_hist = tk.Button(
            row4, text="Equalized Histogram", command=self.save_enhanced_histogram,
            font=("Segoe UI", 9), width=16, state="disabled"
        )
        self.btn_save_enh_hist.pack(side="left", padx=4)

        # ==================== Display Area (2x2) ====================
        self.display_frame = tk.Frame(root)
        self.display_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
        self.ax_img_orig, self.ax_hist_orig = axes[0, 0], axes[0, 1]
        self.ax_img_enh,  self.ax_hist_enh  = axes[1, 0], axes[1, 1]
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._clear_all_plots()
        self.canvas.draw()

    def on_closing(self):
        plt.close(self.fig)
        self.root.destroy()

    def _clear_all_plots(self):
        for ax, title in [
            (self.ax_img_orig, "Original Image"),
            (self.ax_hist_orig, "Original Histogram"),
            (self.ax_img_enh, "Enhanced Image"),
            (self.ax_hist_enh, "Equalized Histogram")
        ]:
            ax.clear()
            ax.set_title(title)
            if "Histogram" in title:
                ax.set_xlabel("Pixel Intensity (0–255)")
                ax.set_ylabel("Frequency")
                ax.set_xlim(0, 255)
                ax.grid(True, alpha=0.3)
            else:
                ax.axis("off")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select a greyscale image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", f"Could not load image:\n{image_path}")
            return

        self.original_img = img
        self.enhanced_img = None
        self.current_image_path = image_path

        self.status_label.config(text=Path(image_path).name, fg="black")
        self.enhance_btn.config(state="normal")
        self.progress["value"] = 0
        self.fitness_label.config(text="Best Fitness: -")

        # Enable / disable save buttons
        self.btn_save_orig_hist.config(state="normal")
        self.btn_save_enh_img.config(state="disabled")
        self.btn_save_enh_hist.config(state="disabled")

        self._clear_all_plots()

        self.ax_img_orig.imshow(img, cmap="gray", vmin=0, vmax=255)
        self.ax_img_orig.set_title("Original Image")
        self.ax_img_orig.axis("off")

        self.ax_hist_orig.hist(img.ravel(), bins=256, range=[0, 256],
                               color="steelblue", alpha=0.85)
        self.ax_hist_orig.set_title("Original Histogram")
        self.ax_hist_orig.set_xlabel("Pixel Intensity (0–255)")
        self.ax_hist_orig.set_ylabel("Frequency")
        self.ax_hist_orig.set_xlim(0, 255)
        self.ax_hist_orig.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    # ============================================================
    #                     SAVE FUNCTIONS
    # ============================================================

    def _get_save_path(self, default_name: str):
        initial_dir = str(Path(self.current_image_path).parent) if self.current_image_path else "."
        file_path = filedialog.asksaveasfilename(
            title="Save as",
            initialdir=initial_dir,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        return file_path

    def save_original_histogram(self):
        if self.original_img is None:
            return
        path = self._get_save_path("original_histogram.png")
        if path:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(self.original_img.ravel(), bins=256, range=[0, 256],
                    color="steelblue", alpha=0.85)
            ax.set_title("Original Histogram")
            ax.set_xlabel("Pixel Intensity (0–255)")
            ax.set_ylabel("Frequency")
            ax.set_xlim(0, 255)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            self.status_label.config(text=f"Saved: {Path(path).name}", fg="green")

    def save_enhanced_image(self):
        if self.enhanced_img is None:
            messagebox.showwarning("No Enhanced Image", "Please run the enhancement first.")
            return
        path = self._get_save_path("enhanced_image.png")
        if path:
            cv2.imwrite(path, self.enhanced_img)
            self.status_label.config(text=f"Saved: {Path(path).name}", fg="green")

    def save_enhanced_histogram(self):
        if self.enhanced_img is None:
            messagebox.showwarning("No Enhanced Image", "Please run the enhancement first.")
            return
        path = self._get_save_path("equalized_histogram.png")
        if path:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(self.enhanced_img.ravel(), bins=256, range=[0, 256],
                    color="darkblue", alpha=0.85)
            ax.set_title("Equalized Histogram")
            ax.set_xlabel("Pixel Intensity (0–255)")
            ax.set_ylabel("Frequency")
            ax.set_xlim(0, 255)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            self.status_label.config(text=f"Saved: {Path(path).name}", fg="green")

    # ============================================================
    #                GENETIC ALGORITHM
    # ============================================================

    def create_chromosome(self, n: int) -> np.ndarray:
        chrom = np.random.randint(0, 256, size=n)
        chrom[0] = 0
        chrom[-1] = 255
        chrom.sort()
        return chrom

    def apply_chromosome(self, img: np.ndarray, unique_levels: np.ndarray,
                         chromosome: np.ndarray) -> np.ndarray:
        lut = np.zeros(256, dtype=np.uint8)
        for i, level in enumerate(unique_levels):
            lut[level] = chromosome[i]
        return lut[img]

    def calculate_fitness(self, enhanced: np.ndarray) -> float:
        g = enhanced.astype(np.float64)
        g_pad = np.pad(g, pad_width=1, mode='edge')

        δh = (g_pad[2:, 0:-2] + 2 * g_pad[2:, 1:-1] + g_pad[2:, 2:] -
              g_pad[0:-2, 0:-2] - 2 * g_pad[0:-2, 1:-1] - g_pad[0:-2, 2:])

        δv = (g_pad[0:-2, 2:] + 2 * g_pad[1:-1, 2:] + g_pad[2:, 2:] -
              g_pad[0:-2, 0:-2] - 2 * g_pad[1:-1, 0:-2] - g_pad[2:, 0:-2])

        magnitude = np.sqrt(δh**2 + δv**2)
        E = np.sum(magnitude)
        n_edges = np.sum(magnitude > 30)

        if E <= 1 or n_edges == 0:
            return 0.0
        return math.log(math.log(E + 1e-5)) * n_edges

    def roulette_wheel_selection(self, population, fitnesses, num_select):
        fitnesses = np.array(fitnesses, dtype=np.float64)
        fitnesses = np.maximum(fitnesses, 0)
        total = np.sum(fitnesses)

        if total == 0:
            indices = np.random.choice(len(population), size=num_select, replace=True)
            return [population[i].copy() for i in indices]

        probs = fitnesses / total
        indices = np.random.choice(len(population), size=num_select, replace=True, p=probs)
        return [population[i].copy() for i in indices]

    def two_point_crossover(self, parent1, parent2):
        n = len(parent1)
        if n < 3:
            return parent1.copy(), parent2.copy()

        pt1, pt2 = sorted(random.sample(range(1, n - 1), 2))
        child1 = np.concatenate([parent1[:pt1], parent2[pt1:pt2], parent1[pt2:]])
        child2 = np.concatenate([parent2[:pt1], parent1[pt1:pt2], parent2[pt2:]])

        child1.sort()
        child2.sort()
        child1[0], child1[-1] = 0, 255
        child2[0], child2[-1] = 0, 255
        return child1, child2

    def mutate(self, chromosome, Pm):
        if random.random() > Pm:
            return chromosome

        chrom = chromosome.copy()
        n = len(chrom)
        num_mutate = max(1, int(0.05 * n))
        indices = random.sample(range(1, n - 1), min(num_mutate, n - 2))

        for idx in indices:
            low = chrom[idx - 1]
            high = chrom[idx + 1]
            if high > low:
                chrom[idx] = random.randint(low, high)

        chrom.sort()
        chrom[0], chrom[-1] = 0, 255
        return chrom

    def enhance_image(self):
        if self.original_img is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            Ps = int(self.entry_ps.get())
            Pc = float(self.entry_pc.get())
            Pm = float(self.entry_pm.get())
            generations = int(self.entry_gen.get())

            if Ps < 4 or generations < 1:
                raise ValueError("Population must be >= 4 and Generations >= 1")
            if not (0 < Pc <= 1) or not (0 <= Pm <= 1):
                raise ValueError("Pc and Pm must be between 0 and 1")
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return

        self.enhance_btn.config(state="disabled")
        self.browse_btn.config(state="disabled")
        self.status_label.config(text="Running Genetic Algorithm...", fg="blue")
        self.progress["value"] = 0
        self.progress["maximum"] = generations
        self.root.update()

        img = self.original_img
        unique_levels = np.unique(img)
        n = len(unique_levels)

        population = [self.create_chromosome(n) for _ in range(Ps)]

        best_chromosome = None
        best_fitness = -1.0

        for gen in range(generations):
            fitnesses = []
            for chrom in population:
                enhanced = self.apply_chromosome(img, unique_levels, chrom)
                fit = self.calculate_fitness(enhanced)
                fitnesses.append(fit)

                if fit > best_fitness:
                    best_fitness = fit
                    best_chromosome = chrom.copy()

            self.progress["value"] = gen + 1
            self.fitness_label.config(
                text=f"Generation {gen+1}/{generations}   |   Best Fitness: {best_fitness:.2f}"
            )
            self.root.update()

            num_elite = max(2, int(Ps * (1 - Pc)))
            elite_indices = np.argsort(fitnesses)[-num_elite:]
            new_population = [population[i].copy() for i in elite_indices]

            num_offspring = Ps - num_elite
            parents = self.roulette_wheel_selection(population, fitnesses, num_offspring)

            offspring = []
            for i in range(0, num_offspring, 2):
                p1 = parents[i]
                p2 = parents[min(i + 1, num_offspring - 1)]
                c1, c2 = self.two_point_crossover(p1, p2)
                offspring.append(self.mutate(c1, Pm))
                if len(offspring) < num_offspring:
                    offspring.append(self.mutate(c2, Pm))

            new_population.extend(offspring[:num_offspring])
            population = new_population

        # Apply best result
        self.enhanced_img = self.apply_chromosome(img, unique_levels, best_chromosome)

        # Update display
        self.ax_img_enh.clear()
        self.ax_img_enh.imshow(self.enhanced_img, cmap="gray", vmin=0, vmax=255)
        self.ax_img_enh.set_title("Enhanced Image (GA)")
        self.ax_img_enh.axis("off")

        self.ax_hist_enh.clear()
        self.ax_hist_enh.hist(self.enhanced_img.ravel(), bins=256, range=[0, 256],
                              color="darkblue", alpha=0.85)
        self.ax_hist_enh.set_title("Enhanced Histogram")
        self.ax_hist_enh.set_xlabel("Pixel Intensity (0–255)")
        self.ax_hist_enh.set_ylabel("Frequency")
        self.ax_hist_enh.set_xlim(0, 255)
        self.ax_hist_enh.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

        self.status_label.config(text=f"Done! Best Fitness: {best_fitness:.2f}", fg="green")
        self.enhance_btn.config(state="normal")
        self.browse_btn.config(state="normal")

        # Enable enhanced save buttons
        self.btn_save_enh_img.config(state="normal")
        self.btn_save_enh_hist.config(state="normal")


def main():
    root = tk.Tk()
    app = ImageHistogramApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()