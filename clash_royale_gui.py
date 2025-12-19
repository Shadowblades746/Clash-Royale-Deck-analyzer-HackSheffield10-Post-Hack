import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import Dict, List, Optional, Callable

import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from clash_royale_archetype_classifier import (
    CARD_NAME_TO_ID,
    CLASH_ROYALE_CARDS,
    get_card_info,
    find_card_id_by_name,
    calculate_deck_stats,
    train_new_model,
    EnhancedQuickClashPredictor,
)


class DragDropCardGUI:
    def __init__(
        self, parent_frame: tk.Frame, on_deck_update_callback: Callable
    ) -> None:
        self.canvas = None
        self.scrollable_frame = None
        self.search_var = None
        self.deck_label = None
        self.search_entry = None
        self.rarity_var = None
        self.elixir_var = None
        self.type_var = None
        self.parent = parent_frame
        self.on_deck_update = on_deck_update_callback
        self.card_images: Dict[str, Optional[Image.Image]] = {}
        self.card_photos: List[ImageTk.PhotoImage] = []
        self.card_buttons: Dict[str, tk.Label] = {}
        self.deck_slots: List[Dict] = []
        self.all_cards: List[Dict] = []

        self.setup_card_database()
        self.create_drag_drop_interface()

    def setup_card_database(self) -> None:
        """Load and prepare card database from CLASH_ROYALE_CARDS."""
        self.all_cards = [
            {
                "id": card_id,
                "name": card_info["name"],
                "elixir": card_info["elixir"],
                "type": card_info["type"],
                "rarity": card_info["rarity"],
            }
            for card_id, card_info in CLASH_ROYALE_CARDS.items()
        ]
        self.all_cards.sort(key=lambda x: (x["elixir"], x["name"]))

    def create_drag_drop_interface(self) -> None:
        """Build the main drag-and-drop UI."""
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill="both", expand=True)

        left_frame = ttk.LabelFrame(main_frame, text="Available Cards", padding="10")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.create_search_filter(left_frame)
        self.create_cards_scrollable(left_frame)

        right_frame = ttk.LabelFrame(main_frame, text="Your Deck (0/8)", padding="10")
        right_frame.pack(side="right", fill="both", expand=True)
        self.deck_label = right_frame

        self.create_deck_slots(right_frame)
        self.create_deck_controls(right_frame)

    def create_search_filter(self, parent: tk.Widget) -> None:
        """Add search and filter controls."""
        search_frame = ttk.Frame(parent)
        search_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(search_frame, text="Search:").pack(side="left", padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(
            search_frame, textvariable=self.search_var, width=20
        )
        self.search_entry.pack(side="left", padx=(0, 10))
        self.search_entry.bind("<KeyRelease>", self.on_search)

        ttk.Label(search_frame, text="Elixir:").pack(side="left", padx=(0, 5))
        self.elixir_var = tk.StringVar(value="All")
        elixir_combo = ttk.Combobox(
            search_frame,
            textvariable=self.elixir_var,
            values=["All", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            state="readonly",
            width=8,
        )
        elixir_combo.pack(side="left", padx=(0, 10))
        elixir_combo.bind("<<ComboboxSelected>>", self.on_filter)

        ttk.Label(search_frame, text="Type:").pack(side="left", padx=(0, 5))
        self.type_var = tk.StringVar(value="All")
        type_combo = ttk.Combobox(
            search_frame,
            textvariable=self.type_var,
            values=["All", "Troop", "Spell", "Building"],
            state="readonly",
            width=10,
        )
        type_combo.pack(side="left", padx=(0, 10))
        type_combo.bind("<<ComboboxSelected>>", self.on_filter)

        ttk.Label(search_frame, text="Rarity:").pack(side="left", padx=(0, 5))
        self.rarity_var = tk.StringVar(value="All")
        rarity_combo = ttk.Combobox(
            search_frame,
            textvariable=self.rarity_var,
            values=["All", "Common", "Rare", "Epic", "Legendary", "Champion"],
            state="readonly",
            width=12,
        )
        rarity_combo.pack(side="left")
        rarity_combo.bind("<<ComboboxSelected>>", self.on_filter)

    def apply_filters(self) -> None:
        """Apply search and filters, then display matching cards."""
        search_text = self.search_var.get().lower()
        elixir_filter = self.elixir_var.get()
        type_filter = self.type_var.get()
        rarity_filter = self.rarity_var.get()

        rarity_name_to_value = {
            "Common": 1,
            "Rare": 2,
            "Epic": 3,
            "Legendary": 4,
            "Champion": 5,
        }

        filtered_cards = []
        for card in self.all_cards:
            if search_text and search_text not in card["name"].lower():
                continue
            if elixir_filter != "All" and str(card["elixir"]) != elixir_filter:
                continue
            if type_filter != "All" and card["type"].title() != type_filter:
                continue
            if rarity_filter != "All":
                target = rarity_name_to_value.get(rarity_filter)
                if target is not None and int(card["rarity"]) != target:
                    continue
            filtered_cards.append(card)

        self.display_cards(filtered_cards)
        self.canvas.yview_moveto(0)

    def create_cards_scrollable(self, parent: tk.Widget) -> None:
        """Create scrollable card list with mouse wheel support."""
        card_container = ttk.Frame(parent)
        card_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(card_container, bg="#34495e")
        scrollbar = ttk.Scrollbar(
            card_container, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.scrollable_frame.bind("<Enter>", lambda e: self._bind_to_mousewheel())
        self.scrollable_frame.bind("<Leave>", lambda e: self._unbind_from_mousewheel())

        self.load_card_images()
        self.display_cards()

    def _bind_to_mousewheel(self) -> None:
        """Bind mouse wheel events on enter."""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_from_mousewheel(self) -> None:
        """Unbind mouse wheel events on leave."""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Handle mouse wheel scrolling across platforms."""
        if hasattr(event, "num") and event.num in (4, 5):
            delta = -1 if event.num == 4 else 1
        else:
            delta = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(delta, "units")

    def load_card_images(self) -> None:
        """Load card images from 'images' directory."""
        images_dir = "images"
        if not os.path.exists(images_dir):
            return

        for card in self.all_cards:
            card_name = card["name"]
            image_path = os.path.join(images_dir, f"{card_name}.png")

            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    image = image.resize((80, 100), Image.Resampling.LANCZOS)
                    self.card_images[card_name] = image
                except Exception:
                    self.card_images[card_name] = None
            else:
                self.card_images[card_name] = None

    def display_cards(self, cards_to_show: Optional[List[Dict]] = None) -> None:
        """Display cards in grid; use images if available."""
        if cards_to_show is None:
            cards_to_show = self.all_cards

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.card_buttons = {}

        max_cols = 6
        for i, card in enumerate(cards_to_show):
            row, col = divmod(i, max_cols)

            card_frame = ttk.Frame(
                self.scrollable_frame, relief="raised", borderwidth=1
            )
            card_frame.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
            card_frame.card_data = card
            card_frame.bind("<Button-1>", self.on_card_click)

            if card["name"] in self.card_images and self.card_images[card["name"]]:
                photo = ImageTk.PhotoImage(self.card_images[card["name"]])
                self.card_photos.append(photo)
                btn = tk.Label(
                    card_frame,
                    image=photo,
                    cursor="hand2",
                    bg="#2c3e50",
                )
            else:
                btn = tk.Label(
                    card_frame,
                    text=f"{card['name']}\n({card['elixir']}⏱️)",
                    cursor="hand2",
                    bg="#34495e",
                    fg="white",
                    wraplength=80,
                    justify="center",
                )

            btn.pack(padx=2, pady=2)
            btn.card_data = card
            btn.bind("<Button-1>", self.on_card_click)
            self.card_buttons[card["name"]] = btn

        for i in range(max_cols):
            self.scrollable_frame.columnconfigure(i, weight=1)

    def create_deck_slots(self, parent: tk.Widget) -> None:
        """Create 8 deck slots in 2x4 grid."""
        slots_frame = ttk.Frame(parent)
        slots_frame.pack(fill="both", expand=True, pady=(0, 10))

        for slot_num in range(8):
            row, col = divmod(slot_num, 4)

            slot_frame = ttk.Frame(
                slots_frame, relief="sunken", borderwidth=2, width=100, height=120
            )
            slot_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            slot_frame.grid_propagate(False)

            slot_label = tk.Label(
                slot_frame, text=f"Slot {slot_num + 1}", bg="#95a5a6", fg="black"
            )
            slot_label.pack(fill="both", expand=True)

            slot_info = {"frame": slot_frame, "label": slot_label, "card": None}
            self.deck_slots.append(slot_info)

            slot_frame.bind("<Button-1>", lambda e, s=slot_num: self.on_slot_click(s))

        for i in range(4):
            slots_frame.columnconfigure(i, weight=1)
        for i in range(2):
            slots_frame.rowconfigure(i, weight=1)

    def create_deck_controls(self, parent: tk.Widget) -> None:
        """Add deck control buttons."""
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill="x")

        ttk.Button(controls_frame, text="Clear Deck", command=self.clear_deck).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(
            controls_frame, text="Auto-fill Example", command=self.auto_fill_example
        ).pack(side="left", padx=(0, 5))

    def on_card_click(self, event: tk.Event) -> None:
        """Add card to first empty slot; prevent duplicates."""
        card_data = event.widget.card_data

        for slot in self.deck_slots:
            if slot["card"] and slot["card"]["id"] == card_data["id"]:
                messagebox.showinfo(
                    "Duplicate Card",
                    f"'{card_data['name']}' is already in your deck.",
                )
                return

        empty_slot = next((s for s in self.deck_slots if s["card"] is None), None)
        if empty_slot:
            self.add_card_to_slot(card_data, empty_slot)
        else:
            messagebox.showinfo("Deck Full", "Your deck is full! Remove a card first.")

    def on_slot_click(self, slot_index: int) -> None:
        """Remove card from clicked slot."""
        slot = self.deck_slots[slot_index]
        if slot["card"] is not None:
            self.remove_card_from_slot(slot)

    def add_card_to_slot(self, card_data: Dict, slot: Dict) -> None:
        """Add card to slot."""
        slot["card"] = card_data

        for widget in slot["frame"].winfo_children():
            widget.destroy()

        if (
            card_data["name"] in self.card_images
            and self.card_images[card_data["name"]]
        ):
            photo = ImageTk.PhotoImage(self.card_images[card_data["name"]])
            self.card_photos.append(photo)
            card_label = tk.Label(slot["frame"], image=photo, bg="#27ae60")
        else:
            card_label = tk.Label(
                slot["frame"],
                text=f"{card_data['name']}\n({card_data['elixir']}⏱️)",
                bg="#27ae60",
                fg="white",
                wraplength=80,
                justify="center",
            )

        card_label.pack(fill="both", expand=True)
        card_label.card_data = card_data
        card_label.bind("<Button-1>", lambda e: self.remove_card_from_slot(slot))

        self.update_deck_status()

    def remove_card_from_slot(self, slot: Dict) -> None:
        """Remove card and reset slot display."""
        slot["card"] = None

        for widget in slot["frame"].winfo_children():
            widget.destroy()

        slot_label = tk.Label(
            slot["frame"],
            text=f"Slot {self.deck_slots.index(slot) + 1}",
            bg="#95a5a6",
            fg="black",
        )
        slot_label.pack(fill="both", expand=True)

        self.update_deck_status()

    def update_deck_status(self) -> None:
        """Update deck title with count, avg elixir, and total."""
        current_cards = [s["card"] for s in self.deck_slots if s["card"]]
        card_count = len(current_cards)

        total_elixir = sum(c["elixir"] for c in current_cards)
        avg_elixir = total_elixir / card_count if card_count > 0 else 0.0

        self.deck_label.configure(
            text=f"Your Deck ({card_count}/8)  —  Avg: {avg_elixir:.2f}  •  Total: {total_elixir}"
        )

        if self.on_deck_update:
            self.on_deck_update(current_cards)

    def clear_deck(self) -> None:
        """Remove all cards from deck."""
        for slot in self.deck_slots:
            if slot["card"]:
                self.remove_card_from_slot(slot)

    def auto_fill_example(self) -> None:
        """Fill deck with example cards."""
        example_cards = [
            "MegaKnight",
            "Pekka",
            "Bandit",
            "Ghost",
            "ElectroWizard",
            "Zap",
            "Poison",
            "Tornado",
        ]

        self.clear_deck()

        for i, card_name in enumerate(example_cards[: len(self.deck_slots)]):
            card_data = next(
                (c for c in self.all_cards if c["name"].lower() == card_name.lower()),
                None,
            )
            if card_data:
                self.add_card_to_slot(card_data, self.deck_slots[i])

    def on_search(self, event: Optional[tk.Event] = None) -> None:
        """Trigger filter application on search input."""
        self.apply_filters()

    def on_filter(self, event: Optional[tk.Event] = None) -> None:
        """Trigger filter application on filter selection."""
        self.apply_filters()

    def get_deck_card_names(self) -> List[str]:
        """Return list of card names in current deck."""
        return [s["card"]["name"] for s in self.deck_slots if s["card"]]


def setup_styles() -> None:
    """Configure ttk styles for UI."""
    style = ttk.Style()
    style.theme_use("clam")

    style.configure(
        "Title.TLabel",
        background="#2c3e50",
        foreground="#ecf0f1",
        font=("Arial", 16, "bold"),
    )
    style.configure(
        "Subtitle.TLabel",
        background="#2c3e50",
        foreground="#bdc3c7",
        font=("Arial", 12),
    )
    style.configure("Card.TFrame", background="#34495e", relief="raised", borderwidth=1)
    style.configure(
        "Accent.TButton",
        background="#e74c3c",
        foreground="white",
        font=("Arial", 10, "bold"),
    )
    style.configure(
        "Success.TButton",
        background="#27ae60",
        foreground="white",
        font=("Arial", 10, "bold"),
    )


def setup_autocomplete(entry: ttk.Entry) -> None:
    """Add tab-based autocomplete for card names."""

    def autocomplete(event: tk.Event) -> str:
        current_text = entry.get().lower()
        if not current_text:
            return "break"

        matches = [
            name for name in CARD_NAME_TO_ID.keys() if name.startswith(current_text)
        ]

        if matches:
            entry.delete(0, tk.END)
            entry.insert(0, matches[0].title())
            entry.select_range(len(current_text), tk.END)
        return "break"

    entry.bind("<Tab>", autocomplete)


class ClashRoyaleGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.progress = None
        self.results_text = None
        self.train_btn = None
        self.ids_entry = None
        self.ids_frame = None
        self.url_entry = None
        self.card_entries = None
        self.names_frame = None
        self.predict_btn = None
        self.drag_predict_btn = None
        self.input_method = None
        self.drag_drop_gui = None
        self.status_label = None
        self.overview_card_photos: List[ImageTk.PhotoImage] = []
        self.overview_fig = None
        self.overview_images_frame = None
        self.overview_canvas = None
        self.overview_canvas_widget = None
        self.overview_ax_train_pie = None
        self.overview_ax_bar = None
        self.overview_ax_pie = None
        self.notebook = None
        self.url_frame = None
        self.root = root
        self.root.title("Clash Royale Archetype Classifier")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        self.predictor: Optional[EnhancedQuickClashPredictor] = None
        self.load_model_attempted = False
        self.current_deck_card_ids: List[int] = []

        setup_styles()
        self.create_main_interface()
        self.load_model_background()

    def create_main_interface(self) -> None:
        """Build main notebook interface."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.create_drag_drop_tab()
        self.create_text_input_tab()
        self.create_results_tab()
        self.create_deck_overview_tab()

        self.notebook.select(0)
        self.on_method_change()

    def create_deck_overview_tab(self) -> None:
        """Create overview tab with charts and deck images."""
        overview_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(overview_frame, text="Deck overview")

        chart_frame = ttk.LabelFrame(overview_frame, text="Deck Summary", padding="8")
        chart_frame.pack(fill="both", expand=True, padx=5, pady=(0, 8))

        fig = plt.Figure(figsize=(12, 3.5), dpi=100)
        self.overview_ax_pie = fig.add_subplot(1, 3, 1)
        self.overview_ax_bar = fig.add_subplot(1, 3, 2)
        self.overview_ax_train_pie = fig.add_subplot(1, 3, 3)

        for ax in [
            self.overview_ax_pie,
            self.overview_ax_bar,
            self.overview_ax_train_pie,
        ]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()

        self.overview_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        self.overview_canvas_widget = self.overview_canvas.get_tk_widget()
        self.overview_canvas_widget.pack(fill="both", expand=True)
        self.overview_fig = fig

        images_frame_label = ttk.LabelFrame(overview_frame, text="Deck", padding="8")
        images_frame_label.pack(fill="x", padx=5, pady=(0, 8))

        self.overview_images_frame = ttk.Frame(images_frame_label)
        self.overview_images_frame.pack(fill="x", expand=False)

    def update_deck_overview(
        self, all_probabilities: Dict, deck_card_ids: List[int]
    ) -> None:
        """Update overview with pie/bar charts and deck images."""
        try:
            stats = calculate_deck_stats(deck_card_ids)
            avg_elixir = stats.get("average_elixir")
            four_card_cycle = stats.get("four_card_cycle")

            stats_parts = []
            if avg_elixir is not None:
                stats_parts.append(f"Avg Elixir: {avg_elixir:.2f}")
            if four_card_cycle is not None:
                stats_parts.append(f"4-Card Cycle: {four_card_cycle}")

            stats_title = (
                "  •  ".join(stats_parts) if stats_parts else "No deck stats available"
            )
            self.overview_fig.suptitle(stats_title, fontsize=10, y=0.98)

            self.overview_ax_pie.clear()
            self.overview_ax_bar.clear()
            self.overview_ax_train_pie.clear()

            # Pie: Card type ratios
            type_counts = {"troop": 0, "spell": 0, "building": 0}
            for cid in deck_card_ids:
                info = get_card_info(cid)
                if info:
                    type_counts[info.get("type", "troop")] += 1

            labels = [f"{k.title()} ({v})" for k, v in type_counts.items() if v > 0]
            sizes = [v for v in type_counts.values() if v > 0]
            colors = ["#2ecc71", "#3498db", "#f1c40f"]

            if sizes:
                self.overview_ax_pie.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=colors[: len(sizes)],
                    startangle=90,
                )
                self.overview_ax_pie.set_title("Card Types", fontsize=10, pad=10)

            # Bar: Archetype probabilities
            if all_probabilities:
                sorted_probs = sorted(
                    all_probabilities.items(), key=lambda x: x[1], reverse=True
                )[:5]
                archetypes = [
                    item[0].replace("_", " ").title() for item in sorted_probs
                ]
                probs = [item[1] * 100 for item in sorted_probs]

                bars = self.overview_ax_bar.barh(archetypes, probs, color="#3498db")
                self.overview_ax_bar.set_xlabel("Probability (%)", fontsize=9)
                self.overview_ax_bar.set_title("Top Archetypes", fontsize=10, pad=10)
                self.overview_ax_bar.invert_yaxis()

                for bar in bars:
                    width = bar.get_width()
                    self.overview_ax_bar.text(
                        width,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.1f}%",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )

            # Pie: Training data distribution (replace bar with pie)
            try:
                training_file = "training_data.py"
                if os.path.exists(training_file):
                    with open(training_file, "r") as f:
                        content = f.read()

                    archetype_counts = {}
                    for arch in [
                        "beatdown",
                        "control",
                        "cycle",
                        "bridge_spam",
                        "bait",
                        "siege",
                        "split_lane",
                    ]:
                        count = content.count(f'"{arch}"')
                        if count > 0:
                            archetype_counts[arch] = count

                    if archetype_counts:
                        labels = [
                            k.replace("_", " ").title() for k in archetype_counts.keys()
                        ]
                        sizes = list(archetype_counts.values())
                        colors = [
                            "#e74c3c",
                            "#3498db",
                            "#2ecc71",
                            "#f39c12",
                            "#9b59b6",
                            "#1abc9c",
                            "#e67e22",
                        ]

                        self.overview_ax_train_pie.pie(
                            sizes,
                            labels=labels,
                            autopct="%1.1f%%",
                            colors=colors[: len(sizes)],
                            startangle=90,
                        )
                        self.overview_ax_train_pie.set_title(
                            "Training Data", fontsize=10, pad=10
                        )
            except Exception as e:
                print(f"Error loading training data: {e}")

            self.overview_fig.tight_layout(rect=[0, 0, 1, 0.94])
            self.overview_canvas.draw()

            # Update deck images
            for w in self.overview_images_frame.winfo_children():
                w.destroy()
            self.overview_card_photos.clear()

            images_dir = "images"
            for cid in deck_card_ids:
                info = get_card_info(cid)
                if info:
                    img_path = os.path.join(images_dir, f"{info['name']}.png")
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).resize(
                                (60, 70), Image.Resampling.LANCZOS
                            )
                            photo = ImageTk.PhotoImage(img)
                            self.overview_card_photos.append(photo)
                            lbl = tk.Label(self.overview_images_frame, image=photo)
                            lbl.pack(side="left", padx=2)
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

        except Exception as e:
            print(f"Error updating overview: {e}")

    def create_drag_drop_tab(self) -> None:
        """Create drag-and-drop deck builder tab."""
        drag_drop_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(drag_drop_frame, text="Drag & Drop Builder")

        status_frame = ttk.Frame(drag_drop_frame)
        status_frame.pack(fill="x", pady=(0, 10))

        self.status_label = ttk.Label(
            status_frame, text="Loading model...", style="Subtitle.TLabel"
        )
        self.status_label.pack(side="left")

        self.drag_drop_gui = DragDropCardGUI(drag_drop_frame, self.on_deck_update)

        predict_frame = ttk.Frame(drag_drop_frame)
        predict_frame.pack(fill="x", pady=(10, 0))

        self.drag_predict_btn = ttk.Button(
            predict_frame,
            text="Predict Archetype from Deck",
            command=self.predict_from_drag_drop,
            style="Accent.TButton",
            state="disabled",
        )
        self.drag_predict_btn.pack()

    def create_text_input_tab(self) -> None:
        """Create text input tab for deck entry."""
        text_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(text_frame, text="Text Input")
        self.create_input_method_section(text_frame)

    def create_results_tab(self) -> None:
        """Create results display tab."""
        results_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(results_frame, text="Analysis Results")

        self.create_results_section(results_frame)
        self.create_training_section(results_frame)

    def create_input_method_section(self, parent: tk.Widget) -> None:
        """Add input method selection and fields."""
        input_frame = ttk.LabelFrame(parent, text="Text Input Methods", padding="15")
        input_frame.pack(fill="both", expand=True)

        method_frame = ttk.Frame(input_frame)
        method_frame.pack(fill="x", pady=(0, 15))

        self.input_method = tk.StringVar(value="names")

        for value, text in [
            ("names", "Card Names"),
            ("url", "Deck URL"),
            ("ids", "Card IDs"),
        ]:
            ttk.Radiobutton(
                method_frame,
                text=text,
                variable=self.input_method,
                value=value,
                command=self.on_method_change,
            ).pack(side="left", padx=(0, 20))

        self.create_names_input(input_frame)
        self.create_url_input(input_frame)
        self.create_ids_input(input_frame)

        self.predict_btn = ttk.Button(
            input_frame,
            text="Predict Archetype",
            command=self.predict_archetype,
            style="Accent.TButton",
            state="disabled",
        )
        self.predict_btn.pack(pady=(10, 0))

    def create_names_input(self, parent: tk.Widget) -> None:
        """Create card names input fields."""
        self.names_frame = ttk.Frame(parent)

        ttk.Label(self.names_frame, text="Enter 8 card names (one per line):").pack(
            anchor="w", pady=(0, 10)
        )

        entries_frame = ttk.Frame(self.names_frame)
        entries_frame.pack(fill="x")

        self.card_entries = []
        for i in range(8):
            row_frame = ttk.Frame(entries_frame)
            row_frame.pack(fill="x", pady=2)

            ttk.Label(row_frame, text=f"Card {i + 1}:", width=8).pack(side="left")

            entry = ttk.Entry(row_frame, width=30)
            entry.pack(side="left", fill="x", expand=True, padx=(5, 0))
            self.card_entries.append(entry)
            setup_autocomplete(entry)

        control_frame = ttk.Frame(self.names_frame)
        control_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(
            control_frame, text="Fill Example Deck", command=self.fill_example_deck
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            control_frame, text="Clear All Fields", command=self.clear_text_fields
        ).pack(side="left")

    def create_url_input(self, parent: tk.Widget) -> None:
        """Create deck URL input field."""
        self.url_frame = ttk.Frame(parent)

        ttk.Label(self.url_frame, text="Enter Clash Royale deck URL:").pack(
            anchor="w", pady=(0, 10)
        )

        self.url_entry = ttk.Entry(self.url_frame, width=80)
        self.url_entry.pack(fill="x")

        ttk.Label(
            self.url_frame,
            text="Example: clashroyale://copyDeck?deck=26000063;26000015;...",
            font=("Arial", 8),
            foreground="#7f8c8d",
        ).pack(anchor="w", pady=(5, 0))

        ttk.Button(
            self.url_frame,
            text="Clear URL",
            command=lambda: self.url_entry.delete(0, tk.END),
        ).pack(anchor="w", pady=(5, 0))

    def create_ids_input(self, parent: tk.Widget) -> None:
        """Create card IDs input field."""
        self.ids_frame = ttk.Frame(parent)

        ttk.Label(self.ids_frame, text="Enter 8 card IDs (semicolon-separated):").pack(
            anchor="w", pady=(0, 10)
        )

        self.ids_entry = ttk.Entry(self.ids_frame, width=80)
        self.ids_entry.pack(fill="x")

        ttk.Label(
            self.ids_frame,
            text="Example: 26000063;26000015;26000009;26000018;26000068;28000012;28000015;27000007",
            font=("Arial", 8),
            foreground="#7f8c8d",
        ).pack(anchor="w", pady=(5, 0))

        ttk.Button(
            self.ids_frame,
            text="Clear IDs",
            command=lambda: self.ids_entry.delete(0, tk.END),
        ).pack(anchor="w", pady=(5, 0))

    def create_results_section(self, parent: tk.Widget) -> None:
        """Create results display area."""
        results_frame = ttk.LabelFrame(parent, text="Prediction Results", padding="15")
        results_frame.pack(fill="both", expand=True, pady=(0, 20))

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=20,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#2c3e50",
            fg="#ecf0f1",
            insertbackground="white",
        )
        self.results_text.pack(fill="both", expand=True)
        self.results_text.config(state="disabled")

    def create_training_section(self, parent: tk.Widget) -> None:
        """Create model training controls."""
        training_frame = ttk.Frame(parent)
        training_frame.pack(fill="x")

        self.train_btn = ttk.Button(
            training_frame,
            text="Train New Model",
            command=self.train_model,
            style="Success.TButton",
        )
        self.train_btn.pack(side="left", padx=(0, 10))

        self.progress = ttk.Progressbar(training_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True)

    def on_deck_update(self, current_cards: List[Dict]) -> None:
        """Enable predict button when deck is complete."""
        state = "normal" if len(current_cards) == 8 else "disabled"
        self.drag_predict_btn.config(state=state)

    def predict_from_drag_drop(self) -> None:
        """Predict from drag-drop deck."""
        if not self.predictor:
            messagebox.showerror(
                "Error", "Model not loaded. Please train a model first."
            )
            return

        card_names = self.drag_drop_gui.get_deck_card_names()

        if len(card_names) != 8:
            messagebox.showwarning(
                "Incomplete Deck", "Please add exactly 8 cards to your deck."
            )
            return

        self.current_deck_card_ids = [
            cid for name in card_names if (cid := find_card_id_by_name(name))
        ]

        self.notebook.select(2)
        self.show_loading_message()

        threading.Thread(
            target=self._predict_from_names_thread, args=(card_names,), daemon=True
        ).start()  # type: ignore

    def clear_text_fields(self) -> None:
        """Clear all text input fields."""
        for entry in self.card_entries:
            entry.delete(0, tk.END)
        self.url_entry.delete(0, tk.END)
        self.ids_entry.delete(0, tk.END)

    def on_method_change(self) -> None:
        """Show/hide input sections based on selection."""
        for frame in [self.names_frame, self.url_frame, self.ids_frame]:
            frame.pack_forget()

        method = self.input_method.get()
        if method == "names":
            self.names_frame.pack(fill="x", pady=(10, 0))
        elif method == "url":
            self.url_frame.pack(fill="x", pady=(10, 0))
        elif method == "ids":
            self.ids_frame.pack(fill="x", pady=(10, 0))

    def fill_example_deck(self) -> None:
        """Fill fields with example deck."""
        example_cards = [
            "MegaKnight",
            "PEKKA",
            "Bandit",
            "Royal Ghost",
            "eWiz",
            "Zap",
            "Poison",
            "Tornado",
        ]

        for i, card_name in enumerate(example_cards[: len(self.card_entries)]):
            self.card_entries[i].delete(0, tk.END)
            self.card_entries[i].insert(0, card_name)

    def load_model_background(self) -> None:
        """Load model in background thread."""

        def load_model() -> None:
            try:
                self.predictor = EnhancedQuickClashPredictor(
                    "clash_royale_classifier.pth"
                )
                self.root.after(
                    0, self.on_model_loaded, True, "Model loaded successfully!"
                )
            except FileNotFoundError:
                self.root.after(
                    0,
                    self.on_model_loaded,
                    False,
                    "Model file not found. Train a model first.",
                )
            except Exception as e:
                self.root.after(
                    0, self.on_model_loaded, False, f"Error loading model: {str(e)}"
                )

        threading.Thread(target=load_model, daemon=True).start()  # type: ignore

    def on_model_loaded(self, success: bool, message: str) -> None:
        """Handle model loading completion."""
        self.load_model_attempted = True
        self.status_label.config(text=message)

        state = "normal" if success else "disabled"
        self.predict_btn.config(state=state)
        self.drag_predict_btn.config(state=state)

    def predict_archetype(self) -> None:
        """Predict based on selected input method."""
        if not self.predictor:
            messagebox.showerror(
                "Error", "Model not loaded. Please train a model first."
            )
            return

        method = self.input_method.get()

        try:
            if method == "names":
                self.predict_from_names()
            elif method == "url":
                self.predict_from_url()
            elif method == "ids":
                self.predict_from_ids()
        except Exception as e:
            messagebox.showerror(
                "Prediction Error", f"Error during prediction: {str(e)}"
            )

    def predict_from_names(self) -> None:
        """Predict from names; remove duplicates."""
        raw_names = [e.get().strip() for e in self.card_entries if e.get().strip()]
        unique_names = list(dict.fromkeys([n.lower() for n in raw_names]))

        if len(unique_names) != len(raw_names):
            for i, entry in enumerate(self.card_entries):
                entry.delete(0, tk.END)
                if i < len(unique_names):
                    entry.insert(0, unique_names[i].title())
            messagebox.showinfo(
                "Duplicates Removed",
                "Duplicate card names were removed from the input fields.",
            )

        if len(unique_names) != 8:
            messagebox.showwarning(
                "Input Error", "Please enter exactly 8 unique card names."
            )
            return

        self.notebook.select(2)
        self.show_loading_message()
        threading.Thread(
            target=self._predict_from_names_thread,
            args=([n.title() for n in unique_names],),
            daemon=True,
        ).start()  # type: ignore

    def _predict_from_names_thread(self, card_names: List[str]) -> None:
        """Background prediction from names."""
        try:
            self.current_deck_card_ids = [
                cid for name in card_names if (cid := find_card_id_by_name(name))
            ]

            # Suppress console output by temporarily redirecting stdout
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                result = self.predictor.predict_from_card_names(card_names)
            finally:
                sys.stdout = old_stdout

            self.root.after(0, self.display_prediction_result, result)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def predict_from_url(self) -> None:
        """Predict from URL; remove duplicates."""
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Input Error", "Please enter a deck URL.")
            return

        self.notebook.select(2)
        self.show_loading_message()

        threading.Thread(
            target=self._predict_from_url_thread, args=(url,), daemon=True
        ).start()  # type: ignore

    def predict_from_ids(self) -> None:
        """Predict from IDs; remove duplicates."""
        ids_text = self.ids_entry.get().strip()
        if not ids_text:
            messagebox.showwarning("Input Error", "Please enter card IDs.")
            return

        parts = [p.strip() for p in ids_text.split(";") if p.strip()]
        unique_ids = list(dict.fromkeys(parts))

        if len(unique_ids) != len(parts):
            self.ids_entry.delete(0, tk.END)
            self.ids_entry.insert(0, ";".join(unique_ids))
            messagebox.showinfo(
                "Duplicates Removed",
                "Duplicate card IDs were removed from the input field.",
            )

        if len(unique_ids) != 8:
            messagebox.showwarning(
                "Input Error", "Please enter exactly 8 unique card IDs."
            )
            return

        self.notebook.select(2)
        self.show_loading_message()
        threading.Thread(
            target=self._predict_from_ids_thread,
            args=(";".join(unique_ids),),
            daemon=True,
        ).start()  # type: ignore

    def _predict_from_url_thread(self, url: str) -> None:
        """Background prediction from URL."""
        try:
            deck = self.predictor.trainer.processor.extract_deck_from_url(url)
            deduped_deck = list(dict.fromkeys(deck))

            if len(deduped_deck) != len(deck):
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Duplicates Removed",
                        "Duplicate card IDs found in URL were removed before prediction.",
                    ),
                )

            if len(deduped_deck) != 8:
                self.root.after(
                    0,
                    lambda: messagebox.showwarning(
                        "Input Error",
                        "Deck must contain exactly 8 unique cards after removing duplicates.",
                    ),
                )
                return

            self.current_deck_card_ids = deduped_deck
            result = self.predictor.predict_from_deck_string(
                ";".join(map(str, deduped_deck))
            )
            self.root.after(0, self.display_prediction_result, result)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def _predict_from_ids_thread(self, ids_text: str) -> None:
        """Background prediction from IDs."""
        try:
            deck = self.predictor.trainer.processor.extract_from_deck_string(ids_text)
            self.current_deck_card_ids = deck

            result = self.predictor.predict_from_deck_string(ids_text)
            self.root.after(0, self.display_prediction_result, result)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def show_loading_message(self) -> None:
        """Display loading message."""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Analyzing deck... Please wait...")
        self.results_text.config(state="disabled")

    def display_prediction_result(self, result: Dict) -> None:
        """Display prediction results."""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)

        if "error" in result:
            self.results_text.insert(tk.END, f"Error: {result['error']}")
        else:
            deck = self.current_deck_card_ids or []
            if "deck_stats" not in result and deck:
                result["deck_stats"] = calculate_deck_stats(deck)

            rarity_names = {
                1: "Common",
                2: "Rare",
                3: "Epic",
                4: "Legendary",
                5: "Champion",
            }

            output = (
                f"Archetype: {result['archetype']}\n"
                f"Confidence: {result['confidence']:.2%}\n\n"
            )

            if "deck_stats" in result:
                stats = result["deck_stats"]
                output += (
                    f"Average Elixir Cost: {stats['average_elixir']:.2f}\n"
                    f"4-Card Cycle Cost: {stats['four_card_cycle']}\n"
                    f"Total Deck Cost: {stats['total_elixir']}\n\n"
                )

            card_types = result.get("card_types", {})
            output += f"Card Types: {card_types}\n\n"

            output += "Deck Composition:\n" + "-" * 40 + "\n"

            if "deck_stats" in result and "card_details" in result["deck_stats"]:
                for i, card in enumerate(result["deck_stats"]["card_details"], 1):
                    output += (
                        f"{i}. {card['name']} ({card['elixir']} elixir) - "
                        f"{card['type'].title()} - {rarity_names[card['rarity']]}\n"
                    )
            elif deck:
                for i, card_id in enumerate(deck, 1):
                    card_info = get_card_info(card_id)
                    output += (
                        f"{i}. {card_info['name']} ({card_info['elixir']} elixir) - "
                        f"{card_info['type'].title()} - {rarity_names[card_info['rarity']]}\n"
                    )

            output += "\nAll Archetype Probabilities:\n" + "-" * 30 + "\n"
            all_probs = result.get("all_probabilities", {})
            for arch, prob in sorted(
                all_probs.items(), key=lambda x: x[1], reverse=True
            ):
                output += f"  {arch}: {prob:.2%}\n"

            self.results_text.insert(tk.END, output)
            self.update_deck_overview(all_probs, deck)

        self.results_text.config(state="disabled")

    def display_error(self, error_message: str) -> None:
        """Display error message."""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"❌ ERROR:\n{error_message}")
        self.results_text.config(state="disabled")

    def train_model(self) -> None:
        """Train new model after confirmation."""
        if not messagebox.askyesno(
            "Confirm Training",
            "Training a new model may take several minutes. Continue?",
        ):
            return

        self.predict_btn.config(state="disabled")
        self.drag_predict_btn.config(state="disabled")
        self.train_btn.config(state="disabled")
        self.progress.start()
        self.status_label.config(text="Training model...")

        threading.Thread(target=self._train_model_thread, daemon=True).start()  # type: ignore

    def _train_model_thread(self) -> None:
        """Background model training."""
        try:
            train_new_model()
            self.root.after(
                0, self.on_training_completed, True, "Training completed successfully!"
            )
        except Exception as e:
            self.root.after(
                0, self.on_training_completed, False, f"Training failed: {str(e)}"
            )

    def on_training_completed(self, success: bool, message: str) -> None:
        """Handle training completion."""
        self.progress.stop()
        self.train_btn.config(state="normal")
        self.status_label.config(text=message)

        if success:
            self.load_model_background()
            messagebox.showinfo("Training Complete", message)
        else:
            messagebox.showerror("Training Failed", message)


def main() -> None:
    """Run the GUI application."""
    try:
        root = tk.Tk()
        app = ClashRoyaleGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")


if __name__ == "__main__":
    main()
