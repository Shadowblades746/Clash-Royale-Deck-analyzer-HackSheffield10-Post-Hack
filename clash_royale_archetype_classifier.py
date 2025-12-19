import re
from typing import List, Dict, Set, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim

import cards
import training_data

# Card database
CLASH_ROYALE_CARDS = cards.cards

# Reverse mapping for name to ID lookup
CARD_NAME_TO_ID = {
    info["name"].lower(): card_id for card_id, info in CLASH_ROYALE_CARDS.items()
}


def get_card_info(card_id: int) -> Dict:
    """Get card information by ID."""
    return CLASH_ROYALE_CARDS.get(
        card_id, {"name": "Unknown", "elixir": 0, "rarity": 0, "type": "unknown"}
    )


def find_card_id_by_name(card_name: str) -> Optional[int]:
    """Find card ID by name (case-insensitive, partial match)."""
    card_name_lower = card_name.lower()

    # Exact match first
    if card_name_lower in CARD_NAME_TO_ID:
        return CARD_NAME_TO_ID[card_name_lower]

    # Partial match
    for name, card_id in CARD_NAME_TO_ID.items():
        if card_name_lower in name or name in card_name_lower:
            return card_id

    return None


def calculate_deck_stats(deck: List[int]) -> Dict:
    """Calculate comprehensive deck statistics."""
    total_elixir = 0
    card_details = []

    for card_id in deck:
        card_info = get_card_info(card_id)
        total_elixir += card_info["elixir"]
        card_details.append(
            {
                "id": card_id,
                "name": card_info["name"],
                "elixir": card_info["elixir"],
                "type": card_info["type"],
                "rarity": card_info["rarity"],
            }
        )

    avg_elixir = total_elixir / 8
    sorted_cards = sorted(card_details, key=lambda x: x["elixir"])
    four_card_cycle = sum(card["elixir"] for card in sorted_cards[:4])

    return {
        "average_elixir": avg_elixir,
        "four_card_cycle": four_card_cycle,
        "total_elixir": total_elixir,
        "card_details": card_details,
    }


def display_deck_analysis(deck: List[int], prediction_result: Dict) -> None:
    """Display comprehensive deck analysis."""
    stats = calculate_deck_stats(deck)

    print("\n" + "=" * 50)
    print("DECK ANALYSIS")
    print("=" * 50)

    print(f"\nArchetype: {prediction_result['archetype']}")
    print(f"Confidence: {prediction_result['confidence']:.2%}")

    print(f"\nAverage Elixir Cost: {stats['average_elixir']:.2f}")
    print(f"4-Card Cycle Cost: {stats['four_card_cycle']}")
    print(f"Total Deck Cost: {stats['total_elixir']}")

    print(f"\nCard Types: {prediction_result['card_types']}")

    print("\nDeck Composition:")
    print("-" * 40)
    rarity_names = {1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Champion"}
    for i, card in enumerate(stats["card_details"], 1):
        print(
            f"{i}. {card['name']} ({card['elixir']} elixir) - "
            f"{card['type'].title()} - {rarity_names[card['rarity']]}"
        )

    print("\nAll Archetype Probabilities:")
    print("-" * 30)
    for arch, prob in sorted(
        prediction_result["all_probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {arch}: {prob:.2%}")


def get_card_type_distribution(deck: List[int]) -> torch.Tensor:
    """Calculate normalized distribution of card types."""
    distribution = torch.zeros(3)  # [troops, spells, buildings]

    for card_id in deck:
        prefix = int(str(card_id)[:2])
        if prefix == 26:
            distribution[0] += 1  # troop
        elif prefix == 27:
            distribution[2] += 1  # building
        elif prefix == 28:
            distribution[1] += 1  # spell
        else:
            print(f"Warning: Unknown card prefix {prefix} for ID {card_id}")

    return distribution / 8.0


def get_card_id_type(card_id: int) -> str:
    """Determine card type from its ID prefix."""
    prefix = int(str(card_id)[:2])
    type_map = {26: "troop", 27: "building", 28: "spell"}
    return type_map.get(prefix, "unknown")


class ClashRoyaleDataProcessor:
    """Process and transform Clash Royale deck data."""

    def __init__(self) -> None:
        self.card_types = {"troop": 26, "building": 27, "spell": 28}
        self.archetypes = [
            "beatdown",
            "control",
            "siege",
            "bridge_spam",
            "cycle",
            "bait",
            "split_lane",
        ]
        self.archetype_to_idx = {arch: i for i, arch in enumerate(self.archetypes)}
        self.idx_to_archetype = {i: arch for i, arch in enumerate(self.archetypes)}

        self.all_card_ids: Set[int] = set()
        self.card_id_to_index: Dict[int, int] = {}
        self.index_to_card_id: Dict[int, int] = {}

    def extract_deck_from_url(self, url: str) -> List[int]:
        """Extract card IDs from various Clash Royale deck URLs."""
        try:
            # Clash Royale deep link format
            if "clashroyale://copyDeck" in url:
                deck_match = re.search(r"deck=([^&]+)", url)
                if deck_match:
                    deck_string = deck_match.group(1)
                    card_ids = [int(cid) for cid in deck_string.split(";")]
                    deck = card_ids[:8]
                else:
                    return []

            # RoyaleAPI format
            elif "royaleapi.com" in url and "/deck/" in url:
                api_url = url if url.endswith(".json") else f"{url}.json"
                response_ = requests.get(api_url, timeout=10)
                data = response_.json()
                deck = [card["id"] for card in data.get("cards", [])]

                if len(deck) != 8:
                    response_ = requests.get(url, timeout=10)
                    card_ids = re.findall(r'"id":(\d{8})', response_.text)
                    deck = [int(cid) for cid in card_ids[:8]]

            # DeckBandit format
            elif "deckbandit" in url:
                response_ = requests.get(url, timeout=10)
                card_ids = re.findall(r'"id":(\d{8})', response_.text)
                deck = [int(cid) for cid in card_ids[:8]]

            # Generic fallback
            else:
                response_ = requests.get(url, timeout=10)
                card_ids = re.findall(r"\b(26\d{6}|27\d{6}|28\d{6})\b", response_.text)
                deck = [int(cid) for cid in card_ids[:8]]

            if len(deck) != 8:
                print(f"Warning: Got {len(deck)} cards instead of 8 from {url}")
                return []

            for card_id in deck:
                self.all_card_ids.add(card_id)

            return deck

        except Exception as e_:
            print(f"Error extracting deck from {url}: {e_}")
            return []

    def extract_from_deck_string(self, deck_string: str) -> List[int]:
        """Extract card IDs from semicolon-separated string."""
        try:
            card_ids = [int(cid.strip()) for cid in deck_string.split(";")]

            if len(card_ids) != 8:
                print(f"Warning: Deck string has {len(card_ids)} cards, expected 8")
                return []

            for card_id in card_ids:
                self.all_card_ids.add(card_id)

            return card_ids

        except (ValueError, AttributeError) as e_:
            print(f"Error parsing deck string: {e_}")
            return []

    def build_card_mapping(self) -> None:
        """Build bidirectional mapping between card IDs and indices."""
        for idx, card_id in enumerate(sorted(self.all_card_ids)):
            self.card_id_to_index[card_id] = idx
            self.index_to_card_id[idx] = card_id

        print(f"Built mapping for {len(self.card_id_to_index)} unique cards")

    def deck_to_vector(self, deck: List[int]) -> torch.Tensor:
        """Convert deck to one-hot encoded vector."""
        if not self.card_id_to_index:
            self.build_card_mapping()

        vector = torch.zeros(len(self.card_id_to_index))

        for card_id in deck:
            if card_id in self.card_id_to_index:
                vector[self.card_id_to_index[card_id]] = 1
            else:
                print(f"Warning: Unknown card ID {card_id}")

        return vector


class DynamicDeckClassifier(nn.Module):
    """Neural network for deck archetype classification."""

    def __init__(
        self,
        card_vocab_size: int = 121,
        type_input_size: int = 3,
        hidden_size: int = 256,
        num_classes: int = 7,
    ) -> None:
        super().__init__()

        self.card_branch = nn.Sequential(
            nn.Linear(card_vocab_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

        self.type_branch = nn.Sequential(
            nn.Linear(type_input_size, hidden_size // 4), nn.ReLU()
        )

        combined_size = hidden_size // 2 + hidden_size // 4

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(
        self, card_features: torch.Tensor, type_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the network."""
        card_out = self.card_branch(card_features)
        type_out = self.type_branch(type_features)
        combined = torch.cat([card_out, type_out], dim=1)
        return self.classifier(combined)


class ArchetypeTrainer:
    """Train and manage deck archetype classification models."""

    def __init__(self) -> None:
        self.optimizer = None
        self.y = None
        self.X_types = None
        self.scheduler = None
        self.X_cards = None
        self.processor = ClashRoyaleDataProcessor()
        self.model: Optional[DynamicDeckClassifier] = None
        self.criterion = nn.CrossEntropyLoss()
        self.deck_vectors: List[List[int]] = []
        self.type_vectors: List[torch.Tensor] = []
        self.labels: List[int] = []

    def load_training_data(self, deck_data: List[Dict]) -> None:
        """Load and process training data."""
        print("Extracting card IDs from training data...")
        successful_decks = 0

        for i, data in enumerate(deck_data):
            if i % 10 == 0:
                print(f"Processed {i}/{len(deck_data)} decks...")

            archetype = data.get("archetype", "")
            if not archetype:
                print(f"Warning: No archetype for deck {i}")
                continue

            deck = None
            if url := data.get("url"):
                deck = self.processor.extract_deck_from_url(url)
            elif deck_string := data.get("deck_string"):
                deck = self.processor.extract_from_deck_string(deck_string)

            if deck and len(deck) == 8:
                self.deck_vectors.append(deck)
                type_vector = get_card_type_distribution(deck)
                label_idx = self.processor.archetype_to_idx[archetype]

                self.type_vectors.append(type_vector)
                self.labels.append(label_idx)
                successful_decks += 1
            else:
                print(f"Warning: Could not extract valid deck from item {i}")

        if successful_decks == 0:
            raise ValueError("No valid decks could be extracted from training data")

        self.processor.build_card_mapping()

        print("Converting decks to feature vectors...")
        deck_feature_vectors = [
            self.processor.deck_to_vector(deck) for deck in self.deck_vectors
        ]

        self.X_cards = torch.stack(deck_feature_vectors)
        self.X_types = torch.stack(self.type_vectors)
        self.y = torch.tensor(self.labels, dtype=torch.long)

        vocab_size = len(self.processor.card_id_to_index)
        self.model = DynamicDeckClassifier(
            card_vocab_size=vocab_size, num_classes=len(self.processor.archetypes)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

        print(f"Successfully loaded {successful_decks} training examples")
        print(f"Vocabulary size: {vocab_size} unique cards")

    def train(
        self, epochs: int = 100, validation_split: float = 0.2
    ) -> Tuple[List[float], List[float]]:
        """Train the classification model."""
        if self.model is None:
            raise ValueError("Must load training data first")

        dataset_size = len(self.X_cards)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_dataset = torch.utils.data.TensorDataset(
            self.X_cards[train_indices],
            self.X_types[train_indices],
            self.y[train_indices],
        )
        val_dataset = torch.utils.data.TensorDataset(
            self.X_cards[val_indices], self.X_types[val_indices], self.y[val_indices]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch_cards, batch_types, batch_labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_cards, batch_types)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.model.eval()
            correct = total = 0

            with torch.no_grad():
                for batch_cards, batch_types, batch_labels in val_loader:
                    outputs = self.model(batch_cards, batch_types)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            train_losses.append(epoch_loss / len(train_loader))

            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], "
                    f"Loss: {epoch_loss / len(train_loader):.4f}, "
                    f"Val Accuracy: {accuracy:.2f}%"
                )

        self.scheduler.step()
        return train_losses, val_accuracies

    def predict_deck(self, deck: List[int]) -> Dict:
        """Predict archetype probabilities for a deck."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()

        with torch.no_grad():
            deck_vector = self.processor.deck_to_vector(deck).unsqueeze(0)
            type_vector = get_card_type_distribution(deck).unsqueeze(0)

            outputs = self.model(deck_vector, type_vector)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            archetype = self.processor.idx_to_archetype[predicted.item()]

            all_probs = {
                arch: probabilities[0][i].item()
                for i, arch in self.processor.idx_to_archetype.items()
            }

            return {
                "archetype": archetype,
                "confidence": confidence.item(),
                "all_probabilities": all_probs,
                "card_types": {
                    "troops": sum(
                        1 for card in deck if get_card_id_type(card) == "troop"
                    ),
                    "spells": sum(
                        1 for card in deck if get_card_id_type(card) == "spell"
                    ),
                    "buildings": sum(
                        1 for card in deck if get_card_id_type(card) == "building"
                    ),
                },
            }

    def save_model(self, filepath: str) -> None:
        """Save trained model and processor state."""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "processor_state": {
                    "card_id_to_index": self.processor.card_id_to_index,
                    "index_to_card_id": self.processor.index_to_card_id,
                    "all_card_ids": list(self.processor.all_card_ids),
                    "archetypes": self.processor.archetypes,
                    "archetype_to_idx": self.processor.archetype_to_idx,
                },
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model and processor state."""
        checkpoint = torch.load(filepath)

        if "processor_state" in checkpoint:
            state = checkpoint["processor_state"]
            self.processor.card_id_to_index = state["card_id_to_index"]
            self.processor.index_to_card_id = state["index_to_card_id"]
            self.processor.all_card_ids = set(state["all_card_ids"])
            self.processor.archetypes = state["archetypes"]
            self.processor.archetype_to_idx = state["archetype_to_idx"]
            self.processor.idx_to_archetype = {
                v: k for k, v in self.processor.archetype_to_idx.items()
            }

        vocab_size = len(self.processor.card_id_to_index)
        self.model = DynamicDeckClassifier(
            card_vocab_size=vocab_size, num_classes=len(self.processor.archetypes)
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        print(
            f"Model loaded with {vocab_size} unique cards and "
            f"{len(self.processor.archetypes)} archetypes"
        )


class QuickClashPredictor:
    """Simple interface for making predictions with a trained model."""

    def __init__(self, model_path: str = "clash_royale_classifier.pth") -> None:
        self.trainer = ArchetypeTrainer()
        self.trainer.load_model(model_path)

    def predict_from_url(self, url: str) -> Dict:
        """Predict archetype from Clash Royale deck URL."""
        deck = self.trainer.processor.extract_deck_from_url(url)
        if not deck:
            return {"error": "Could not extract deck from URL"}
        result = self.trainer.predict_deck(deck)
        display_deck_analysis(deck, result)
        return result

    def predict_from_deck_string(self, deck_string: str) -> Dict:
        """Predict archetype from semicolon-separated deck string."""
        deck = self.trainer.processor.extract_from_deck_string(deck_string)
        if not deck:
            return {"error": "Invalid deck string"}
        result = self.trainer.predict_deck(deck)
        display_deck_analysis(deck, result)
        return result

    def predict_from_card_ids(self, card_ids: List[int]) -> Dict:
        """Predict archetype from list of card IDs."""
        if len(card_ids) != 8:
            return {"error": "Deck must contain exactly 8 cards"}
        result = self.trainer.predict_deck(card_ids)
        display_deck_analysis(card_ids, result)
        return result

    def predict_from_card_names(self, card_names: List[str]) -> Dict:
        """Predict archetype from list of card names."""
        if len(card_names) != 8:
            return {"error": "Deck must contain exactly 8 cards"}

        card_ids = []
        unknown_cards = []

        for name in card_names:
            if card_id := find_card_id_by_name(name):
                card_ids.append(card_id)
            else:
                unknown_cards.append(name)

        if unknown_cards:
            return {"error": f"Unknown cards: {', '.join(unknown_cards)}"}

        result = self.trainer.predict_deck(card_ids)
        display_deck_analysis(card_ids, result)
        return result


def get_deck_analysis_text(result: Dict) -> str:
    """Generate formatted analysis text for GUI display."""
    if "error" in result:
        return f"Error: {result['error']}"

    output = [
        "🏰 CLASH ROYALE DECK ANALYSIS 🏰",
        "=" * 50,
        "",
        f"🏷️  PRIMARY ARCHETYPE: {result['archetype'].replace('_', ' ').title()}",
        f"🎯 CONFIDENCE: {result['confidence']:.2%}",
        "",
    ]

    if "deck_stats" in result:
        stats = result["deck_stats"]
        output.append("📊 DECK STATISTICS:")
        output.append(f"   • Average Elixir Cost: {stats['average_elixir']:.2f}")
        output.append(f"   • 4-Card Cycle Cost: {stats['four_card_cycle']}")
        output.append(f"   • Total Deck Cost: {stats['total_elixir']}")
        output.append("")

    card_types = result.get("card_types", {})
    output.append("🎴 CARD TYPE DISTRIBUTION:")
    output.append(f"   • Troops: {card_types.get('troops', 0)}/8")
    output.append(f"   • Spells: {card_types.get('spells', 0)}/8")
    output.append(f"   • Buildings: {card_types.get('buildings', 0)}/8")
    output.append("")

    if "card_details" in result:
        output.append("🃏 DECK COMPOSITION:")
        rarity_names = {
            1: "Common",
            2: "Rare",
            3: "Epic",
            4: "Legendary",
            5: "Champion",
        }
        for i, card in enumerate(result["card_details"], 1):
            output.append(
                f"   {i}. {card['name']} ({card['elixir']}⏱️) - "
                f"{card['type'].title()} - {rarity_names[card['rarity']]}"
            )
        output.append("")

    output.append("📈 ALL ARCHETYPE PROBABILITIES:")
    all_probs = result.get("all_probabilities", {})
    for arch, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        arch_name = arch.replace("_", " ").title()
        bar_length = int(prob * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        output.append(f"   • {arch_name:<15} {prob:>6.2%} {bar}")

    return "\n".join(output)


class EnhancedQuickClashPredictor(QuickClashPredictor):
    """Enhanced predictor with better error handling and GUI support."""

    def predict_from_card_names_with_details(self, card_names: List[str]) -> Dict:
        """Predict archetype with detailed card information."""
        if len(card_names) != 8:
            return {"error": "Deck must contain exactly 8 cards"}

        card_ids = []
        unknown_cards = []
        card_details = []

        for name in card_names:
            if card_id := find_card_id_by_name(name):
                card_ids.append(card_id)
                card_info = get_card_info(card_id)
                card_details.append(
                    {
                        "id": card_id,
                        "name": card_info["name"],
                        "elixir": card_info["elixir"],
                        "type": card_info["type"],
                        "rarity": card_info["rarity"],
                    }
                )
            else:
                unknown_cards.append(name)

        if unknown_cards:
            return {"error": f"Unknown cards: {', '.join(unknown_cards)}"}

        result = self.trainer.predict_deck(card_ids)
        stats = calculate_deck_stats(card_ids)
        result["deck_stats"] = stats
        result["card_details"] = card_details

        return result


def train_new_model() -> None:
    """Train a new model with deck data."""
    trainer = ArchetypeTrainer()

    print("Training new model...")
    trainer.load_training_data(training_data.training_data)
    trainer.train(epochs=100)
    trainer.save_model("clash_royale_classifier.pth")
    print("Training completed! Model saved as 'clash_royale_classifier.pth'")


def predict_example() -> None:
    """Example prediction using a trained model."""
    predictor = QuickClashPredictor("clash_royale_classifier.pth")

    result = predictor.predict_from_url(
        "https://link.clashroyale.com/en/?clashroyale://copyDeck?"
        "deck=26000063;26000015;26000009;26000018;26000068;28000012;28000015;27000007"
        "&l=Royals&tt=159000000"
    )

    if "error" in result:
        print(f"Error: {result['error']}")


def interactive_deck_input() -> None:
    """Get deck input interactively from user."""
    print("\nChoose input method:")
    print("1. Enter card names (type each card name)")
    print("2. Enter deck URL")
    print("3. Enter card IDs (semicolon-separated)")

    choice = input("\nEnter your choice (1-3): ").strip()

    predictor = QuickClashPredictor("clash_royale_classifier.pth")

    if choice == "1":
        print("\nEnter 8 card names (one per line):")
        card_names = [input(f"Card {i + 1}: ").strip() for i in range(8)]

        result = predictor.predict_from_card_names(card_names)
        if "error" in result:
            print(f"Error: {result['error']}")

    elif choice == "2":
        url = input("\nEnter deck URL: ").strip()
        result = predictor.predict_from_url(url)
        if "error" in result:
            print(f"Error: {result['error']}")

    elif choice == "3":
        deck_string = input("\nEnter card IDs (semicolon-separated): ").strip()
        result = predictor.predict_from_deck_string(deck_string)
        if "error" in result:
            print(f"Error: {result['error']}")

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    print("Clash Royale Archetype Classifier")
    print("=" * 50)

    response = input(
        "Do you want to (t)rain a new model, (p)redict with existing, "
        "or (i)nteractive input? [t/p/i]: "
    ).lower()

    if response == "t":
        print("\nTraining new model...")
        print("This may take several minutes depending on your hardware...")
        train_new_model()
    elif response == "p":
        try:
            predict_example()
        except FileNotFoundError:
            print("Model file not found. Please train a new model first.")
        except Exception as e:
            print(f"Error during prediction: {e}")
    elif response == "i":
        try:
            interactive_deck_input()
        except FileNotFoundError:
            print("Model file not found. Please train a new model first.")
        except Exception as e:
            print(f"Error during interactive input: {e}")
    else:
        print("Invalid choice. Please run again and choose 't', 'p', or 'i'.")
