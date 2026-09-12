import random
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset

from atomic.tokenizer import AtomicTokenizer

def generate_arithmetic_reasoning() -> str:
    """Generate multi-step arithmetic word problems with Chain-of-Thought."""
    op = random.choice(["add_sub", "mul_add", "shopping", "speed_dist", "percentage"])
    
    if op == "add_sub":
        a = random.randint(15, 95)
        b = random.randint(10, 50)
        c = random.randint(5, 30)
        res = a + b - c
        return (
            f"Question: A bookstore had {a} books. They received a delivery of {b} new books, "
            f"and then sold {c} books by the evening. How many books remain in the store?\n"
            f"<think>\n"
            f"1. Start with initial inventory: {a} books.\n"
            f"2. Add received books: {a} + {b} = {a + b} books.\n"
            f"3. Subtract sold books: {a + b} - {c} = {res} books.\n"
            f"</think>\n"
            f"<answer>{res}</answer>"
        )
    elif op == "mul_add":
        items = random.randint(3, 8)
        price = random.randint(4, 12)
        extra = random.randint(5, 25)
        total = items * price + extra
        return (
            f"Question: Alex bought {items} notebooks at ${price} each, plus a backpack for ${extra}. "
            f"What was the total cost of the purchase?\n"
            f"<think>\n"
            f"1. Calculate the cost of the notebooks: {items} * ${price} = ${items * price}.\n"
            f"2. Add the backpack cost: ${items * price} + ${extra} = ${total}.\n"
            f"</think>\n"
            f"<answer>${total}</answer>"
        )
    elif op == "shopping":
        budget = random.randint(80, 150)
        item1 = random.randint(20, 40)
        item2 = random.randint(15, 35)
        rem = budget - item1 - item2
        return (
            f"Question: Maria started with ${budget}. She spent ${item1} on groceries and ${item2} on gas. "
            f"How much money does she have left?\n"
            f"<think>\n"
            f"1. Total amount spent: ${item1} + ${item2} = ${item1 + item2}.\n"
            f"2. Subtract total expenditure from starting budget: ${budget} - ${item1 + item2} = ${rem}.\n"
            f"</think>\n"
            f"<answer>${rem}</answer>"
        )
    elif op == "speed_dist":
        speed = random.choice([40, 50, 60, 70, 80])
        hours = random.choice([2, 3, 4, 5])
        dist = speed * hours
        return (
            f"Question: A train travels at an average speed of {speed} km/h for {hours} hours. "
            f"What is the total distance covered?\n"
            f"<think>\n"
            f"1. Formula for distance is Distance = Speed * Time.\n"
            f"2. Multiply speed by duration: {speed} * {hours} = {dist} km.\n"
            f"</think>\n"
            f"<answer>{dist} km</answer>"
        )
    else:
        orig = random.choice([50, 80, 100, 120, 200])
        pct = random.choice([10, 20, 25, 50])
        discount = int(orig * pct / 100)
        final_p = orig - discount
        return (
            f"Question: A jacket originally priced at ${orig} is on sale with a {pct}% discount. "
            f"What is the discounted price?\n"
            f"<think>\n"
            f"1. Calculate discount amount: {pct}% of ${orig} = ({pct}/100) * ${orig} = ${discount}.\n"
            f"2. Subtract discount from original price: ${orig} - ${discount} = ${final_p}.\n"
            f"</think>\n"
            f"<answer>${final_p}</answer>"
        )

def generate_logic_reasoning() -> str:
    """Generate formal logic, syllogisms, and deductive reasoning."""
    types = ["transitive", "modus_ponens", "modus_tollens", "quantifier"]
    t = random.choice(types)

    if t == "transitive":
        names = ["Alice", "Bob", "Charlie", "David", "Emma"]
        random.shuffle(names)
        n1, n2, n3 = names[0], names[1], names[2]
        attr = random.choice(["taller than", "faster than", "older than", "stronger than"])
        return (
            f"Question: Premise 1: {n1} is {attr} {n2}. Premise 2: {n2} is {attr} {n3}. "
            f"Is {n1} {attr} {n3}?\n"
            f"<think>\n"
            f"1. The relation '{attr}' is transitive: if X > Y and Y > Z, then X > Z.\n"
            f"2. We are given {n1} > {n2} and {n2} > {n3}.\n"
            f"3. Applying the transitive law yields {n1} > {n3}.\n"
            f"</think>\n"
            f"<answer>Yes</answer>"
        )
    elif t == "modus_ponens":
        entities = [
            ("it rains", "the ground is wet"),
            ("the temperature drops below 0", "water freezes"),
            ("the code has a syntax error", "the compiler fails"),
            ("the battery is completely drained", "the device turns off")
        ]
        p, q = random.choice(entities)
        return (
            f"Question: Rule: If {p}, then {q}. Fact: {p.capitalize()}. What follows?\n"
            f"<think>\n"
            f"1. We have a conditional statement: P -> Q.\n"
            f"2. The antecedent P ('{p}') is affirmed to be true.\n"
            f"3. By the Modus Ponens rule of inference, Q must follow.\n"
            f"</think>\n"
            f"<answer>{q.capitalize()}</answer>"
        )
    elif t == "modus_tollens":
        entities = [
            ("it is raining", "the grass is wet"),
            ("the switch is on", "the light glows"),
            ("a number is divisible by 4", "it is an even number")
        ]
        p, q = random.choice(entities)
        return (
            f"Question: Rule: If {p}, then {q}. Fact: It is not true that {q}. Can we conclude whether {p}?\n"
            f"<think>\n"
            f"1. The rule is P -> Q. The contrapositive is not Q -> not P.\n"
            f"2. We are given not Q ('{q}' is false).\n"
            f"3. By Modus Tollens, we deduce not P ('{p}' is false).\n"
            f"</think>\n"
            f"<answer>No, {p} is false</answer>"
        )
    else:
        # Quantifier
        group = random.choice(["mammals", "birds", "reptiles", "metals", "insects"])
        prop = {
            "mammals": ("warm-blooded", "Dolphins", "mammals", "warm-blooded"),
            "birds": ("feathered", "Penguins", "birds", "feathered"),
            "reptiles": ("cold-blooded", "Iguanas", "reptiles", "cold-blooded"),
            "metals": ("conductors of electricity", "Copper", "metals", "a conductor of electricity"),
            "insects": ("six-legged", "Ants", "insects", "six-legged")
        }[group]
        return (
            f"Question: All {prop[2]} are {prop[0]}. {prop[1]} are {prop[2]}. "
            f"Are {prop[1]} {prop[3]}?\n"
            f"<think>\n"
            f"1. Universal premise: For all x in {prop[2]}, x has property '{prop[0]}'.\n"
            f"2. Particular premise: {prop[1]} belong to the set of {prop[2]}.\n"
            f"3. Universal instantiation dictates that {prop[1]} must be {prop[3]}.\n"
            f"</think>\n"
            f"<answer>Yes</answer>"
        )

def generate_algorithmic_reasoning() -> str:
    """Generate algorithmic execution, stack simulation, and sequence puzzles."""
    t = random.choice(["reverse", "parity", "min_max", "pattern"])

    if t == "reverse":
        length = random.randint(3, 5)
        nums = [random.randint(1, 20) for _ in range(length)]
        reversed_nums = list(reversed(nums))
        return (
            f"Question: Reverse the following ordered list: {nums}.\n"
            f"<think>\n"
            f"1. Scan the sequence from rightmost element to leftmost.\n"
            f"2. The reverse ordering places {nums[-1]} first and {nums[0]} last.\n"
            f"3. Reading backwards yields: {reversed_nums}.\n"
            f"</think>\n"
            f"<answer>{reversed_nums}</answer>"
        )
    elif t == "parity":
        nums = [random.randint(1, 30) for _ in range(random.randint(3, 5))]
        evens = [x for x in nums if x % 2 == 0]
        return (
            f"Question: Count how many even numbers are in the list: {nums}.\n"
            f"<think>\n"
            f"1. Check divisibility by 2 for each element in {nums}:\n"
            + "".join([f"   - {x}: {'Even' if x % 2 == 0 else 'Odd'}\n" for x in nums]) +
            f"2. The even numbers found are {evens}, giving a total count of {len(evens)}.\n"
            f"</think>\n"
            f"<answer>{len(evens)}</answer>"
        )
    elif t == "min_max":
        nums = [random.randint(5, 99) for _ in range(4)]
        m = min(nums)
        return (
            f"Question: Find the minimum value in the list: {nums}.\n"
            f"<think>\n"
            f"1. Compare each number sequentially against the current minimum.\n"
            f"2. Comparing values {nums}, the smallest value is {m}.\n"
            f"</think>\n"
            f"<answer>{m}</answer>"
        )
    else:
        start = random.randint(2, 10)
        step = random.randint(2, 5)
        seq = [start + i * step for i in range(4)]
        next_val = start + 4 * step
        return (
            f"Question: Find the next number in the arithmetic progression: {seq}, ?\n"
            f"<think>\n"
            f"1. Compute common difference: {seq[1]} - {seq[0]} = {step}.\n"
            f"2. Verify consistency: {seq[2]} - {seq[1]} = {step} and {seq[3]} - {seq[2]} = {step}.\n"
            f"3. Add difference {step} to the last term {seq[-1]}: {seq[-1]} + {step} = {next_val}.\n"
            f"</think>\n"
            f"<answer>{next_val}</answer>"
        )

def build_reasoning_corpus(num_samples: int = 3000) -> List[str]:
    """Build a comprehensive balanced reasoning corpus."""
    generators = [
        generate_arithmetic_reasoning,
        generate_logic_reasoning,
        generate_algorithmic_reasoning
    ]
    corpus = []
    for _ in range(num_samples):
        gen = random.choice(generators)
        corpus.append(gen())
    return corpus


class ReasoningDataset(Dataset):
    """PyTorch Dataset for Reasoning LM with proper causal LM masking."""
    def __init__(
        self,
        samples: List[str],
        tokenizer: AtomicTokenizer,
        max_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for sample in samples:
            # Append EOS token to delimit generation
            full_text = sample + "<eos>"
            tokens = tokenizer.encode(full_text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            # Causal LM input is tokens[:-1], target is tokens[1:]
            if len(tokens) >= 4:
                self.data.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        return seq[:-1], seq[1:]


def collate_reasoning_fn(batch, pad_id: int):
    """Collate variable length sequences with right padding."""
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    padded_inputs = torch.full((len(inputs), max_len), pad_id, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), -100, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :inp.size(0)] = inp
        padded_targets[i, :tgt.size(0)] = tgt

    return padded_inputs, padded_targets
