# app/data/product_dataset.py
"""Technical Product Dataset for Training and Evaluation.

Contains sample product descriptions and expected structured output for the summarizer.
"""

import json
import os
from typing import List, Dict, Optional


class ProductDataset:
    """Dataset manager for training/evaluating the product summarizer."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.products = self._load_from_file(data_path) if data_path else self._load_sample_data()

    def _load_sample_data(self) -> List[Dict]:
        """Provides a static sample dataset of various electronics."""
        return [
            {
                "product_id": "laptop_001",
                "product_name": "TechPro UltraBook X1",
                "category": "Laptop",
                "full_description": "The TechPro UltraBook X1 is a premium business laptop designed for professionals who demand performance and portability. It features a 14-inch 4K OLED display with 100% DCI-P3 color gamut, delivering stunning visuals for content creation and media consumption. At its heart lies the latest Intel Core i7-13700H processor with 14 cores and 20 threads, paired with 32GB of LPDDR5 RAM running at 5200MHz for seamless multitasking. Storage is handled by a blazing-fast 1TB PCIe 4.0 NVMe SSD, ensuring quick boot times and rapid file access. The discrete NVIDIA GeForce RTX 4060 graphics card with 8GB GDDR6 memory handles demanding applications and light gaming with ease. The laptop weighs just 1.3kg and measures 15.9mm at its thickest point, making it incredibly portable. Battery life is impressive at up to 12 hours of mixed usage, supported by fast charging that reaches 80% in just 60 minutes. Connectivity options include Thunderbolt 4, USB-A 3.2, HDMI 2.1, and Wi-Fi 6E. The aluminum unibody construction feels premium and durable, while the backlit keyboard offers excellent typing comfort. Security features include a fingerprint reader, TPM 2.0 chip, and IR camera for Windows Hello facial recognition. The laptop runs on Windows 11 Pro and comes with a 3-year warranty.",
                "target_summary": {
                    "product_name": "TechPro UltraBook X1",
                    "category": "Laptop",
                    "key_specs": {
                        "processor": "Intel Core i7-13700H (14-core, 20-thread)",
                        "ram": "32GB LPDDR5 5200MHz",
                        "storage": "1TB PCIe 4.0 NVMe SSD",
                        "display": "14-inch 4K OLED, 100% DCI-P3",
                        "graphics": "NVIDIA RTX 4060 8GB GDDR6",
                        "weight": "1.3kg",
                        "battery": "12 hours, 80% fast charge in 60min"
                    },
                    "pros": ["Lightweight and portable", "Excellent display quality", "Strong performance", "Long battery life"],
                    "cons": ["Premium pricing", "Limited upgradeability"],
                    "best_for": "Professionals needing portable high-performance laptop for content creation and business tasks",
                    "price_range": "Premium ($1800-2200)"
                }
            },
            {
                "product_id": "laptop_002",
                "product_name": "BudgetTech Essential 15",
                "category": "Laptop",
                "full_description": "The BudgetTech Essential 15 is an affordable laptop targeted at students and home users who need a reliable computer for everyday tasks. It sports a 15.6-inch Full HD IPS display with anti-glare coating, providing decent viewing angles and reduced eye strain during extended use. Performance comes from an AMD Ryzen 5 5500U processor with 6 cores and 12 threads, offering sufficient power for web browsing, office applications, and light multimedia work. The laptop includes 8GB of DDR4 RAM running at 3200MHz, which is adequate for basic multitasking. A 512GB SATA SSD provides reasonable storage space and much faster performance than traditional hard drives. Integrated AMD Radeon Graphics handle display output and casual tasks, though gaming and intensive graphics work are not its forte. The plastic chassis keeps costs down but feels sturdy enough for daily use, weighing in at 1.8kg. Battery life averages around 7 hours with typical usage patterns. Port selection includes USB 3.1, USB 2.0, HDMI, SD card reader, and audio jack, plus Wi-Fi 5 and Bluetooth 5.0 for wireless connectivity. The keyboard is comfortable for typing, though it lacks backlighting. The laptop comes with Windows 11 Home in S Mode and a 1-year limited warranty. While not designed for power users, it represents excellent value for money for basic computing needs.",
                "target_summary": {
                    "product_name": "BudgetTech Essential 15",
                    "category": "Laptop",
                    "key_specs": {
                        "processor": "AMD Ryzen 5 5500U (6-core, 12-thread)",
                        "ram": "8GB DDR4 3200MHz",
                        "storage": "512GB SATA SSD",
                        "display": "15.6-inch Full HD IPS",
                        "graphics": "AMD Radeon Integrated",
                        "weight": "1.8kg",
                        "battery": "7 hours typical use"
                    },
                    "pros": ["Affordable price point", "Decent performance for basic tasks", "Full HD display", "Good value for money"],
                    "cons": ["Plastic build quality", "Limited graphics capability", "No backlit keyboard"],
                    "best_for": "Students and home users needing an affordable laptop for everyday computing tasks",
                    "price_range": "Budget ($450-600)"
                }
            },
            {
                "product_id": "smartphone_001",
                "product_name": "PhoneMax Pro 14",
                "category": "Smartphone",
                "full_description": "The PhoneMax Pro 14 represents the pinnacle of smartphone technology, combining cutting-edge features with premium design. Its 6.7-inch LTPO AMOLED display supports dynamic refresh rates from 1Hz to 120Hz, with peak brightness reaching 2000 nits for excellent outdoor visibility. The screen is protected by Ceramic Shield glass and supports HDR10+ content. Under the hood, the latest A17 Pro chip built on 3nm process technology delivers unprecedented performance and efficiency, featuring a 6-core CPU, 6-core GPU, and 16-core Neural Engine for advanced AI tasks. The camera system is truly impressive with a 48MP main sensor featuring sensor-shift OIS, a 12MP ultra-wide camera with macro capability, a 12MP telephoto lens offering 3x optical zoom, and LiDAR for improved low-light autofocus. Video recording supports 4K ProRes at 60fps and 1080p at 240fps. The 12MP front camera excels at selfies and Face ID authentication. The device comes in 256GB, 512GB, and 1TB storage options with no expandability. Battery capacity is 4323mAh, providing all-day usage with support for 30W fast wired charging, 15W MagSafe wireless charging, and reverse wireless charging. Build quality is exceptional with a titanium frame and textured matte glass back. 5G connectivity, Wi-Fi 6E, Ultra Wideband, and satellite emergency SOS are all included. The phone is IP68 rated for dust and water resistance and runs the latest mobile OS with guaranteed 5 years of updates.",
                "target_summary": {
                    "product_name": "PhoneMax Pro 14",
                    "category": "Smartphone",
                    "key_specs": {
                        "display": "6.7-inch LTPO AMOLED, 1-120Hz, 2000 nits",
                        "processor": "A17 Pro 3nm (6-core CPU, 6-core GPU)",
                        "camera": "48MP main (OIS) + 12MP ultrawide + 12MP 3x telephoto + LiDAR",
                        "storage": "256GB/512GB/1TB",
                        "battery": "4323mAh, 30W wired, 15W wireless",
                        "connectivity": "5G, Wi-Fi 6E, UWB",
                        "durability": "IP68, Titanium frame"
                    },
                    "pros": ["Exceptional performance", "Professional-grade cameras", "Premium build quality", "Long software support"],
                    "cons": ["Very expensive", "No storage expansion", "Slower charging than competitors"],
                    "best_for": "Power users and professionals who want the absolute best smartphone experience",
                    "price_range": "Premium ($1099-1599)"
                }
            },
            {
                "product_id": "smartphone_002",
                "product_name": "ValuePhone A52",
                "category": "Smartphone",
                "full_description": "The ValuePhone A52 is a mid-range smartphone that balances features with affordability. It features a 6.5-inch Super AMOLED display with FHD+ resolution and 90Hz refresh rate, delivering smooth scrolling and vibrant colors that rival more expensive phones. The screen is protected by Gorilla Glass 5. Powering the device is a Qualcomm Snapdragon 778G processor built on 6nm technology, paired with 6GB or 8GB of RAM depending on configuration. This combination handles everyday apps smoothly and can manage light gaming. Storage options include 128GB or 256GB with microSD card expansion up to 1TB. The quad camera setup includes a 64MP main sensor, 12MP ultra-wide lens, 5MP macro camera, and 5MP depth sensor. While not flagship quality, photos in good lighting are quite impressive. The 32MP front camera is excellent for selfies and video calls. A substantial 4500mAh battery easily lasts a full day, supported by 25W fast charging that takes about 75 minutes for a full charge. The phone has a plastic frame with glass front and back, feeling solid despite the materials. It includes a 3.5mm headphone jack, which is increasingly rare in modern phones. Connectivity covers 5G, Wi-Fi 5, Bluetooth 5.2, and NFC. The phone is IP67 rated for water resistance and comes with Android 13 with promised 3 years of OS updates. At its price point, the ValuePhone A52 offers exceptional value.",
                "target_summary": {
                    "product_name": "ValuePhone A52",
                    "category": "Smartphone",
                    "key_specs": {
                        "display": "6.5-inch Super AMOLED FHD+, 90Hz",
                        "processor": "Snapdragon 778G 6nm",
                        "ram": "6GB/8GB",
                        "camera": "64MP main + 12MP ultrawide + 5MP macro + 5MP depth",
                        "storage": "128GB/256GB + microSD",
                        "battery": "4500mAh, 25W fast charge",
                        "durability": "IP67, Gorilla Glass 5"
                    },
                    "pros": ["Excellent display quality", "Good battery life", "Headphone jack included", "MicroSD expansion", "Affordable price"],
                    "cons": ["Plastic build", "Average camera performance", "Slower processor than flagships"],
                    "best_for": "Budget-conscious users wanting a feature-rich phone without premium pricing",
                    "price_range": "Mid-range ($399-499)"
                }
            },
            {
                "product_id": "tablet_001",
                "product_name": "ProTab Studio 12",
                "category": "Tablet",
                "full_description": "The ProTab Studio 12 is a professional-grade tablet designed for creators, designers, and business professionals. Its stunning 12.9-inch Liquid Retina XDR display uses mini-LED technology with 2732x2048 resolution, 120Hz ProMotion refresh rate, and supports P3 wide color gamut with True Tone. Peak brightness reaches 1600 nits for HDR content. The powerful M2 chip with 8-core CPU and 10-core GPU provides desktop-class performance in a thin 6.4mm body weighing just 682g. Configuration options include 8GB or 16GB of unified memory and storage from 128GB to 2TB. The 12MP wide and 10MP ultra-wide rear cameras with LiDAR scanner enable AR applications and document scanning. The landscape-oriented 12MP front camera with Center Stage is perfect for video conferencing. The tablet supports the second-generation stylus pen with pressure sensitivity, tilt recognition, and wireless charging when magnetically attached. An optional magnetic keyboard transforms it into a laptop replacement. Battery life extends up to 10 hours with typical use. Connectivity includes Thunderbolt/USB 4, 5G on cellular models, Wi-Fi 6E, and Bluetooth 5.3. Four speakers deliver impressive spatial audio. Face recognition provides secure biometric authentication. The all-screen design with slim bezels maximizes screen real estate while remaining portable.",
                "target_summary": {
                    "product_name": "ProTab Studio 12",
                    "category": "Tablet",
                    "key_specs": {
                        "display": "12.9-inch mini-LED XDR, 2732x2048, 120Hz ProMotion",
                        "processor": "M2 chip (8-core CPU, 10-core GPU)",
                        "memory": "8GB/16GB unified",
                        "storage": "128GB to 2TB",
                        "camera": "12MP wide + 10MP ultrawide + LiDAR",
                        "weight": "682g, 6.4mm thick",
                        "battery": "10 hours",
                        "connectivity": "Thunderbolt/USB 4, 5G, Wi-Fi 6E"
                    },
                    "pros": ["Outstanding display quality", "Desktop-class performance", "Excellent stylus support", "Premium build", "Great speakers"],
                    "cons": ["Very expensive", "Keyboard sold separately", "No headphone jack"],
                    "best_for": "Professional creators and power users needing a portable workstation for design and multimedia",
                    "price_range": "Premium ($1099-2399)"
                }
            },
            {
                "product_id": "monitor_001",
                "product_name": "DisplayMaster 4K Pro 32",
                "category": "Monitor",
                "full_description": "The DisplayMaster 4K Pro 32 is a professional monitor designed for content creators, photographers, and anyone requiring exceptional color accuracy. This 32-inch IPS panel delivers 4K UHD resolution (3840x2160) with 99% Adobe RGB, 100% sRGB, and 98% DCI-P3 color space coverage. Each monitor is factory calibrated with Delta E < 2 color accuracy and includes a calibration report. The 10-bit color depth displays over 1.07 billion colors for smooth gradients. Maximum brightness reaches 400 nits with HDR400 support. The 60Hz refresh rate with 5ms response time is adequate for professional work though not ideal for competitive gaming. The monitor features extensive connectivity including DisplayPort 1.4, HDMI 2.0, USB-C with 90W power delivery for single-cable laptop connection, and a USB hub with four USB 3.0 ports. The ergonomic stand offers height adjustment (150mm), tilt (-5° to 20°), swivel (±30°), and pivot for portrait orientation. Built-in hardware calibration capability with software support ensures color accuracy maintenance over time. An anti-glare matte coating reduces reflections in bright environments. The slim bezels on three sides make it excellent for multi-monitor setups. Additional features include picture-in-picture, picture-by-picture, and preset modes optimized for different workflows. The monitor comes with a 3-year warranty including coverage for uniformity issues.",
                "target_summary": {
                    "product_name": "DisplayMaster 4K Pro 32",
                    "category": "Monitor",
                    "key_specs": {
                        "display": "32-inch IPS 4K UHD (3840x2160), 60Hz",
                        "color_accuracy": "99% Adobe RGB, 100% sRGB, 98% DCI-P3, Delta E < 2",
                        "brightness": "400 nits, HDR400",
                        "connectivity": "DisplayPort 1.4, HDMI 2.0, USB-C 90W PD",
                        "ergonomics": "Height, tilt, swivel, pivot adjustable",
                        "features": "10-bit color, Factory calibrated, Hardware calibration support"
                    },
                    "pros": ["Exceptional color accuracy", "Factory calibration included", "USB-C with power delivery", "Excellent ergonomics", "3-year warranty"],
                    "cons": ["60Hz limit", "Not ideal for gaming", "Premium price"],
                    "best_for": "Professional content creators and photographers requiring accurate color reproduction",
                    "price_range": "Professional ($799-999)"
                }
            }
        ]

    def _load_from_file(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_to_file(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.products, f, indent=2, ensure_ascii=False)

    def get_product(self, product_id: str) -> Optional[Dict]:
        return next((p for p in self.products if p['product_id'] == product_id), None)

    def get_products_by_category(self, category: str) -> List[Dict]:
        return [p for p in self.products if p['category'].lower() == category.lower()]

    def get_training_pairs(self) -> List[tuple]:
        """Returns list of (full_description, target_summary) tuples for training."""
        return [(p['full_description'], p['target_summary']) for p in self.products]

    def add_product(self, product: Dict):
        self.products.append(product)

    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        return self.products[idx]


def create_default_dataset() -> ProductDataset:
    return ProductDataset()
