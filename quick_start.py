#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Start Guide - Train in 3 steps
"""

import torch
import numpy as np
from advanced_audio_classifier import (
    AudioConfig, TrainingConfig, 
    MultiDatasetAudioDataset, ConvNeXtAudio,
    AdvancedTrainer
)

def quick_train():
    """Quick training start"""
    
    # 1. Configuration
    print("[1/3] Initializing configuration...")
    audio_config = AudioConfig()
    train_config = TrainingConfig(
        batch_size=16,      # Small batch for memory saving
        num_epochs=30,      # Quick test
        learning_rate=1e-3
    )
    
    # 2. Load data
    print("[2/3] Loading data...")
    train_dataset = MultiDatasetAudioDataset(
        urbansound_path='./UrbanSound8K',
        fold=1,
        config=audio_config,
        is_training=True
    )
    
    # 3. Train model
    print("[3/3] Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ConvNeXt model (fast and efficient)
    model = ConvNeXtAudio(num_classes=10, pretrained=True)
    
    # Dataloader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Training
    trainer = AdvancedTrainer(model, train_config, device)
    
    print("[START] Training started!")
    print("   Expected time: 30-60 min (with GPU)")
    
    # Simple training loop
    for epoch in range(5):  # Test with 5 epochs only
        print(f"\nEpoch {epoch+1}/5")
        loss, acc = trainer.train_epoch(train_loader)
        print(f"Loss: {loss:.4f}, Acc: {acc:.2f}%")
    
    print("\n[SUCCESS] Quick test completed!")
    print("For full training, run main() in advanced_audio_classifier.py")

if __name__ == '__main__':
    print("="*60)
    print("VEHICLE SOUND SPECIALIZED AUDIO CLASSIFIER - QUICK START")
    print("="*60)
    
    quick_train()
