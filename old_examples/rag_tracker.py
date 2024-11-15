"""
rag_tracker.py
Goal: Track RAG pipeline improvements and metrics over time
"""

import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path

class RAGTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.tracking_dir = Path("rag_tracking")
        self.tracking_dir.mkdir(exist_ok=True)
        
        # Create directories for different tracking aspects
        self.metrics_dir = self.tracking_dir / "metrics"
        self.configs_dir = self.tracking_dir / "configs"
        self.metrics_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        
        # Load history if exists
        self.history_file = self.tracking_dir / "improvement_history.csv"
        if self.history_file.exists():
            self.history = pd.read_csv(self.history_file)
        else:
            self.history = pd.DataFrame(columns=[
                'timestamp', 'experiment_name', 'faithfulness', 'answer_relevancy',
                'context_precision', 'context_recall', 'retriever_k', 'model_name',
                'changes_made', 'notes'
            ])

    def log_experiment(self, metrics_df: pd.DataFrame, config: dict, notes: str = ""):
        """Log an experiment run with its metrics and configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate mean metrics
        mean_metrics = {
            'faithfulness': metrics_df['faithfulness'].mean(),
            'answer_relevancy': metrics_df['answer_relevancy'].mean(),
            'context_precision': metrics_df['context_precision'].mean(),
            'context_recall': metrics_df['context_recall'].mean()
        }
        
        # Save detailed metrics
        metrics_df.to_csv(self.metrics_dir / f"metrics_{timestamp}.csv")
        
        # Save configuration
        with open(self.configs_dir / f"config_{timestamp}.json", 'w') as f:
            json.dump(config, f, indent=4)
        
        # Add to history
        new_row = {
            'timestamp': timestamp,
            'experiment_name': self.experiment_name,
            **mean_metrics,
            'retriever_k': config.get('retriever_k', None),
            'model_name': config.get('model_name', None),
            'changes_made': config.get('changes_made', ''),
            'notes': notes
        }
        
        self.history = pd.concat([self.history, pd.DataFrame([new_row])], ignore_index=True)
        self.history.to_csv(self.history_file, index=False)
        
        return timestamp

    def get_improvement_summary(self):
        """Generate a summary of improvements over time"""
        if len(self.history) < 2:
            return "Not enough data for improvement analysis"
        
        first_run = self.history.iloc[0]
        last_run = self.history.iloc[-1]
        
        improvements = {
            'faithfulness': last_run['faithfulness'] - first_run['faithfulness'],
            'answer_relevancy': last_run['answer_relevancy'] - first_run['answer_relevancy'],
            'context_precision': last_run['context_precision'] - first_run['context_precision'],
            'context_recall': last_run['context_recall'] - first_run['context_recall']
        }
        
        return improvements

    def plot_metrics_over_time(self):
        """Plot metrics trends over time"""
        try:
            import matplotlib.pyplot as plt
            
            metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
            plt.figure(figsize=(12, 6))
            
            for metric in metrics:
                plt.plot(self.history['timestamp'], self.history[metric], label=metric, marker='o')
            
            plt.title('RAG Metrics Over Time')
            plt.xlabel('Experiment Timestamp')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(self.tracking_dir / 'metrics_trend.png')
            plt.close()
            
        except ImportError:
            print("matplotlib is required for plotting. Install it with: pip install matplotlib") 