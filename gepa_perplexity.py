import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from datetime import datetime
import json
import traceback

sequence_stats_history = []

class SequenceStatsTracker:
    """Track sequence statistics (perplexity, confidence) aligned with GEPA candidates."""
    
    def __init__(self):
        self.stats = []
        self.current_stage = "training"  # Default to training during GEPA optimization
        
    def set_stage(self, stage):
        """Set current stage: 'baseline', 'training', or 'test'"""
        self.current_stage = stage
        
    def compute_sequence_stats(self, prediction, prompt_text, system_prompt, score=None, candidate_id=None, predictor_id=None):
        """
        Extract and store perplexity and confidence metrics.
        system_prompt: full candidate prompt from predictor.signature.instructions
        """
        
        stat_entry = {
            'candidate_prompt': system_prompt,  # ✅ Store full candidate prompt
            'candidate_id': candidate_id,
            'user_prompt': prompt_text[:100],  # First 100 chars of user prompt
            'score': score,  # ✅ Track score from metric
            'stage': self.current_stage,
            'perplexity': None,
            'top_token_confidence': None,
            'token_count': 0,
            'timestamp': datetime.now().isoformat(),
        }
        
        # ✅ Extract logprobs from litellm format (if available)
        if hasattr(prediction, 'logprobs') and prediction.logprobs:
            logprobs = prediction.logprobs
            
            if hasattr(logprobs, 'content') and isinstance(logprobs.content, list):
                token_logprobs = []
                top_logprobs = []
                
                for token_obj in logprobs.content:
                    if hasattr(token_obj, 'logprob'):
                        token_logprobs.append(token_obj.logprob)
                    
                    if hasattr(token_obj, 'top_logprobs') and token_obj.top_logprobs:
                        top_lps = [lp.logprob for lp in token_obj.top_logprobs if hasattr(lp, 'logprob')]
                        if top_lps:
                            top_logprobs.append(top_lps[0])
                
                if token_logprobs:
                    mean_logprob = np.mean(token_logprobs)
                    stat_entry['perplexity'] = np.exp(-mean_logprob)
                    stat_entry['token_count'] = len(token_logprobs)
                
                if top_logprobs:
                    stat_entry['top_token_confidence'] = np.exp(np.mean(top_logprobs))
        
        self.stats.append(stat_entry)

    def map_predictor_ids_to_candidates(self, results):
        """
        After GEPA completes, map predictor object IDs to candidate indices.
        This is done by matching the predictor objects from results.candidates.
        """
        print("\n" + "="*80)
        print("🔗 MAPPING PREDICTOR IDs TO GEPA CANDIDATES")
        print("="*80)
        
        # Build mapping: predictor_id -> candidate_idx
        predictor_to_candidate = {}
        
        for i, candidate in enumerate(results.candidates):
            for name, predictor in candidate.named_predictors():
                predictor_id = id(predictor)
                predictor_to_candidate[predictor_id] = i
                print(f"   Mapped: Predictor ID {predictor_id} → Candidate #{i}")
                break
        
        # Update all stats with candidate indices
        mapped_count = 0
        unmapped_count = 0
        
        for stat in self.stats:
            pred_id = stat.get('predictor_id')
            if pred_id and pred_id in predictor_to_candidate:
                stat['candidate_id'] = predictor_to_candidate[pred_id]
                mapped_count += 1
            else:
                unmapped_count += 1
        
        print(f"\n📈 Mapping Results:")
        print(f"   ✅ Mapped: {mapped_count}/{len(self.stats)}")
        print(f"   ❌ Unmapped: {unmapped_count}/{len(self.stats)}")
        
        unique_candidates = len(set(s['candidate_id'] for s in self.stats if s['candidate_id'] is not None))
        print(f"   📊 Unique candidates with stats: {unique_candidates}/{len(results.candidates)}")
        
        return mapped_count, unmapped_count

    def get_candidate_summary(self):
        """Get summary statistics grouped by candidate."""
        candidate_stats = defaultdict(lambda: {
            'count': 0,
            'perplexities': [],
            'confidences': [],
            'scores': [],
        })
        
        for stat in self.stats:
            cand_id = stat.get('candidate_id')
            if cand_id is None:
                continue
            
            candidate_stats[cand_id]['count'] += 1
            
            if stat['perplexity'] is not None:
                candidate_stats[cand_id]['perplexities'].append(stat['perplexity'])
            
            if stat['top_token_confidence'] is not None:
                candidate_stats[cand_id]['confidences'].append(stat['top_token_confidence'])
            
            if stat['score'] is not None:
                candidate_stats[cand_id]['scores'].append(stat['score'])
        
        # Calculate aggregates
        result = {}
        for cand_id, stats in candidate_stats.items():
            result[cand_id] = {
                'count': stats['count'],
                'mean_perplexity': np.mean(stats['perplexities']) if stats['perplexities'] else None,
                'std_perplexity': np.std(stats['perplexities']) if stats['perplexities'] else None,
                'mean_confidence': np.mean(stats['confidences']) if stats['confidences'] else None,
                'std_confidence': np.std(stats['confidences']) if stats['confidences'] else None,
                'mean_score': np.mean(stats['scores']) if stats['scores'] else None,
            }
        
        return result

    def save_stats(self, filepath):
        """Save raw statistics to JSON."""
        export_data = {
            'total_samples': len(self.stats),
            'samples': [
                {
                    'predictor_id': s.get('predictor_id'),
                    'candidate_id': s.get('candidate_id'),
                    'score': s['score'],
                    'stage': s.get('stage', 'unknown'),
                    'perplexity': s['perplexity'],
                    'confidence': s['top_token_confidence'],
                    'tokens': s['token_count'],
                }
                for s in self.stats
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Saved statistics to {filepath}")

    def save_candidate_stats(self, filepath):
        """Save per-candidate statistics to JSON."""
        candidate_stats = self.get_candidate_summary()
        
        export_data = {}
        for cand_id, stats in candidate_stats.items():
            export_data[str(cand_id)] = {
                'count': stats['count'],
                'mean_perplexity': float(stats['mean_perplexity']) if stats['mean_perplexity'] else None,
                'mean_confidence': float(stats['mean_confidence']) if stats['mean_confidence'] else None,
                'mean_score': float(stats['mean_score']) if stats['mean_score'] else None,
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Saved per-candidate statistics to {filepath}")

    def plot_candidate_stats(self, filepath, top_n=10):
        """Create visualization of top N candidates by score."""
        import matplotlib.pyplot as plt
        
        candidate_stats = self.get_candidate_summary()
        
        sorted_candidates = sorted(
            candidate_stats.items(),
            key=lambda x: x[1]['mean_score'] if x[1]['mean_score'] else 0,
            reverse=True
        )[:top_n]
        
        cand_ids = [str(cid) for cid, _ in sorted_candidates]
        perplexities = [stats['mean_perplexity'] or 0 for _, stats in sorted_candidates]
        confidences = [stats['mean_confidence'] or 0 for _, stats in sorted_candidates]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.barh(cand_ids, perplexities, color='steelblue')
        ax1.set_xlabel('Mean Perplexity (Lower is Better)')
        ax1.set_title(f'Top {top_n} Candidates by Score\n(Perplexity)')
        ax1.invert_yaxis()
        
        ax2.barh(cand_ids, confidences, color='seagreen')
        ax2.set_xlabel('Mean Confidence (Higher is Better)')
        ax2.set_title(f'Top {top_n} Candidates by Score\n(Confidence)')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved candidate plot to {filepath}")
        plt.close()