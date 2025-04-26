import open_clip
import torch

class OpenClipModel:
    def __init__(self, device="cuda"):
        
        self.device = device
        self.feature_dim = 512
        
        # Load the OpenCLIP model
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
        self.model = model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        
        # Canonical phrases stay fixed
        self.canonical_phrases = ["object", "things", "stuff", "texture"]
        with torch.no_grad():
            canon_tokens = self.tokenizer(self.canonical_phrases)
            canon_embed = model.encode_text(canon_tokens.to(device))
            canon_embed /= canon_embed.norm(dim=-1, keepdim=True)
            self.canon_embed = canon_embed.cpu() # (4, 512)
            
    def encode_text_for_resMLP(self, prompt):
        with torch.no_grad():
            # prompt_tokens = self.tokenizer([prompt] + self.canonical_phrases)
            # prompt_embed = self.model.encode_text(prompt_tokens.to(self.device))  # (4, 512)
            # prompt_embed /= prompt_embed.norm(dim=-1, keepdim=True) # (4, 512)
            
            prompt_tokens = self.tokenizer([prompt])
            prompt_embed = self.model.encode_text(prompt_tokens.to(self.device))  # (1, 512)
            prompt_embed /= prompt_embed.norm(dim=-1, keepdim=True) # (1, 512)
            
        return prompt_embed
        
    def encode_texts(self, promts):
        
        self.prompts = promts
        
        with torch.no_grad():
            prompt_tokens = self.tokenizer(promts)
            prompt_embed = self.model.encode_text(prompt_tokens.to(self.device))  # (C, 512)
            prompt_embed /= prompt_embed.norm(dim=-1, keepdim=True)
            self.prompt_embed = prompt_embed.cpu() # (n_phrases, 512)
            
    def compute_relevancy_map_single_text(self, sem_map, prompt):
        
        prompt_embed = self.encode_text_for_resMLP(prompt)
        sem_map = sem_map.to(self.device, non_blocking=True)
        
        relevancy_map = torch.matmul(sem_map, prompt_embed.t())  # (N, C)
        
        assert torch.min(relevancy_map) >= -1 and torch.max(relevancy_map) <= 1, "Values outside expected range [-1, 1]"
        
        return relevancy_map
        
            
    def compute_relevancy_map(self, sem_map): # (h*w, 512)
        
        n_prompts = len(self.prompts)
        
        # Move to device
        sem_map = sem_map.to(self.device, non_blocking=True)
        prompt_embeds = self.prompt_embed.to(self.device, non_blocking=True)
        canon_embeds = self.canon_embed.to(self.device, non_blocking=True)
        
        _, c = sem_map.shape
        
        # Original ratio‑based relevancy score
        dot_lang_text = torch.matmul(sem_map, prompt_embeds.t())    # (h*w, C)
        dot_lang_canon = torch.matmul(sem_map, canon_embeds.t())  # (h*w, 4)
        
        exp_lang_text = dot_lang_text.exp()    # (h*w, C)
        exp_lang_canon = dot_lang_canon.exp()  # (h*w, 4)
        
        relevancy_scores = []
        for c_idx in range(n_prompts):
            text_c_exp = exp_lang_text[:, c_idx].unsqueeze(-1)  # (h*w,1)
            ratio_c = text_c_exp / (exp_lang_canon + text_c_exp)  # (h*w, K)
            score_c = torch.min(ratio_c, dim=1).values  # (h*w,)
            relevancy_scores.append(score_c)
        
        relevancy_matrix = torch.stack(relevancy_scores, dim=0).t() # (h*w, C)
        
        return relevancy_matrix
        
if __name__ == "__main__":
    clip_model = OpenClipModel()
    print("Clip model loaded successfully.")
    # You can now use clip_model to compute similarities, etc.