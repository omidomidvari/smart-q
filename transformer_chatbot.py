#!/usr/bin/env python3
"""
Transformer-based AI Chatbot
Uses a pre-trained transformer model from Hugging Face
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class TransformerChatbot:
    """A chatbot powered by a transformer-based language model."""
    
    def __init__(self, model_name: str = "distilgpt2", device: Optional[str] = None):
        """
        Initialize the transformer-based chatbot.
        
        Args:
            model_name: Name of the model from Hugging Face (default: distilgpt2)
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conversation_history: List[Tuple[str, str]] = []
        self.system_prompt = "You are a helpful and friendly AI assistant."
        
        print(f"🤖 Loading transformer model: {model_name}")
        print(f"📍 Using device: {self.device}")
        
        self.load_model()
    
    def load_model(self) -> None:
        """Load the transformer model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print(f"✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def generate_response(
        self,
        user_input: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate a response using the transformer model.
        
        Args:
            user_input: The user's input message
            max_length: Maximum length of generated response
            temperature: Controls randomness (lower = more deterministic)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            Generated response text
        """
        # Format the input with conversation context
        prompt = self._format_prompt(user_input)
        
        try:
            # Generate response
            outputs = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            # Extract and clean response
            response = outputs[0]["generated_text"]
            response = self._clean_response(response, prompt)
            
            return response.strip()
        
        except Exception as e:
            print(f"✗ Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your message."
    
    def _format_prompt(self, user_input: str) -> str:
        """Format the prompt with conversation context."""
        prompt = f"{self.system_prompt}\n\n"
        
        # Add recent conversation history (last 3 exchanges)
        recent_history = self.conversation_history[-3:] if len(self.conversation_history) > 3 else self.conversation_history
        for user_msg, bot_msg in recent_history:
            prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        
        prompt += f"User: {user_input}\nAssistant:"
        return prompt
    
    def _clean_response(self, full_text: str, prompt: str) -> str:
        """Extract just the generated response from the full output."""
        # Remove the prompt part
        response = full_text[len(prompt):].strip()
        
        # Remove unwanted patterns
        response = response.split("\nUser:")[0].strip()
        response = response.split("\n\n")[0].strip()
        
        return response
    
    def chat(self, user_input: str, **kwargs) -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: The user's message
            **kwargs: Additional parameters for generate_response
            
        Returns:
            The chatbot's response
        """
        if not user_input.strip():
            return "Please say something! 😊"
        
        response = self.generate_response(user_input, **kwargs)
        self.conversation_history.append((user_input, response))
        
        return response
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt to guide the model's behavior.
        
        Args:
            prompt: The system prompt
        """
        self.system_prompt = prompt
        print(f"✓ System prompt updated")
    
    def get_history(self) -> List[Tuple[str, str]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def save_conversation(self, filepath: str) -> None:
        """
        Save the conversation to a JSON file.
        
        Args:
            filepath: Path to save the conversation
        """
        try:
            data = {
                "model": self.model_name,
                "system_prompt": self.system_prompt,
                "conversation": [
                    {"user": user, "assistant": assistant}
                    for user, assistant in self.conversation_history
                ]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Conversation saved to {filepath}")
        except Exception as e:
            print(f"✗ Error saving conversation: {e}")
    
    def load_conversation(self, filepath: str) -> None:
        """
        Load a conversation from a JSON file.
        
        Args:
            filepath: Path to the conversation file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.conversation_history = [
                    (turn["user"], turn["assistant"])
                    for turn in data.get("conversation", [])
                ]
            print(f"✓ Conversation loaded from {filepath}")
        except Exception as e:
            print(f"✗ Error loading conversation: {e}")
    
    def run_interactive(self) -> None:
        """Run the chatbot in interactive mode."""
        print("=" * 60)
        print("🤖 Transformer-based AI Chatbot")
        print("Commands:")
        print("  'quit' - Exit the chatbot")
        print("  'history' - View conversation history")
        print("  'clear' - Clear conversation history")
        print("  'save <file>' - Save conversation to file")
        print("  'prompt <text>' - Set system prompt")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    print("🤖 Chatbot: Goodbye! 👋")
                    break
                elif user_input.lower() == "history":
                    self._print_history()
                elif user_input.lower() == "clear":
                    self.clear_history()
                elif user_input.lower().startswith("save "):
                    filepath = user_input[5:].strip()
                    self.save_conversation(filepath)
                elif user_input.lower().startswith("prompt "):
                    new_prompt = user_input[7:].strip()
                    self.set_system_prompt(new_prompt)
                else:
                    response = self.chat(user_input)
                    print(f"🤖 Assistant: {response}")
            
            except KeyboardInterrupt:
                print("\n🤖 Chatbot: Goodbye! 👋")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
    
    def _print_history(self) -> None:
        """Print the conversation history."""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        print("\n" + "=" * 60)
        print("--- Conversation History ---")
        for i, (user, bot) in enumerate(self.conversation_history, 1):
            print(f"\n{i}. 👤 You: {user}")
            print(f"   🤖 Assistant: {bot}")
        print("\n" + "=" * 60)


def main():
    """Main entry point for the chatbot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transformer-based AI Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="Model name from Hugging Face (default: distilgpt2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    try:
        chatbot = TransformerChatbot(model_name=args.model, device=args.device)
        chatbot.run_interactive()
    except KeyboardInterrupt:
        print("\n\nShutting down...")


if __name__ == "__main__":
    main()
