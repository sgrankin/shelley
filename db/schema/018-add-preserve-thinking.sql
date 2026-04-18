-- Add preserve_thinking column to models.
-- When enabled the server retains historical thinking/reasoning traces in
-- subsequent requests. Works for openai chat-completions (Qwen3.6 style
-- chat_template_kwargs.preserve_thinking) and anthropic (skip the historical
-- thinking-block strip).

ALTER TABLE models ADD COLUMN preserve_thinking INTEGER NOT NULL DEFAULT 0;
