package oai

import (
	"encoding/json"
	"testing"

	"github.com/sashabaranov/go-openai"
	"shelley.exe.dev/llm"
)

func TestOpenAIUserMessageWithImage(t *testing.T) {
	msg := llm.Message{
		Role: llm.MessageRoleUser,
		Content: []llm.Content{
			{Type: llm.ContentTypeText, Text: "What's in this image?"},
			{Type: llm.ContentTypeText, MediaType: "image/png", Data: "iVBORw0KGgo..."},
		},
	}

	msgs := fromLLMMessage(msg)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	m := msgs[0]
	if m.Role != "user" {
		t.Errorf("role = %q, want %q", m.Role, "user")
	}
	if m.Content != "" {
		t.Errorf("Content = %q, want empty (MultiContent should be used)", m.Content)
	}
	if len(m.MultiContent) != 2 {
		t.Fatalf("MultiContent length = %d, want 2", len(m.MultiContent))
	}
	if m.MultiContent[0].Type != openai.ChatMessagePartTypeText || m.MultiContent[0].Text != "What's in this image?" {
		t.Errorf("part[0] = %+v", m.MultiContent[0])
	}
	if m.MultiContent[1].Type != openai.ChatMessagePartTypeImageURL {
		t.Fatalf("part[1] type = %q, want image_url", m.MultiContent[1].Type)
	}
	want := "data:image/png;base64,iVBORw0KGgo..."
	if m.MultiContent[1].ImageURL == nil || m.MultiContent[1].ImageURL.URL != want {
		t.Errorf("part[1] URL = %+v, want %q", m.MultiContent[1].ImageURL, want)
	}

	// Marshaling should produce a single content array with the image_url entry.
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	arr, ok := decoded["content"].([]any)
	if !ok {
		t.Fatalf("content not an array: %v", decoded["content"])
	}
	if len(arr) != 2 {
		t.Fatalf("content array length = %d, want 2", len(arr))
	}
	img := arr[1].(map[string]any)
	if img["type"] != "image_url" {
		t.Errorf("content[1].type = %v, want image_url", img["type"])
	}
	imgURL := img["image_url"].(map[string]any)
	if imgURL["url"] != want {
		t.Errorf("content[1].image_url.url = %v, want %q", imgURL["url"], want)
	}
}

func TestOpenAIUserMessageTextOnlyStillUsesContentString(t *testing.T) {
	msg := llm.Message{
		Role:    llm.MessageRoleUser,
		Content: []llm.Content{{Type: llm.ContentTypeText, Text: "Hello"}},
	}
	msgs := fromLLMMessage(msg)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Content != "Hello" {
		t.Errorf("Content = %q, want %q", msgs[0].Content, "Hello")
	}
	if msgs[0].MultiContent != nil {
		t.Errorf("MultiContent = %+v, want nil for text-only", msgs[0].MultiContent)
	}
}

func TestOpenAIToolResultWithImageEmitsFollowupUserMessage(t *testing.T) {
	msg := llm.Message{
		Role: llm.MessageRoleUser,
		Content: []llm.Content{{
			Type:      llm.ContentTypeToolResult,
			ToolUseID: "call_abc",
			ToolResult: []llm.Content{
				{Type: llm.ContentTypeText, Text: "screenshot taken"},
				{Type: llm.ContentTypeText, MediaType: "image/png", Data: "AAAA"},
			},
		}},
	}

	msgs := fromLLMMessage(msg)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (tool + user image), got %d", len(msgs))
	}

	// First: the tool-role message with only text.
	if msgs[0].Role != "tool" {
		t.Errorf("msgs[0].Role = %q, want tool", msgs[0].Role)
	}
	if msgs[0].Content != "screenshot taken" {
		t.Errorf("msgs[0].Content = %q, want %q", msgs[0].Content, "screenshot taken")
	}
	if msgs[0].ToolCallID != "call_abc" {
		t.Errorf("msgs[0].ToolCallID = %q, want %q", msgs[0].ToolCallID, "call_abc")
	}
	if msgs[0].MultiContent != nil {
		t.Errorf("msgs[0].MultiContent should be nil (OpenAI tool role can't hold images)")
	}

	// Second: a user-role message carrying the image.
	if msgs[1].Role != "user" {
		t.Errorf("msgs[1].Role = %q, want user", msgs[1].Role)
	}
	if len(msgs[1].MultiContent) != 1 {
		t.Fatalf("msgs[1].MultiContent length = %d, want 1", len(msgs[1].MultiContent))
	}
	if msgs[1].MultiContent[0].Type != openai.ChatMessagePartTypeImageURL {
		t.Errorf("msgs[1].MultiContent[0].Type = %q, want image_url", msgs[1].MultiContent[0].Type)
	}
	want := "data:image/png;base64,AAAA"
	if msgs[1].MultiContent[0].ImageURL.URL != want {
		t.Errorf("image URL = %q, want %q", msgs[1].MultiContent[0].ImageURL.URL, want)
	}
}

func TestOpenAIToolResultImageOnlyFillsPlaceholderText(t *testing.T) {
	msg := llm.Message{
		Role: llm.MessageRoleUser,
		Content: []llm.Content{{
			Type:      llm.ContentTypeToolResult,
			ToolUseID: "call_xyz",
			ToolResult: []llm.Content{
				{Type: llm.ContentTypeText, MediaType: "image/jpeg", Data: "/9j/"},
			},
		}},
	}
	msgs := fromLLMMessage(msg)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Content == "" || msgs[0].Content == " " {
		t.Errorf("expected placeholder content in tool message, got %q", msgs[0].Content)
	}
}

func TestResponsesAPIUserMessageWithImage(t *testing.T) {
	msg := llm.Message{
		Role: llm.MessageRoleUser,
		Content: []llm.Content{
			{Type: llm.ContentTypeText, Text: "Look at this"},
			{Type: llm.ContentTypeText, MediaType: "image/jpeg", Data: "AAAA"},
		},
	}
	items := fromLLMMessageResponses(msg)
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	if items[0].Type != "message" || items[0].Role != "user" {
		t.Errorf("item = %+v, want message/user", items[0])
	}
	if len(items[0].Content) != 2 {
		t.Fatalf("content length = %d, want 2", len(items[0].Content))
	}
	if items[0].Content[0].Type != "input_text" || items[0].Content[0].Text != "Look at this" {
		t.Errorf("content[0] = %+v", items[0].Content[0])
	}
	if items[0].Content[1].Type != "input_image" {
		t.Errorf("content[1].Type = %q, want input_image", items[0].Content[1].Type)
	}
	want := "data:image/jpeg;base64,AAAA"
	if items[0].Content[1].ImageURL != want {
		t.Errorf("content[1].ImageURL = %q, want %q", items[0].Content[1].ImageURL, want)
	}

	// Image-type entries must not emit an empty "text" field.
	raw, err := json.Marshal(items[0].Content[1])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := decoded["text"]; ok {
		t.Errorf("image content should not include text field, got %v", decoded)
	}
}

func TestResponsesAPIToolResultWithImage(t *testing.T) {
	msg := llm.Message{
		Role: llm.MessageRoleUser,
		Content: []llm.Content{{
			Type:      llm.ContentTypeToolResult,
			ToolUseID: "call_1",
			ToolResult: []llm.Content{
				{Type: llm.ContentTypeText, Text: "ok"},
				{Type: llm.ContentTypeText, MediaType: "image/png", Data: "BBBB"},
			},
		}},
	}
	items := fromLLMMessageResponses(msg)
	if len(items) != 2 {
		t.Fatalf("expected 2 items (function_call_output + user message), got %d", len(items))
	}
	if items[0].Type != "function_call_output" || items[0].CallID != "call_1" || items[0].Output != "ok" {
		t.Errorf("items[0] = %+v", items[0])
	}
	if items[1].Type != "message" || items[1].Role != "user" {
		t.Errorf("items[1] = %+v, want message/user", items[1])
	}
	if len(items[1].Content) != 1 || items[1].Content[0].Type != "input_image" {
		t.Fatalf("items[1].Content = %+v", items[1].Content)
	}
	want := "data:image/png;base64,BBBB"
	if items[1].Content[0].ImageURL != want {
		t.Errorf("image URL = %q, want %q", items[1].Content[0].ImageURL, want)
	}
}
