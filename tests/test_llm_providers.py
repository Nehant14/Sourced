import importlib
import sys
import types


def test_build_llm_provider_supports_gemini(monkeypatch):
    fake_module = types.ModuleType("google.generativeai")
    fake_module.configure = lambda api_key: None
    fake_module.GenerativeModel = lambda model_name, system_instruction=None: types.SimpleNamespace(
        generate_content=lambda prompt: types.SimpleNamespace(text='{"ok": true}')
    )

    google_module = types.ModuleType("google")
    google_module.generativeai = fake_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_module)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    from app.providers.llm import build_llm_provider

    provider = build_llm_provider()
    assert provider.__class__.__name__ == "GeminiProvider"


def test_dotenv_values_are_loaded_for_llm_provider(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=gemini\nGEMINI_API_KEY=fake-key-from-dotenv\n", encoding="utf-8")

    fake_module = types.ModuleType("google.generativeai")
    fake_module.configure = lambda api_key: None
    fake_module.GenerativeModel = lambda model_name, system_instruction=None: types.SimpleNamespace(
        generate_content=lambda prompt: types.SimpleNamespace(text='{"ok": true}')
    )

    google_module = types.ModuleType("google")
    google_module.generativeai = fake_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_module)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import app

    importlib.reload(app)
    from app.providers.llm import build_llm_provider

    provider = build_llm_provider()
    assert provider.__class__.__name__ == "GeminiProvider"
