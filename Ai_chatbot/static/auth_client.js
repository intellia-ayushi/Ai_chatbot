(function () {
  const supabaseUrl = window.__SUPABASE_URL__ || "";
  const supabaseAnon = window.__SUPABASE_ANON__ || "";

  if (!supabaseUrl || !supabaseAnon || !window.supabase) {
    console.warn("[auth] Supabase not configured");
    return;
  }

  const supa = window.supabase.createClient(supabaseUrl, supabaseAnon);

  async function ensureSession() {
    const { data: { session } } = await supa.auth.getSession();
    if (!session) {
      console.warn("[auth] No session found, redirecting to login...");
      if (location.pathname !== "/") location.href = "/";
      return null;
    }
    return session;
  }

  (async function init() {
    console.log("[Chat Debug] Initializing auth_client.js...");
    const session = await ensureSession();
    if (!session) return;

    let accessToken = session.access_token;
    let currentChatId = null;

    supa.auth.onAuthStateChange((_event, s) => {
      if (s && s.access_token) accessToken = s.access_token;
    });

    // 🔧 Patch fetch to auto-inject auth token for local API calls
    const originalFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      init = init || {};
      const url = typeof input === "string" ? input : input?.url || "";
      const sameOrigin = url && (url.startsWith("/") || url.startsWith(location.origin));

      if (sameOrigin && accessToken) {
        const headers = new Headers(init.headers || {});
        if (!headers.has("Authorization")) {
          headers.set("Authorization", "Bearer " + accessToken);
        }
        init.headers = headers;
      }

      // auto attach chat_id
      if (
        sameOrigin &&
        url.includes("/ask") &&
        (init.method || "POST").toUpperCase() === "POST" &&
        init.body
      ) {
        try {
          const data = JSON.parse(init.body);
          if (currentChatId && !data.chat_id) {
            data.chat_id = currentChatId;
            init.body = JSON.stringify(data);
          }
        } catch (e) {}
      }
      return originalFetch(input, init);
    };

    // 🟢 Sync user profile on backend
    try {
      console.log("[Chat Debug] Syncing user profile...");
      await originalFetch("/auth/sync", {
        method: "POST",
        headers: { Authorization: "Bearer " + accessToken },
      });
    } catch (e) {
      console.warn("[auth] Profile sync failed:", e);
    }

    // Load per-user messages (aggregated) and render if present
    try {
      const ur = await originalFetch('/user/messages', { headers: { Authorization: 'Bearer ' + accessToken } });
      const uj = await ur.json();
      const userMsgs = (uj && uj.messages) || [];
      console.log('[Chat Debug] userMsgs loaded from DB:', userMsgs);

      // Check if user has any documents
      let hasDocs = false;
      let docs = [];
      try {
        const dr = await originalFetch('/documents', { headers: { Authorization: 'Bearer ' + accessToken } });
        const dj = await dr.json();
        docs = (dj && dj.documents) || [];
        hasDocs = docs.length > 0;
      } catch (e) {}

      // Now, after checking for docs, set chat input enabled/disabled globally
      const userInput = document.getElementById('userInput');
      if (userInput) {
        if (hasDocs) {
          userInput.disabled = false;
          userInput.placeholder = 'Ask your question...';
        } else {
          userInput.disabled = true;
          userInput.placeholder = 'Please upload a file first...';
        }
      }

      // Load and render messages to chat UI
      const container = document.getElementById('chatContainer');
      if (container) {
        if (Array.isArray(userMsgs) && userMsgs.length > 0) {
          container.innerHTML = '';
          userMsgs.forEach(m => {
            const role = (m.role || 'assistant').toLowerCase();
            const text = m.content || '';
            const div = document.createElement('div');
            div.className = `message ${role === 'user' ? 'user-message' : 'bot-message'}`;
            div.textContent = text;
            container.appendChild(div);
          });
          container.scrollTop = container.scrollHeight;
          console.log('Messages appended to DOM:', container.children.length);
        } else if (!hasDocs) {
          container.innerHTML = '';
          const div = document.createElement('div');
          div.className = 'message bot-message';
          div.textContent = 'Please upload document first.';
          container.appendChild(div);
        } else {
          container.innerHTML = '';
          const div = document.createElement('div');
          div.className = 'bot-message message';
          div.textContent = 'No previous chat found for this user.';
          container.appendChild(div);
        }
      }
    } catch (e) {
      console.warn('[auth] chat bootstrap failed:', e);
    }
  })();
})();
