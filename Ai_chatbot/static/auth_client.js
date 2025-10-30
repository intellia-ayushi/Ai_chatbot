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

    // 🟢 Fetch and render previous user messages
    try {
      console.log("[Chat Debug] Fetching user messages...");
      const ur = await originalFetch("/user/messages", {
        headers: { Authorization: "Bearer " + accessToken },
      });

      console.log("[Chat Debug] /user/messages fetch completed:", ur.status);
      if (!ur.ok) {
        console.warn("[Chat Debug] Non-OK response from /user/messages:", ur.status);
      }

      const uj = await ur.json();
      console.log("[Chat Debug] /user/messages JSON parsed:", uj);

      const userMsgs = (uj && uj.messages) || [];
      console.log("[Chat Debug] userMsgs loaded:", userMsgs);

      // 🟢 Fetch user documents (optional)
      let hasDocs = false;
      try {
        const dr = await originalFetch("/documents", {
          headers: { Authorization: "Bearer " + accessToken },
        });
        const dj = await dr.json();
        hasDocs = Array.isArray(dj?.documents) && dj.documents.length > 0;
      } catch (e) {
        console.warn("[Chat Debug] Document check failed:", e);
      }

      // 🟢 Render to DOM
      const container = document.getElementById("chatContainer");
      if (!container) {
        console.warn("[Chat Debug] No chatContainer found in DOM.");
        return;
      }

      container.innerHTML = "";

      if (Array.isArray(userMsgs) && userMsgs.length > 0) {
        console.log("[Chat Debug] Rendering", userMsgs.length, "messages...");
        userMsgs.forEach((m) => {
          const role = (m.role || "assistant").toLowerCase();
          const text = m.content || "";
          const div = document.createElement("div");
          div.className = `message ${role === "user" ? "user-message" : "bot-message"}`;
          div.textContent = text;
          container.appendChild(div);
        });
        container.scrollTop = container.scrollHeight;
      } else if (!hasDocs) {
        const div = document.createElement("div");
        div.className = "message bot-message";
        div.textContent = "Please upload a document first.";
        container.appendChild(div);
      } else {
        const div = document.createElement("div");
        div.className = "bot-message message";
        div.textContent = "No previous chat found for this user.";
        container.appendChild(div);
      }

      console.log("[Chat Debug] Render complete.");
    } catch (e) {
      console.error("[Chat Debug] Error while fetching messages:", e);
    }
  })();
})();
