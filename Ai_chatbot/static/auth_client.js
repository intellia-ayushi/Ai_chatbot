(function () {
  const supabaseUrl = window.__SUPABASE_URL__ || "";
  const supabaseAnon = window.__SUPABASE_ANON__ || "";

  if (!supabaseUrl || !supabaseAnon || !window.supabase) {
    console.warn("[auth] Supabase not configured");
    return;
  }

  const supa = window.supabase.createClient(supabaseUrl, supabaseAnon);

  function setLogoutEmailTooltip(email){
    try{
      const btnSide = document.getElementById('logoutBtnSide');
      if (btnSide){
        if (email){ btnSide.title = email; }
        else { btnSide.removeAttribute('title'); }
      }
    }catch(e){}
    try{
      const btnTop = document.getElementById('logout'); // chat_supabase.html
      if (btnTop){
        if (email){ btnTop.title = email; }
        else { btnTop.removeAttribute('title'); }
      }
    }catch(e){}
  }

  async function ensureSession() {
    const { data: { session } } = await supa.auth.getSession();
    if (!session) {
      console.warn("[auth] No session found, redirecting to login...");
      if (location.pathname !== "/") location.href = "/";
      return null;
    }
    try{ setLogoutEmailTooltip(session.user && session.user.email); }catch(e){}
    return session;
  }

  (async function init() {
    console.log("[Chat Debug] Initializing auth_client.js...");
    const session = await ensureSession();
    if (!session) return;

    let accessToken = session.access_token;
    let currentChatId = null;

    supa.auth.onAuthStateChange((_event, s) => {
      if (s && s.access_token) {
        accessToken = s.access_token;
        try{ setLogoutEmailTooltip(s.user && s.user.email); }catch(e){}
      } else {
        try{ setLogoutEmailTooltip(null); }catch(e){}
      }
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

    // helper: render docs list
    async function refreshDocsUI(){
      try{
        const dr = await originalFetch('/documents', { headers: { Authorization: 'Bearer ' + accessToken } });
        const dj = await dr.json();
        const docs = (dj && dj.documents) || [];
        const listEl = document.getElementById('docList');
        if (listEl){
          listEl.innerHTML = '';
          docs.forEach(d => {
            const name = (d && (d.display_name || d.filename || (d.path ? d.path.split('/').pop() : ''))) || '';
            const li = document.createElement('li');
            li.className = 'truncate font-semibold text-white';
            li.title = name;
            li.textContent = '• ' + name;
            listEl.appendChild(li);
          });
        }
        const userInput = document.getElementById('userInput');
        if (userInput){
          if (docs.length > 0){ userInput.disabled = false; userInput.placeholder = 'Ask your question...'; }
          else { userInput.disabled = true; userInput.placeholder = 'Please upload a file first...'; }
        }
      }catch(e){ console.warn('[auth] refreshDocsUI failed:', e); }
    }
    window.refreshDocsUI = refreshDocsUI;

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

      // Render docs list and enable input
      await refreshDocsUI();

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
        }
      }
    } catch (e) {
      console.warn('[auth] chat bootstrap failed:', e);
    }
  })();
})();
