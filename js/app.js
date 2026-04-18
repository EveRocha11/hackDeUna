/* ============================================================
   app.js
   Lógica centralizada para toda la aplicación:
   - Funcionalidad de index.html (Cobrar, Gestionar)
   - Flujo splash → onboarding → index
   - Modal de selección de rol
   ============================================================ */

/* ────────────────────────────────────────────────────────── */
/* ─ FUNCIONES DE INDEX.HTML (Cobrar, Gestionar) ─── */
/* ────────────────────────────────────────────────────────── */

let montoRaw = '12';

function switchTab(tabId) {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.classList.remove('active');
  });
  document.getElementById('tab-' + tabId).classList.add('active');

  document.querySelectorAll('.view').forEach(view => {
    view.classList.remove('active');
  });
  document.getElementById('view-' + tabId).classList.add('active');
}

function selectTipo(clickedBtn) {
  document.querySelectorAll('.tipo-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  clickedBtn.classList.add('active');
}

function pressKey(key) {
  const display = document.getElementById('monto-display');

  if (key === 'del') {
    montoRaw = montoRaw.slice(0, -1);
    if (montoRaw === '') montoRaw = '0';

  } else if (key === ',') {
    if (!montoRaw.includes(',')) {
      montoRaw += ',';
    }

  } else {
    if (montoRaw === '0') {
      montoRaw = key;
    } else {
      montoRaw += key;
    }
  }

  display.textContent = '$ ' + montoRaw;
}

/* ────────────────────────────────────────────────────────── */
/* ─ FUNCIONES DE SPLASH.HTML ──────────────── */
/* ────────────────────────────────────────────────────────── */

function goToOnboarding() {
  window.location.href = 'onboarding.html';
}

// Auto-redirect desde splash después de 3 segundos
(function initSplash() {
  const isSplashPage =
    document.body?.classList.contains('splash-page') ||
    document.documentElement.classList.contains('splash-page');

  if (isSplashPage) {
    globalThis.setTimeout(() => {
      globalThis.location.href = 'onboarding.html';
    }, 3000);
  }
})();

/* ────────────────────────────────────────────────────────── */
/* ─ FUNCIONES DE ONBOARDING.HTML ──────────── */
/* ────────────────────────────────────────────────────────── */

function goToAdmin() {
  window.location.href = 'index.html';
}

(function initOnboarding() {
  const openRoleModalBtn = document.getElementById('openRoleModal');
  const roleModal = document.getElementById('roleModal');
  const closeBtn = document.querySelector('[data-close="role-modal"]');

  if (!openRoleModalBtn || !roleModal) return;

  function openModal() {
    document.documentElement.classList.add('show-modal');
    roleModal.setAttribute('open', '');

    if (closeBtn instanceof HTMLElement) closeBtn.focus();
  }

  function closeModal() {
    document.documentElement.classList.remove('show-modal');
    roleModal.removeAttribute('open');
  }

  openRoleModalBtn.addEventListener('click', openModal);

  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (target.matches('[data-close="role-modal"]')) closeModal();
  });

  roleModal.addEventListener('cancel', (event) => {
    event.preventDefault();
    closeModal();
  });
})();