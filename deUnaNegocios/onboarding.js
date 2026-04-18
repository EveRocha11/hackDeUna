(function () {
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
