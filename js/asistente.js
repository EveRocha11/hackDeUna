/* ============================================================
   asistente.js
   Lógica para cargar y gestionar las preguntas frecuentes
   ============================================================ */

// Cargar las preguntas desde el archivo txt
async function cargarPreguntas() {
  try {
    const response = await fetch('../data/Informacion.txt');
    const texto = await response.text();
    
    // Parsear el archivo txt: una pregunta por línea
    const preguntas = texto
      .split('\n')
      .map((línea, index) => ({
        id: index + 1,
        texto: línea.trim()
      }))
      .filter(pregunta => pregunta.texto.length > 0) // Eliminar líneas vacías
      .slice(0, 5); // Limitar a máximo 5 preguntas
    
    mostrarPreguntas(preguntas);
  } catch (error) {
    console.error('Error al cargar las preguntas:', error);
  }
}

// Mostrar las preguntas como burbujas clicables
function mostrarPreguntas(preguntas) {
  const container = document.getElementById('preguntas-container');
  
  if (!container) return;

  container.innerHTML = '';

  preguntas.forEach(pregunta => {
    const bubble = document.createElement('button');
    bubble.className = 'pregunta-bubble';
    bubble.textContent = pregunta.texto;
    
    bubble.addEventListener('click', () => seleccionarPregunta(pregunta));
    
    container.appendChild(bubble);
  });
}

// Manejar la selección de una pregunta
function seleccionarPregunta(pregunta) {
  const chatContainer = document.querySelector('.asistente-chat');
  
  // Agregar la pregunta del usuario al chat
  const userMessage = document.createElement('div');
  userMessage.className = 'chat-message user-message';
  userMessage.innerHTML = `<div class="chat-bubble user-bubble"><p>${pregunta.texto}</p></div>`;
  chatContainer.appendChild(userMessage);

  // Agregar animación de carga mientras se procesa
  const loadingMessage = document.createElement('div');
  loadingMessage.className = 'chat-message bot-message';
  loadingMessage.id = 'loading-message';
  loadingMessage.innerHTML = `
    <img class="chat-avatar chat-avatar-animated" src="../img/duni.jpeg" alt="Avatar del asistente" />
    <div class="chat-bubble">
      <div class="loading-indicator">
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
      </div>
    </div>
  `;
  chatContainer.appendChild(loadingMessage);

  // Simular respuesta del asistente (puedes personalizarla)
  setTimeout(() => {
    // Remover el mensaje de carga
    const loading = document.getElementById('loading-message');
    if (loading) {
      loading.remove();
    }

    const botMessage = document.createElement('div');
    botMessage.className = 'chat-message bot-message';
    botMessage.innerHTML = `
      <img class="chat-avatar" src="../img/duni.jpeg" alt="Avatar del asistente" />
      <div class="chat-bubble">
        <p>Estoy procesando tu pregunta: "${pregunta.texto}"</p>
      </div>
    `;
    chatContainer.appendChild(botMessage);
    
    // Scroll al final del chat
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }, 2000);

  // Ocultar las preguntas frecuentes después de seleccionar una
  const preguntasSection = document.querySelector('.preguntas-frecuentes-section');
  if (preguntasSection) {
    preguntasSection.style.display = 'none';
  }

  // Scroll al final del chat
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', cargarPreguntas);
