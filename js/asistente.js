/* ============================================================
   asistente.js
   Lógica para cargar y gestionar las preguntas frecuentes
   ============================================================ */

const API_URL = 'http://localhost:8000/api/v1/agent/query';
const THREAD_ID = 'frontend-chat-1';

async function cargarPreguntas() {
  try {
    const response = await fetch('../data/Informacion.txt');
    const texto = await response.text();

    const preguntas = texto
      .split('\n')
      .map((línea, index) => ({
        id: index + 1,
        texto: línea.trim()
      }))
      .filter(pregunta => pregunta.texto.length > 0)
      .slice(0, 5);

    mostrarPreguntas(preguntas);
  } catch (error) {
    console.error('Error al cargar las preguntas:', error);
  }
}

function mostrarPreguntas(preguntas) {
  const container = document.getElementById('preguntas-container');
  if (!container) return;

  container.innerHTML = '';

  preguntas.forEach(pregunta => {
    const bubble = document.createElement('button');
    bubble.className = 'pregunta-bubble';
    bubble.textContent = pregunta.texto;
    bubble.addEventListener('click', () => enviarPregunta(pregunta.texto));
    container.appendChild(bubble);
  });
}

function crearMensajeUsuario(texto) {
  const userMessage = document.createElement('div');
  userMessage.className = 'chat-message user-message';
  userMessage.innerHTML = `<div class="chat-bubble user-bubble"><p>${texto}</p></div>`;
  return userMessage;
}

function crearMensajeBot(texto) {
  const botMessage = document.createElement('div');
  botMessage.className = 'chat-message bot-message';
  botMessage.innerHTML = `
    <img class="chat-avatar" src="../img/duni.jpeg" alt="Avatar del asistente" />
    <div class="chat-bubble">
      <p>${texto}</p>
    </div>
  `;
  return botMessage;
}

function crearMensajeCarga() {
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
  return loadingMessage;
}

function removerMensajeCarga() {
  const loading = document.getElementById('loading-message');
  if (loading) {
    loading.remove();
  }
}

function scrollChatAlFinal() {
  const chatContainer = document.querySelector('.asistente-chat');
  if (chatContainer) {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
}

function ocultarPreguntasFrecuentes() {
  const preguntasSection = document.querySelector('.preguntas-frecuentes-section');
  if (preguntasSection) {
    preguntasSection.style.display = 'none';
  }
}

async function askAgent(question) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      thread_id: THREAD_ID
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  return data;
}

async function enviarPregunta(texto) {
  const chatContainer = document.querySelector('.asistente-chat');
  if (!chatContainer || !texto || !texto.trim()) return;

  const pregunta = texto.trim();
  chatContainer.appendChild(crearMensajeUsuario(pregunta));
  const loadingMessage = crearMensajeCarga();
  chatContainer.appendChild(loadingMessage);
  scrollChatAlFinal();
  ocultarPreguntasFrecuentes();

  try {
    const data = await askAgent(pregunta);
    removerMensajeCarga();

    const answer = data?.answer || 'No obtuve respuesta del servidor.';
    chatContainer.appendChild(crearMensajeBot(answer));
  } catch (error) {
    console.error('Error al llamar la API:', error);
    removerMensajeCarga();
    chatContainer.appendChild(crearMensajeBot('Lo siento, ocurrió un error al conectar con el asistente. Intenta de nuevo.'));
  } finally {
    scrollChatAlFinal();
  }
}

function enviarTextoDeEntrada() {
  const input = document.querySelector('.asistente-text-input');
  if (!input) return;

  const texto = input.value.trim();
  if (!texto) return;

  enviarPregunta(texto);
  input.value = '';
}

function inicializarEventos() {
  const sendButton = document.querySelector('.btn-send');
  const textInput = document.querySelector('.asistente-text-input');

  if (sendButton) {
    sendButton.addEventListener('click', enviarTextoDeEntrada);
  }

  if (textInput) {
    textInput.addEventListener('keydown', event => {
      if (event.key === 'Enter') {
        event.preventDefault();
        enviarTextoDeEntrada();
      }
    });
  }
}

function inicializarAsistente() {
  cargarPreguntas();
  inicializarEventos();
}

document.addEventListener('DOMContentLoaded', inicializarAsistente);
