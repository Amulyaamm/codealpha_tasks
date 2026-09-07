// =====================================
// CodeAlpha AI Assistant
// =====================================

const chatBox = document.getElementById("chat-box");
const input = document.getElementById("message");

// ---------------------------
// Send Message
// ---------------------------

async function sendMessage() {

    const message = input.value.trim();

    if (message === "") return;

    appendUserMessage(message);

    input.value = "";

    showTyping();

    try {

        const response = await fetch("/chat_api", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: message
            })

        });

        const data = await response.json();

        removeTyping();

        appendBotMessage(data.answer);

    }

    catch (error) {

        removeTyping();

        appendBotMessage("⚠ Unable to connect to the server.");

        console.log(error);

    }

}

// ---------------------------
// User Bubble
// ---------------------------

function appendUserMessage(message){

    const div=document.createElement("div");

    div.className="user-message";

    div.innerHTML=message;

    chatBox.appendChild(div);

    scrollBottom();

}

// ---------------------------
// Bot Bubble
// ---------------------------

function appendBotMessage(message){

    const div=document.createElement("div");

    div.className="bot-message";

    div.innerHTML=message;

    chatBox.appendChild(div);

    scrollBottom();

}

// ---------------------------
// Typing
// ---------------------------

function showTyping(){

    const typing=document.createElement("div");

    typing.className="typing";

    typing.id="typing";

    typing.innerHTML=`

        <span></span>

        <span></span>

        <span></span>

    `;

    chatBox.appendChild(typing);

    scrollBottom();

}

function removeTyping(){

    const typing=document.getElementById("typing");

    if(typing){

        typing.remove();

    }

}

// ---------------------------
// Scroll
// ---------------------------

function scrollBottom(){

    chatBox.scrollTop=chatBox.scrollHeight;

}

// ---------------------------
// Quick Buttons
// ---------------------------

function quickQuestion(question){

    input.value=question;

    sendMessage();

}

// ---------------------------
// Clear Chat
// ---------------------------

function clearChat(){

    chatBox.innerHTML=`

    <div class="bot-message">

    👋 Hello!

    <br><br>

    I'm your CodeAlpha AI Internship Assistant.

    <br><br>

    Ask me anything about Internship, Tasks, GitHub, Certificates or Submission.

    </div>

    `;

}

// ---------------------------
// Enter Key
// ---------------------------

input.addEventListener("keypress",function(e){

    if(e.key==="Enter"){

        sendMessage();

    }

});
function speakText(text) {

    speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(text);

    speech.lang = "en-US";

    speech.rate = 1;

    speech.pitch = 1;

    speech.volume = 1;

    speechSynthesis.speak(speech);

}
chatBox.innerHTML += `
<div class="bot-message">

    <div class="message">

        ${response.answer}

    </div>

    <button class="speak-btn"
        onclick="speakText(\`${response.answer}\`)">

        🔊 Listen

    </button>

</div>
`;
chatMessages.innerHTML += `
<div class="bot-container">

    <img src="/static/images/ai-avatar.png" class="bot-avatar">

    <div class="bot-message">

        ${formatMessage(response.answer)}

    </div>

</div>
`;