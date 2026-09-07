const chat=document.getElementById("chatMessages");

const typing=document.getElementById("typing");

async function sendMessage(){

const input=document.getElementById("message");

const message=input.value.trim();

if(message==="") return;

chat.innerHTML+=`<div class="user">${message}</div>`;

input.value="";

chat.scrollTop=chat.scrollHeight;

typing.style.display="block";

const response=await fetch("/chat",{

method:"POST",

headers:{

"Content-Type":"application/json"

},

body:JSON.stringify({

message:message

})

});

const data=await response.json();

typing.style.display="none";

chat.innerHTML+=`<div class="bot">${data.answer}</div>`;

chat.scrollTop=chat.scrollHeight;

}

function quickQuestion(text){

document.getElementById("message").value=text;

sendMessage();

}

document.getElementById("message")

.addEventListener("keypress",function(e){

if(e.key==="Enter"){

sendMessage();

}

});