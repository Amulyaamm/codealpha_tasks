/* =========================================================
   LINGUATRANSLATE
   Language Translation Tool
   ========================================================= */


/* ================= GET HTML ELEMENTS ================= */

const sourceText =
    document.getElementById("sourceText");

const translatedText =
    document.getElementById("translatedText");

const sourceLanguage =
    document.getElementById("sourceLanguage");

const targetLanguage =
    document.getElementById("targetLanguage");

const translateButton =
    document.getElementById("translateButton");

const clearButton =
    document.getElementById("clearButton");

const copyButton =
    document.getElementById("copyButton");

const speakButton =
    document.getElementById("speakButton");

const swapButton =
    document.getElementById("swapButton");

const characterCount =
    document.getElementById("characterCount");

const statusMessage =
    document.getElementById("statusMessage");



/* ================= CHECK ELEMENTS ================= */

console.log("LinguaTranslate JavaScript loaded.");

console.log("Source text:", sourceText);
console.log("Translated text:", translatedText);
console.log("Source language:", sourceLanguage);
console.log("Target language:", targetLanguage);
console.log("Translate button:", translateButton);



/* ================= CHARACTER COUNT ================= */

sourceText.addEventListener("input", function () {

    characterCount.textContent =
        `${sourceText.value.length} / 5000`;

});



/* =========================================================
   TRANSLATE
   ========================================================= */

translateButton.addEventListener("click", async function () {

    const text =
        sourceText.value.trim();


    /* ---------- EMPTY TEXT ---------- */

    if (!text) {

        statusMessage.textContent =
            "Please enter text to translate.";

        translatedText.value = "";

        return;
    }


    /* ---------- BUTTON ---------- */

    translateButton.disabled = true;

    translateButton.textContent =
        "Translating...";


    statusMessage.textContent =
        "Processing your translation...";


    try {

        console.log("Sending translation request...");

        console.log({
            text: text,
            source: sourceLanguage.value,
            target: targetLanguage.value
        });


        /* ---------- SEND REQUEST ---------- */

        const response =
            await fetch("/translate", {

                method: "POST",

                headers: {

                    "Content-Type":
                        "application/json"

                },

                body: JSON.stringify({

                    text: text,

                    source:
                        sourceLanguage.value,

                    target:
                        targetLanguage.value

                })

            });


        console.log(
            "Server response status:",
            response.status
        );


        /* ---------- READ RESPONSE ---------- */

        const data =
            await response.json();


        console.log(
            "Server response:",
            data
        );


        /* ---------- SERVER ERROR ---------- */

        if (!response.ok) {

            throw new Error(
                data.message ||
                "Translation request failed."
            );

        }


        /* ---------- SUCCESS ---------- */

        if (data.success === true) {

            translatedText.value =
                data.translation || "";


            statusMessage.textContent =
                "Translation completed.";


            console.log(
                "Translation:",
                data.translation
            );

        }


        /* ---------- BACKEND ERROR ---------- */

        else {

            translatedText.value = "";

            statusMessage.textContent =
                data.message ||
                "Translation failed.";

        }


    }


    /* ---------- CONNECTION ERROR ---------- */

    catch (error) {

        console.error(
            "Translation error:",
            error
        );


        translatedText.value = "";


        statusMessage.textContent =
            "Unable to connect to the server.";

    }


    /* ---------- RESET BUTTON ---------- */

    finally {

        translateButton.disabled = false;

        translateButton.textContent =
            "Translate →";

    }

});



/* =========================================================
   CLEAR
   ========================================================= */

clearButton.addEventListener("click", function () {

    sourceText.value = "";

    translatedText.value = "";

    characterCount.textContent =
        "0 / 5000";

    statusMessage.textContent = "";

});



/* =========================================================
   COPY
   ========================================================= */

copyButton.addEventListener("click", async function () {

    const text =
        translatedText.value.trim();


    if (!text) {

        statusMessage.textContent =
            "Nothing to copy.";

        return;

    }


    try {

        await navigator.clipboard.writeText(text);


        statusMessage.textContent =
            "Translation copied.";

    }


    catch (error) {

        console.error(
            "Copy error:",
            error
        );


        statusMessage.textContent =
            "Unable to copy translation.";

    }

});



/* =========================================================
   TEXT TO SPEECH
   ========================================================= */

speakButton.addEventListener("click", function () {

    const text =
        translatedText.value.trim();


    if (!text) {

        statusMessage.textContent =
            "Nothing to speak.";

        return;

    }


    /* Stop previous speech */

    window.speechSynthesis.cancel();


    const speech =
        new SpeechSynthesisUtterance(text);


    /* Set language */

    speech.lang =
        targetLanguage.value;


    speech.rate = 1;

    speech.pitch = 1;


    window.speechSynthesis.speak(
        speech
    );


    statusMessage.textContent =
        "Speaking translation...";

});



/* =========================================================
   SWAP LANGUAGES
   ========================================================= */

swapButton.addEventListener("click", function () {

    const source =
        sourceLanguage.value;

    const target =
        targetLanguage.value;


    /* ---------- SWAP LANGUAGE ---------- */

    sourceLanguage.value =
        target;

    targetLanguage.value =
        source;


    /* ---------- SWAP TEXT ---------- */

    const sourceValue =
        sourceText.value;

    sourceText.value =
        translatedText.value;

    translatedText.value =
        sourceValue;


    /* ---------- UPDATE CHARACTER COUNT ---------- */

    characterCount.textContent =
        `${sourceText.value.length} / 5000`;


    statusMessage.textContent =
        "Languages swapped.";

});



/* =========================================================
   ENTER KEY / CTRL + ENTER
   ========================================================= */

sourceText.addEventListener(
    "keydown",
    function (event) {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            translateButton.click();

        }

    }
);