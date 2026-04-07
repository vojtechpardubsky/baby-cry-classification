const uploadForm = document.getElementById("upload-form");
const resultBox = document.getElementById("result");
const classLabels = {
    hungry: "hlad",
    belly_pain: "bolest břicha",
    discomfort: "diskomfort",
    tired: "únava",
    burping: "potřeba odříhnutí"
};

const recordBtn = document.getElementById("record-btn");
const stopBtn = document.getElementById("stop-btn");
const recordingStatus = document.getElementById("recording-status");
const audioPreview = document.getElementById("audio-preview");
const sendRecordingBtn = document.getElementById("send-recording-btn");

let mediaRecorder = null;
let audioChunks = [];
let recordedWavBlob = null;
let autoStopTimeout = null;
let previewUrl = null;

async function classifyFile(file) {
    resultBox.innerText = "Probíhá klasifikace...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            resultBox.innerText = "Chyba při klasifikaci: " + errorText;
            return;
        }

        const data = await response.json();

        const label = classLabels[data.predicted_class] || data.predicted_class;

        resultBox.innerText =
            "Predikovaná třída: " + label +
            "\nConfidence: " + data.confidence.toFixed(2);
    } catch (error) {
        resultBox.innerText = "Chyba komunikace se serverem.";
    }
}

uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("file");
    const file = fileInput.files[0];

    if (!file) {
        resultBox.innerText = "Nejprve vyberte audio soubor.";
        return;
    }

    await classifyFile(file);
});

recordBtn.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        audioChunks = [];
        recordedWavBlob = null;

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            try {
                const webmBlob = new Blob(audioChunks, { type: "audio/webm" });
                recordedWavBlob = await convertBlobToWav(webmBlob);

                if (previewUrl) {
                    URL.revokeObjectURL(previewUrl);
                }

                previewUrl = URL.createObjectURL(recordedWavBlob);
                audioPreview.src = previewUrl;
                audioPreview.classList.remove("hidden");
                sendRecordingBtn.classList.remove("hidden");
                recordingStatus.innerText = "Nahrávání dokončeno.";
            } catch (error) {
                recordingStatus.innerText = "Chyba při převodu nahrávky do WAV.";
            }

            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();

        recordingStatus.innerText = "Nahrávání probíhá... Maximálně 7 sekund.";
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        sendRecordingBtn.classList.add("hidden");
        audioPreview.classList.add("hidden");

        autoStopTimeout = setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === "recording") {
                mediaRecorder.stop();
                stopBtn.disabled = true;
                recordBtn.disabled = false;
            }
        }, 7000);

    } catch (error) {
        recordingStatus.innerText = "Nepodařilo se získat přístup k mikrofonu.";
    }
});

stopBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        clearTimeout(autoStopTimeout);
        stopBtn.disabled = true;
        recordBtn.disabled = false;
    }
});

sendRecordingBtn.addEventListener("click", async () => {
    if (!recordedWavBlob) {
        resultBox.innerText = "Nejdříve pořiďte nahrávku.";
        return;
    }

    const recordedFile = new File([recordedWavBlob], "recording.wav", {
        type: "audio/wav"
    });

    await classifyFile(recordedFile);
});

async function convertBlobToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const wavBuffer = audioBufferToWav(audioBuffer);
    return new Blob([wavBuffer], { type: "audio/wav" });
}

function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1;
    const bitDepth = 16;

    const channelData = [];
    for (let i = 0; i < numChannels; i++) {
        channelData.push(buffer.getChannelData(i));
    }

    const interleaved = interleave(channelData);
    const dataLength = interleaved.length * 2;
    const bufferLength = 44 + dataLength;
    const arrayBuffer = new ArrayBuffer(bufferLength);
    const view = new DataView(arrayBuffer);

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataLength, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bitDepth / 8, true);
    view.setUint16(32, numChannels * bitDepth / 8, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, "data");
    view.setUint32(40, dataLength, true);

    floatTo16BitPCM(view, 44, interleaved);

    return arrayBuffer;
}

function interleave(channelData) {
    if (channelData.length === 1) {
        return channelData[0];
    }

    const length = channelData[0].length;
    const result = new Float32Array(length * channelData.length);
    let index = 0;

    for (let i = 0; i < length; i++) {
        for (let channel = 0; channel < channelData.length; channel++) {
            result[index++] = channelData[channel][i];
        }
    }

    return result;
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        s = s < 0 ? s * 0x8000 : s * 0x7FFF;
        output.setInt16(offset, s, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}