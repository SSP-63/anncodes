# pip install SpeechRecognition
# pip install pyaudio


import speech_recognition as sr

def recognize_speech():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the microphone for input
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)  # Adjusts for ambient noise
        print("Listening for your speech...")
        audio = recognizer.listen(source)  # Listen to the speech

    try:
        # Use Google Web Speech API to recognize speech
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)  # Recognize the speech
        print(f"You said: {text}")  # Output the recognized text

    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")  # If speech is not clear
    except sr.RequestError:
        print("Sorry, the service is down or there is a problem with the API.")  # API issues

if __name__ == "__main__":
    recognize_speech()
