import requests


filepath = "C:/Users/Lobby/Look At What Happens When I Heat Treat a Metal Lattice! [W2xxT3b-4H0].webm"

def whisperx_response(video_path):
    url = "http://127.0.0.1:8127/api/audio"
    params = {"file": video_path}
    try:
        response = requests.post(url, params=params)
        if response.status_code != 200:
            print(response.content)  # Print the response content if the status code is not 200
        data = response.json()
        segments = data['segments']
        language = data['language']
        return segments, language
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    segments, language = whisperx_response(filepath)
    print("Segments:", segments)
    print("Language:", language)