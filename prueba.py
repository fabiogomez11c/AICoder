import requests


def test_stream():
    url = "http://localhost:8000/stream"
    data = {"message": "Write a Python function to generate Fibonacci numbers"}

    with requests.post(url, json=data, stream=True) as response:
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print("Streamed Response:")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    print(decoded_line[6:])  # Remove 'data: ' prefix
                else:
                    print(decoded_line)  # Print non-data lines as-is


if __name__ == "__main__":
    test_stream()
