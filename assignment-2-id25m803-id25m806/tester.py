#Load important libraries
import requests
from collections import Counter
import concurrent.futures


# Change the API endpoint according to needs
URL = "http://localhost:8000/api/v1/ner"

PAYLOAD = {
    "text": "Apple is hiring engineers in London."
}


def call_api():
    """
    Function to call the API
    
    Returns:
        str: Container ID

    """
    try:
        response = requests.post(URL, json=PAYLOAD, timeout=5)# Make the request
        data = response.json() # Parse the response
        return data.get("container_id", "No ID")
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """
    Main function
    
    Returns:
        None
    """
    print("Sending 100 requests\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(lambda _: call_api(), range(100)))

    counts = Counter(results)

    print("Load Balancing Results:\n")
    for container, count in counts.items():
        print(f"{container} -> {count} requests")


if __name__ == "__main__":
    main()
