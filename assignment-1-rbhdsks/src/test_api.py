import requests

API_URL = "http://127.0.0.1:8000/predict"

def test_api():
    # Example test input
    payload = {
        "Type": 0,
        "Air_temperature_K": 300,
        "Process_temperature_K": 310,
        "Rotational_speed_rpm": 1500,
        "Torque_Nm": 40,
        "Tool_wear_min": 100
    }

    try:
        response = requests.post(API_URL, json=payload)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("Response JSON:", data)

            # Basic validation checks
            assert "prediction" in data
            assert "probability" in data
            assert data["prediction"] in [0, 1]
            assert 0.0 <= data["probability"] <= 1.0

            print(" API Test Passed Successfully!")

        else:
            print(" API returned error.")
            print(response.text)

    except Exception as e:
        print(" Failed to connect to API.")
        print(str(e))


if __name__ == "__main__":
    test_api()
