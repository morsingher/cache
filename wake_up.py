from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import sys

# Terrible hack to keep the app awake

APP_URL = "https://cache-finance.streamlit.app"

def wake_up():
    print(f"Waking up {APP_URL}...")
    
    # Configure Chrome to run without a window (headless)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(APP_URL)
        # Wait 60 seconds for the app to load resources
        time.sleep(60) 
        print(f"Page title: {driver.title}")
        print("Success: App loaded.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        driver.quit()

if __name__ == "__main__":
    wake_up()