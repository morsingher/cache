from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import sys

APP_URL = "https://cache-finance.streamlit.app"

def wake_up():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print(f"Opening {APP_URL}...")
        driver.get(APP_URL)
        
        # Wait up to 20 seconds to see if the "Wake up" button appears
        print("Searching for the wake-up button...")
        try:
            # This looks for a button that contains the text 'back up' 
            # which is part of 'Yes, get this app back up!'
            wait = WebDriverWait(driver, 20)
            button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'back up')]")))
            
            button.click()
            print("Button found and clicked! Waking up the oven...")
            
            # Give it time to actually boot after the click
            time.sleep(30) 
        except:
            print("Button not found. The app might already be awake or loading.")
            
        print(f"Final Page Title: {driver.title}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)
    finally:
        driver.quit()

if __name__ == "__main__":
    wake_up()