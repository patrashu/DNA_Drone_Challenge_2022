from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import urllib.request # download image
import time # sleep
import os # mkdir
import signal # timeout

class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print("Time is up")
    raise TimeOutException()

path = '../datasets/crawl' # root
driver = webdriver.Chrome("./chromedriver") # 크롬드라이버 경로
keywords = [] # 키워드 [a, b, c, d, e]

for keyword in keywords:
    img_path = f'{path}/{keyword}'
    os.makedirs(img_path, exist_ok=True)
    driver.get("https://www.google.co.kr/imghp?hl=ko&authuser=0&ogbl")
    driver.maximize_window()
    time.sleep(2)

    driver.find_element_by_css_selector("input.gLFyf").send_keys(keyword)
    driver.find_element_by_css_selector("input.gLFyf").send_keys(Keys.RETURN)
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element_by_css_selector(".mye4qd").click()
            except:
                break
        last_height = new_height

    list = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
    print(len(list))
    i=0

    for img in list:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, 10) # 10초안에 안끝나면 건너뛰기
        ActionChains(driver).click(img).perform()
        time.sleep(3)

        try: # 안돌아가면 xpath 수정
            imgurl = driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute(
            "src")
            urllib.request.urlretrieve(imgurl, img_path+"/"+str(keyword)+str(i)+".jpg")
            i += 1
        except:
            continue
        finally:
            signal.alarm(0)