#!/usr/bin/python3.7
"""
Scrape using Compute vision approch
"""
import asyncio
import os

import cv2
from matplotlib import pyplot as plt
from pyppeteer import launch

path = os.path.dirname(os.path.realpath(__file__))

args = [
    # '--no-sandbox',
    # '--disable-setuid-sandbox',
    "--disable-infobars",
    # '--window-position=0,0',
    "--ignore-certifcate-errors",
    "--ignore-certifcate-errors-spki-list",
    '--user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3312.0 Safari/537.36"',
    # '--proxy-server=https://157.230.179.73:8080',
]


async def make_screenshot():
    success = False
    url = "https://www.ftadviser.com"
    # while success is False:
    # proxy_id = proxy.get_random_proxy()
    try:
        browser = await launch(args=args)
        page = await browser.newPage()
        await page.setViewport({"width": 1920, "height": 1000})
        await page.goto(url)
        await page.waitFor(10000)
        # await retry(() => page.goto('http://localhost:3000'), 1000)
        await page.screenshot({"path": path + "/screenshot.png"})
        await browser.close()
        # success = True
    except Exception as identifier:
        print(identifier)


def match():
    img = cv2.imread(path + "/screenshot.png", 0)
    # img2 = img.copy()
    template = cv2.imread(path + "/logo.png", 0)
    w, h = template.shape[::-1]
    # img = img2.copy()
    method = cv2.TM_CCOEFF_NORMED

    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val, max_val, min_loc, max_loc)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 244, 2)

    plt.subplot(121), plt.imshow(res, cmap="gray")
    plt.title("Matching Result"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap="gray")
    plt.title("Detected Point"), plt.xticks([]), plt.yticks([])
    plt.suptitle("cv2.TM_CCOEFF_NORMED")

    plt.show()


if __name__ == "__main__":
    # asyncio.get_event_loop().run_until_complete(make_screenshot())
    match()
