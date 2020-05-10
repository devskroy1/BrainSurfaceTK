import os
import sys
import time

SETTINGS = os.path.join(os.getcwd(), "GUI/BasicSite/settings.py")
MANAGE_PATH = os.path.join(os.getcwd(), "GUI/manage.py")
MEDIA_PATH = os.path.join(os.getcwd(), "GUI/media")

args = [arg for arg in sys.argv]

# expect ordering of args dev runmodwsgi --url-alias /media media

def set_debug_to(boolv=True):
    if boolv:
        cboolv = False
    else:
        cboolv = True
    boolv = str(boolv).capitalize()
    cboolv = str(cboolv).capitalize()
    info = dict()
    with open(SETTINGS, 'r') as file:
        filedata = file.readlines()
    for i, line in enumerate(filedata):
        if "DEBUG" in line:
            filedata[i] = line.replace(cboolv, boolv)
        if "MEDIA_URL" in line:
            info["MEDIA_URL"] = line.split(" ")[-1].split('"')[-2][:-1]
    with open(SETTINGS, 'w') as file:
        file.writelines(filedata)
    return info


if __name__ == "__main__":
    if "prod" in args:
        info = set_debug_to(False)
        os.system(" ".join(["python", MANAGE_PATH, "collectstatic", "--noinput"]))
        time.sleep(2)
        os.system(" ".join(["python", MANAGE_PATH, "runmodwsgi", "--url-alias", info["MEDIA_URL"], MEDIA_PATH]))
    else:
        set_debug_to(True)
        os.system(" ".join(["python", MANAGE_PATH, "runserver"]))
