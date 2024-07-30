import os
import time


class Server:
    def __init__(self, state_repr_type, game_version, if_head, headless_server = True):
        self.if_head = if_head
        self.state_repr_type = state_repr_type
        self.game_version = game_version
        self.headless_server = headless_server

    def start(self):
        # if server already started, do nothing
        server_procs = os.popen('ps -u | grep "game_playing_interface.jar"').read().split("\n")[:-1]
        for proc in server_procs:
            if 'grep' not in proc:
                return None
        if not self.if_head:
            if self.state_repr_type == 'symbolic':
                if self.headless_server:
                    os.system(
                        "bash -c \"cd ../sciencebirdsgames/{} && nohup java -jar ./game_playing_interface.jar --headless --dev > out 2>&1 &\"".format(
                            self.game_version))

                else:
                    os.system(
                        "gnome-terminal -- bash -c \"cd ../sciencebirdsgames/{} && java -jar ./game_playing_interface.jar --headless --dev \"".format(
                            self.game_version))

            else:
                if self.headless_server:
                    os.system(
                        "bash -c \"cd ../sciencebirdsgames/{} && nohup java -jar ./game_playing_interface.jar --dev > out 2>&1 &\"".format(
                            self.game_version))
                else:
                    os.system(
                        "gnome-terminal -- bash -c \"cd ../sciencebirdsgames/{} && java -jar ./game_playing_interface.jar --dev \"".format(
                            self.game_version))

        elif self.if_head == 'headless':
            if self.headless_server:
                os.system(
                    "bash -c \"cd ../sciencebirdsgames/{} && nohup java -jar ./game_playing_interface.jar --headless --dev > out 2>&1 &\"".format(
                        self.game_version))
            else:
                os.system(
                    "gnome-terminal -- bash -c \"cd ../sciencebirdsgames/{} && java -jar ./game_playing_interface.jar --headless --dev \"".format(
                        self.game_version))
        else:
            if self.headless_server:
                os.system(
                    "bash -c \"cd ../sciencebirdsgames/{} && nohup java -jar ./game_playing_interface.jar --dev > out 2>&1 &\"".format(
                        self.game_version))
            else:
                os.system(
                    "gnome-terminal -- bash -c \"cd ../sciencebirdsgames/{} && java -jar ./game_playing_interface.jar --dev \"".format(
                        self.game_version))
        # logger.debug("Server started...")
        print("Server started...")
        time.sleep(2)

    def close(self):
        os.system("ps -ef | grep \"[j]ava -jar ./game_playing_interface.jar\" | awk '{ print $2 }' | xargs kill -9 ")
        os.system("ps -ef | grep \"[.]/9001.x86_64\" | awk '{ print $2 }' | xargs kill -9 ")
