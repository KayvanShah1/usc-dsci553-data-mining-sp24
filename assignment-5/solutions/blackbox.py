import random


class BlackBox:
    def ask(self, file, num):
        lines = open(file, "r").readlines()
        users = [0 for i in range(num)]
        for i in range(num):
            users[i] = lines[random.randint(0, len(lines) - 1)].rstrip("\n")
        return users
