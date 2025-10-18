"""
Just for ptint in any color you want
"""
def blue(x : str)-> str :
    return '\033[34m' + x + "\033[00m"

def red(x : str)-> str :
    return '\033[91m' + x +  "\033[00m"

def green(x : str)-> str :
    return '\033[92m' + x +  "\033[00m"

def purple(x : str)-> str :
    return '\033[95m' + x +  "\033[00m"

def cyan(x : str)-> str :
    return "\033[96m" + x +  "\033[00m"

def yellow(x : str)-> str :
    return '\033[33m' + x + '\033[0m'

if __name__ == "__main__":
    print(blue("hello world"))
    print(red("hello world"))
    print(green("hello world"))
    print(purple("hello world"))
    print(cyan("hello world"))
    print(yellow("hello world"))