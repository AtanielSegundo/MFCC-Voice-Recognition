import sys

sys.stdout.reconfigure(encoding='utf-8')
OUT = sys.stdout

def setOut(p:str):
    global OUT    
    OUT = p

def LOG(tag: str, msg: str, out=sys.stdout):
    formatted_msg = f"[{tag}] {msg}\n"
    if isinstance(out, str):
        with open(out, "a", encoding="utf-8") as f:
            f.write(formatted_msg)
    else:
        out.write(formatted_msg)
        out.flush()

def NEWLINE(n=1,out=None):
    out = out or OUT
    if isinstance(out, str):
        with open(out, "a", encoding="utf-8") as f:
            f.write("\n" * n)
    else:
        out.write("\n" * n)
        out.flush()

def ERROR(msg: str, out=None):    
    out = out or OUT
    LOG("ERROR", msg, out)
    sys.exit(1)

def INFO(msg: str, out=None):
    out = out or OUT
    LOG("INFO", msg, out)

def WARNING(msg: str, out=None):    
    out = out or OUT
    LOG("WARNING", msg, out)