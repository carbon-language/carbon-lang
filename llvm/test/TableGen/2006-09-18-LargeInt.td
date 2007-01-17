// RUN: tblgen %s | grep -- -65536

def X {
  int Y = 0xFFFF0000;
}
