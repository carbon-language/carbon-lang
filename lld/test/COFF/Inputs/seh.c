__declspec(noinline) void triggerSEH() {
  volatile int *p = 0;
  *p = 1;
}

int main() {
  __try {
    triggerSEH();
  } __except(1) {
    return 42;
  }
  return 0;
}
