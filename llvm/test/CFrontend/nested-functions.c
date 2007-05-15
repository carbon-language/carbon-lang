// RUN: %llvmgcc -S %s -o -  -fnested-functions
void Bork() {
  void Fork(const int *src, int size) {
    int i = 1;
    int x;

    while (i < size)
      x = src[i];
  }
}
