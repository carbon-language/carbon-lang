// RUN: %llvmgxx -S %s -o - | grep cX

namespace C {
  int c = 1;
  namespace {
    int cX = 6;
    void marker2() {
     cX;
    }
  }
}

int main() {
  C::marker2();
  return 0;
}
