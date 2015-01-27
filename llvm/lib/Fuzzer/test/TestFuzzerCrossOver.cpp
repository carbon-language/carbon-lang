#include "FuzzerInternal.h"

int main() {
  using namespace fuzzer;
  Unit A({0, 1, 2, 3, 4}), B({5, 6, 7, 8, 9});
  Unit C;
  for (size_t Len = 1; Len < 15; Len++) {
    for (int Iter = 0; Iter < 1000; Iter++) {
      CrossOver(A, B, &C, Len);
      Print(C);
    }
  }
}
