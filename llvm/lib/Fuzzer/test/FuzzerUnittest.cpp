#include "FuzzerInternal.h"
#include "gtest/gtest.h"
#include <set>

using namespace fuzzer;

// For now, have LLVMFuzzerTestOneInput just to make it link.
// Later we may want to make unittests that actually call LLVMFuzzerTestOneInput.
extern "C" void LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  abort();
}

TEST(Fuzzer, CrossOver) {
  FuzzerRandomLibc Rand(0);
  Unit A({0, 1, 2}), B({5, 6, 7});
  Unit C;
  Unit Expected[] = {
       { 0 },
       { 0, 1 },
       { 0, 5 },
       { 0, 1, 2 },
       { 0, 1, 5 },
       { 0, 5, 1 },
       { 0, 5, 6 },
       { 0, 1, 2, 5 },
       { 0, 1, 5, 2 },
       { 0, 1, 5, 6 },
       { 0, 5, 1, 2 },
       { 0, 5, 1, 6 },
       { 0, 5, 6, 1 },
       { 0, 5, 6, 7 },
       { 0, 1, 2, 5, 6 },
       { 0, 1, 5, 2, 6 },
       { 0, 1, 5, 6, 2 },
       { 0, 1, 5, 6, 7 },
       { 0, 5, 1, 2, 6 },
       { 0, 5, 1, 6, 2 },
       { 0, 5, 1, 6, 7 },
       { 0, 5, 6, 1, 2 },
       { 0, 5, 6, 1, 7 },
       { 0, 5, 6, 7, 1 },
       { 0, 1, 2, 5, 6, 7 },
       { 0, 1, 5, 2, 6, 7 },
       { 0, 1, 5, 6, 2, 7 },
       { 0, 1, 5, 6, 7, 2 },
       { 0, 5, 1, 2, 6, 7 },
       { 0, 5, 1, 6, 2, 7 },
       { 0, 5, 1, 6, 7, 2 },
       { 0, 5, 6, 1, 2, 7 },
       { 0, 5, 6, 1, 7, 2 },
       { 0, 5, 6, 7, 1, 2 }
  };
  for (size_t Len = 1; Len < 8; Len++) {
    std::set<Unit> FoundUnits, ExpectedUnitsWitThisLength;
    for (int Iter = 0; Iter < 3000; Iter++) {
      C.resize(Len);
      size_t NewSize = CrossOver(A.data(), A.size(), B.data(), B.size(),
                                 C.data(), C.size(), Rand);
      C.resize(NewSize);
      FoundUnits.insert(C);
    }
    for (const Unit &U : Expected)
      if (U.size() <= Len)
        ExpectedUnitsWitThisLength.insert(U);
    EXPECT_EQ(ExpectedUnitsWitThisLength, FoundUnits);
  }
}

TEST(Fuzzer, Hash) {
  uint8_t A[] = {'a', 'b', 'c'};
  fuzzer::Unit U(A, A + sizeof(A));
  EXPECT_EQ("a9993e364706816aba3e25717850c26c9cd0d89d", fuzzer::Hash(U));
  U.push_back('d');
  EXPECT_EQ("81fe8bfe87576c3ecb22426f8e57847382917acf", fuzzer::Hash(U));
}

typedef size_t (*Mutator)(uint8_t *Data, size_t Size, size_t MaxSize,
                          FuzzerRandomBase &Rand);

void TestEraseByte(Mutator M, int NumIter) {
  uint8_t REM0[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM1[8] = {0x00, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM2[8] = {0x00, 0x11, 0x33, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM3[8] = {0x00, 0x11, 0x22, 0x44, 0x55, 0x66, 0x77};
  uint8_t REM4[8] = {0x00, 0x11, 0x22, 0x33, 0x55, 0x66, 0x77};
  uint8_t REM5[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x66, 0x77};
  uint8_t REM6[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x77};
  uint8_t REM7[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
  FuzzerRandomLibc Rand(0);
  int FoundMask = 0;
  for (int i = 0; i < NumIter; i++) {
    uint8_t T[8] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
    size_t NewSize = Mutate_EraseByte(T, sizeof(T), sizeof(T), Rand);
    EXPECT_EQ(7UL, NewSize);
    if (!memcmp(REM0, T, 7)) FoundMask |= 1 << 0;
    if (!memcmp(REM1, T, 7)) FoundMask |= 1 << 1;
    if (!memcmp(REM2, T, 7)) FoundMask |= 1 << 2;
    if (!memcmp(REM3, T, 7)) FoundMask |= 1 << 3;
    if (!memcmp(REM4, T, 7)) FoundMask |= 1 << 4;
    if (!memcmp(REM5, T, 7)) FoundMask |= 1 << 5;
    if (!memcmp(REM6, T, 7)) FoundMask |= 1 << 6;
    if (!memcmp(REM7, T, 7)) FoundMask |= 1 << 7;
  }
  EXPECT_EQ(FoundMask, 255);
}

TEST(FuzzerMutate, EraseByte1) { TestEraseByte(Mutate_EraseByte, 50); }
TEST(FuzzerMutate, EraseByte2) { TestEraseByte(Mutate, 100); }
