//===-- asan_benchmarks_test.cc ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Some benchmarks for the instrumented code.
//===----------------------------------------------------------------------===//

#include "asan_test_config.h"
#include "asan_test_utils.h"

template<class T>
__attribute__((noinline))
static void ManyAccessFunc(T *x, size_t n_elements, size_t n_iter) {
  for (size_t iter = 0; iter < n_iter; iter++) {
    break_optimization(0);
    // hand unroll the loop to stress the reg alloc.
    for (size_t i = 0; i <= n_elements - 16; i += 16) {
      x[i + 0] = i;
      x[i + 1] = i;
      x[i + 2] = i;
      x[i + 3] = i;
      x[i + 4] = i;
      x[i + 5] = i;
      x[i + 6] = i;
      x[i + 7] = i;
      x[i + 8] = i;
      x[i + 9] = i;
      x[i + 10] = i;
      x[i + 11] = i;
      x[i + 12] = i;
      x[i + 13] = i;
      x[i + 14] = i;
      x[i + 15] = i;
    }
  }
}

TEST(AddressSanitizer, ManyAccessBenchmark) {
  size_t kLen = 1024;
  int *int_array = new int[kLen];
  ManyAccessFunc(int_array, kLen, 1 << 24);
  delete [] int_array;
}

// access 7 char elements in a 7 byte array (i.e. on the border).
__attribute__((noinline))
static void BorderAccessFunc(char *x, size_t n_iter) {
  for (size_t iter = 0; iter < n_iter; iter++) {
    break_optimization(x);
    x[0] = 0;
    x[1] = 0;
    x[2] = 0;
    x[3] = 0;
    x[4] = 0;
    x[5] = 0;
    x[6] = 0;
  }
}

TEST(AddressSanitizer, BorderAccessBenchmark) {
  char *char_7_array = new char[7];
  BorderAccessFunc(char_7_array, 1 << 30);
  delete [] char_7_array;
}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
