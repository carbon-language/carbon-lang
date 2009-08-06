// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -analyzer-constraints=range %s -verify

//===-- unions-region.m ---------------------------------------------------===//
//
// This file tests the analyzer's reasoning about unions.
//
//===----------------------------------------------------------------------===//

// When using RegionStore, this test case previously had a false positive
// of a 'pass-by-value argument is uninitialized' warning at the call to
// 'testA_aux'.

union u_testA {
  unsigned i;
  float f;
};
 
float testA(float f) {
  int testA_aux(unsigned x);
  int testA_aux_2(union u_testA z);
  
  union u_testA swap;
  swap.f = f;

  if (testA_aux(swap.i))  // no-warning
    swap.i = ((swap.i & 0xffff0000) >> 16) | ((swap.i & 0x0000fffff) << 16);

  testA_aux_2(swap); // no-warning

  return swap.f;  
}

