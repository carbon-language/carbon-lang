// Check that llvm-cov loads real coverage mapping data for a function
// even though dummy coverage data for that function comes first.
// Dummy coverage data is exported if the definition of an inline function
// is seen but the function is not used in the translation unit.

// If you need to rebuild the 'covmapping' file for this test, please use
// the following commands:
// clang++ -fprofile-instr-generate -fcoverage-mapping -o tmp -x c++ prefer_used_to_unused.h prefer_used_to_unused.cpp
// llvm-cov convert-for-testing -o prefer_used_to_unused.covmapping tmp

// RUN: llvm-profdata merge %S/Inputs/prefer_used_to_unused.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/prefer_used_to_unused.covmapping -instr-profile %t.profdata -path-equivalence=/tmp/,%S %s | FileCheck %s

// Coverage data for this function has a non-zero hash value if it is used in the translation unit.
inline int sampleFunc(int A) { // CHECK:      [[@LINE]]| 1|inline int sampleFunc(int A) {
  if (A > 0)                   // CHECK-NEXT: [[@LINE]]| 1|  if (A > 0)
    return A;                  // CHECK-NEXT: [[@LINE]]| 1|    return A;
  return 0;                    // CHECK-NEXT: [[@LINE]]| 0|  return 0;
}                              // CHECK-NEXT: [[@LINE]]| 0|}

// The hash for this function is zero in both cases, either it is used in the translation unit or not.
inline int simpleFunc(int A) { // CHECK:      [[@LINE]]| 1|inline int simpleFunc(int A) {
  return A;                    // CHECK-NEXT: [[@LINE]]| 1|  return A;
}                              // CHECK-NEXT: [[@LINE]]| 1|}
