// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-feature +avx | FileCheck %s

namespace test1 {
typedef double __m256d __attribute__((__vector_size__(32)));

class PR22753 {
public:
  __m256d data;
};

// CHECK: define <4 x double> @_ZN5test14testENS_7PR22753E(<4 x double>
PR22753 test(PR22753 x) {
  return x;
}
}
