// RUN: %clangxx -fsanitize=integer-divide-by-zero -DDIVIDEND=0 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=integer-divide-by-zero -DDIVIDEND=1U %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=float-divide-by-zero -DDIVIDEND=1.5 %s -o %t && %t 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=integer-divide-by-zero -DDIVIDEND='intmax(123)' %s -o %t && %t 2>&1 | FileCheck %s

#ifdef __SIZEOF_INT128__
typedef __int128 intmax;
#else
typedef long long intmax;
#endif

int main() {
  // CHECK: div-zero.cpp:[[@LINE+1]]:12: runtime error: division by zero
  DIVIDEND / 0;
}
