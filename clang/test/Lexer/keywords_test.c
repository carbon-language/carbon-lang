// RUN: %clang_cc1 -std=c99 -E %s -o - | FileCheck --check-prefix=CHECK-NONE %s

// RUN: %clang_cc1 -std=gnu89 -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-GNU-KEYWORDS %s
// RUN: %clang_cc1 -std=c99 -fgnu-keywords -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-GNU-KEYWORDS %s
// RUN: %clang_cc1 -std=gnu89 -fno-gnu-keywords -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-NONE %s

// RUN: %clang_cc1 -std=c99 -fms-extensions -E %s -o - \
// RUN:     | FileCheck --check-prefix=CHECK-MS-KEYWORDS %s

void f() {
// CHECK-NONE: int asm
// CHECK-GNU-KEYWORDS: asm ("ret" : :)
#if __is_identifier(asm)
  int asm;
#else
  asm ("ret" : :);
#endif
}

// CHECK-NONE: no_ms_wchar
// CHECK-MS-KEYWORDS: has_ms_wchar
#if __is_identifier(__wchar_t)
void no_ms_wchar();
#else
void has_ms_wchar();
#endif
