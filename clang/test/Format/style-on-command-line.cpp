// RUN: grep -Ev "// *[A-Z0-9_]+:" %s > %t.cpp
// RUN: clang-format -style="{BasedOnStyle: Google, IndentWidth: 8}" %t.cpp | FileCheck -strict-whitespace -check-prefix=CHECK1 %s
// RUN: clang-format -style="{BasedOnStyle: LLVM, IndentWidth: 7}" %t.cpp | FileCheck -strict-whitespace -check-prefix=CHECK2 %s
// RUN: clang-format -style="{BasedOnStyle: invalid, IndentWidth: 7}" -fallback-style=LLVM %t.cpp 2>&1 | FileCheck -strict-whitespace -check-prefix=CHECK3 %s
// RUN: clang-format -style="{lsjd}" %t.cpp -fallback-style=LLVM 2>&1 | FileCheck -strict-whitespace -check-prefix=CHECK4 %s
// RUN: printf "BasedOnStyle: google\nIndentWidth: 5\n" > %T/.clang-format
// RUN: clang-format -style=file %t.cpp 2>&1 | FileCheck -strict-whitespace -check-prefix=CHECK5 %s
// RUN: printf "\n" > %T/.clang-format
// RUN: clang-format -style=file -fallback-style=webkit %t.cpp 2>&1 | FileCheck -strict-whitespace -check-prefix=CHECK6 %s
// RUN: rm %T/.clang-format
// RUN: printf "BasedOnStyle: google\nIndentWidth: 6\n" > %T/_clang-format
// RUN: clang-format -style=file %t.cpp 2>&1 | FileCheck -strict-whitespace -check-prefix=CHECK7 %s
// RUN: clang-format -style="{BasedOnStyle: LLVM, PointerBindsToType: true}" %t.cpp | FileCheck -strict-whitespace -check-prefix=CHECK8 %s
// RUN: clang-format -style="{BasedOnStyle: WebKit, PointerBindsToType: false}" %t.cpp | FileCheck -strict-whitespace -check-prefix=CHECK9 %s
void f() {
// CHECK1: {{^        int\* i;$}}
// CHECK2: {{^       int \*i;$}}
// CHECK3: Unknown value for BasedOnStyle: invalid
// CHECK3: Error parsing -style: {{I|i}}nvalid argument, using LLVM style
// CHECK3: {{^  int \*i;$}}
// CHECK4: Error parsing -style: {{I|i}}nvalid argument, using LLVM style
// CHECK4: {{^  int \*i;$}}
// CHECK5: {{^     int\* i;$}}
// CHECK6: {{^Error reading .*\.clang-format: (I|i)nvalid argument}}
// CHECK6: {{^    int\* i;$}}
// CHECK7: {{^      int\* i;$}}
// CHECK8: {{^  int\* i;$}}
// CHECK9: {{^    int \*i;$}}
int*i;
int j;
}

// On Windows, the 'rm' commands fail when the previous process is still alive.
// This happens enough to make the test useless.
// REQUIRES: shell
