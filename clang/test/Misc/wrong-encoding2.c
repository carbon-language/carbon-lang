// RUN: %clang_cc1 -fsyntax-only -fmessage-length=100 %s 2>&1 | FileCheck -strict-whitespace %s
// REQUIRES: asserts

int main(void) {
    "�#x#p)6�)ѽ�$��>U �h����|� থϻg�Y|`?�;;ƿVj�\\����ݪW9���:̊O�E�ېy?SK�y�����i&n";
}

// CHECK-NOT:Assertion
