// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=debug.ExprInspection \
// RUN:  -verify %s 2>&1 | FileCheck %s

// Self-tests for the debug.ExprInspection checker.

void clang_analyzer_dump(int x);
void clang_analyzer_printState();
void clang_analyzer_numTimesReached();

void foo(int x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump(x + (-1)); // expected-warning{{(reg_$0<int x>) + -1}}
  int y = 1;
  for (; y < 3; ++y) {
    clang_analyzer_numTimesReached(); // expected-warning{{2}}

    if (y == 2) {
      int z = x > 13;
      if (!z)
        clang_analyzer_printState();
    }
  }
}

// CHECK:      "store": [
// CHECK-NEXT:   { "cluster": "y", "items": [
// CHECK-NEXT:     { "kind": "Direct", "offset": 0, "value": "2 S32b" }
// CHECK-NEXT:   ]}
// CHECK-NEXT: ]

// CHECK:      Expressions by stack frame:
// CHECK-NEXT: #0 Calling foo
// CHECK-NEXT: (LC1, S847) clang_analyzer_printState : &code{clang_analyzer_printState}

// CHECK:      Ranges of symbol values:
// CHECK-NEXT:  reg_$0<int x> : { [-2147483648, 13] }
