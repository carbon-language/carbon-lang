// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

void use(int);
id foo(int x) {
  if (x)
    return 0;
  static id p = foo(1); 
    clang_analyzer_eval(p == 0); // expected-warning{{TRUE}}
  return p;
}

const int &globalIntRef = 42;

void testGlobalRef() {
  // FIXME: Should be TRUE, but should at least not crash.
  clang_analyzer_eval(globalIntRef == 42); // expected-warning{{UNKNOWN}}
}

extern int globalInt;
extern struct {
  int value;
} globalStruct;
extern void invalidateGlobals();

void testGlobalInvalidation() {
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{UNKNOWN}}

  if (globalInt != 42)
    return;
  if (globalStruct.value != 43)
    return;
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{TRUE}}

  invalidateGlobals();
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{UNKNOWN}}
}

void testGlobalInvalidationWithDirectBinding() {
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{UNKNOWN}}

  globalInt = 42;
  globalStruct.value = 43;
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{TRUE}}

  invalidateGlobals();
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalStruct.value == 43); // expected-warning{{UNKNOWN}}
}
