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
extern void invalidateGlobals();

void testGlobalInvalidation() {
  if (globalInt != 42)
    return;
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}

  invalidateGlobals();
  clang_analyzer_eval(globalInt == 42); // expected-warning{{UNKNOWN}}
}


//---------------------------------
// False negatives
//---------------------------------

void testGlobalInvalidationWithDirectBinding() {
  globalInt = 42;
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}

  invalidateGlobals();
  // FIXME: Should be UNKNOWN.
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}
}
