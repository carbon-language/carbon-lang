// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(bool);

void usePointer(int * const *);
void useReference(int * const &);

void testPointer() {
  int x;
  int *p;

  p = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  usePointer(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  p = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useReference(p);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  int * const cp1 = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  usePointer(&cp1);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  int * const cp2 = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useReference(cp2);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}


struct Wrapper {
  int *ptr;
};

void useStruct(Wrapper &w);
void useConstStruct(const Wrapper &w);

void testPointerStruct() {
  int x;
  Wrapper w;

  w.ptr = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}

  w.ptr = &x;
  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useConstStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}


struct RefWrapper {
  int &ref;
};

void useStruct(RefWrapper &w);
void useConstStruct(const RefWrapper &w);

void testReferenceStruct() {
  int x;
  RefWrapper w = { x };

  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}

// FIXME: This test is split into two functions because region invalidation
// does not preserve reference bindings. <rdar://problem/13320347>
void testConstReferenceStruct() {
  int x;
  RefWrapper w = { x };

  x = 42;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  useConstStruct(w);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
}


void usePointerPure(int * const *) __attribute__((pure));
void usePointerConst(int * const *) __attribute__((const));

void testPureConst() {
  extern int global;
  int x;
  int *p;

  p = &x;
  x = 42;
  global = -5;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(global == -5); // expected-warning{{TRUE}}

  usePointerPure(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(global == -5); // expected-warning{{TRUE}}

  usePointerConst(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(global == -5); // expected-warning{{TRUE}}

  usePointer(&p);
  clang_analyzer_eval(x == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(global == -5); // expected-warning{{UNKNOWN}}
}


