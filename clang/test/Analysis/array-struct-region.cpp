// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core,debug.ExprInspection -verify -x c %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core,debug.ExprInspection -verify -x c++ -analyzer-config c++-inlining=constructors %s

void clang_analyzer_eval(int);

struct S {
  int field;

#if __cplusplus
  const struct S *getThis() const { return this; }
#endif
};

struct S getS();


void testAssignment() {
  struct S s = getS();

  if (s.field != 42) return;
  clang_analyzer_eval(s.field == 42); // expected-warning{{TRUE}}

  s.field = 0;
  clang_analyzer_eval(s.field == 0); // expected-warning{{TRUE}}

#if __cplusplus
  clang_analyzer_eval(s.getThis() == &s); // expected-warning{{TRUE}}
#endif
}


void testImmediateUse() {
  int x = getS().field;

  if (x != 42) return;
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}

#if __cplusplus
  clang_analyzer_eval((void *)getS().getThis() == (void *)&x); // expected-warning{{FALSE}}
#endif
}

int getConstrainedField(struct S s) {
  if (s.field != 42) return 42;
  return s.field;
}

int getAssignedField(struct S s) {
  s.field = 42;
  return s.field;
}

void testArgument() {
  clang_analyzer_eval(getConstrainedField(getS()) == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(getAssignedField(getS()) == 42); // expected-warning{{TRUE}}
}


//--------------------
// C++-only tests
//--------------------

#if __cplusplus
void testReferenceAssignment() {
  const S &s = getS();

  if (s.field != 42) return;
  clang_analyzer_eval(s.field == 42); // expected-warning{{TRUE}}

  clang_analyzer_eval(s.getThis() == &s); // expected-warning{{TRUE}}
}


int getConstrainedFieldRef(const S &s) {
  if (s.field != 42) return 42;
  return s.field;
}

bool checkThis(const S &s) {
  return s.getThis() == &s;
}

void testReferenceArgument() {
  clang_analyzer_eval(getConstrainedFieldRef(getS()) == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(checkThis(getS())); // expected-warning{{TRUE}}
}
#endif
