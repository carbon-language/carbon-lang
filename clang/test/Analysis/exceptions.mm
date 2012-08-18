// RUN: %clang_cc1 -analyze -fexceptions -fobjc-exceptions -fcxx-exceptions -analyzer-ipa=inlining -analyzer-checker=core,unix.Malloc,debug.ExprInspection -verify %s

void clang_analyzer_checkInlined(bool);

typedef typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);


id getException();
void inlinedObjC() {
  clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
  @throw getException();
}

int testObjC() {
  int a; // uninitialized
  void *mem = malloc(4); // no-warning (ObjC exceptions are usually fatal)
  inlinedObjC();
  free(mem);
  return a; // no-warning
}

