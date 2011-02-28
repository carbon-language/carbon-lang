// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-inline-call -analyzer-store region -verify %s

// Test when entering f1(), we set the right AnalysisContext to Environment.
// Otherwise, block-level expr '1 && a' would not be block-level.
int a;

void f1() {
  if (1 && a)
    return;
}

void f2() {
  f1();
}
