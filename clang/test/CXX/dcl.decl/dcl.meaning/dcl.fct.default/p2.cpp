// RUN: clang-cc -fsyntax-only -verify %s

void point(int = 3, int = 4);

void test_point() {
  point(1,2); 
  point(1); 
  point();
}
