// Test for "sancov.py missing ...".

// RUN: ASAN_OPTIONS=coverage=1:coverage_dir=%T/coverage-missing

// First case: coverage from executable. main() is called on every code path;
// other than that, the foo and bar code paths are complementary in terms of
// PCs covered.
// RUN: %clangxx_asan -fsanitize-coverage=1 %s -o %t -DFOOBAR -DMAIN
// RUN: rm -rf %T/coverage-missing
// RUN: mkdir -p %T/coverage-missing
// RUN: cd %T/coverage-missing
// RUN: ASAN_OPTIONS=$ASAN_OPTIONS %t
// RUN: %sancov print *.sancov > main.txt
// RUN: rm *.sancov
// RUN: [ $(cat main.txt | wc -l) == 1 ]
// RUN: ASAN_OPTIONS=$ASAN_OPTIONS %t x
// RUN: %sancov print *.sancov > foo.txt
// RUN: rm *.sancov
// RUN: [ $(cat foo.txt | wc -l) == 3 ]
// RUN: ASAN_OPTIONS=$ASAN_OPTIONS %t x x
// RUN: %sancov print *.sancov > bar.txt
// RUN: rm *.sancov
// RUN: [ $(cat bar.txt | wc -l) == 4 ]
// RUN: %sancov missing %t < foo.txt > foo-missing.txt
// RUN: sort main.txt foo-missing.txt -o foo-missing-with-main.txt
// RUN: diff bar.txt foo-missing-with-main.txt

// Second case: coverage from DSO. Strictly complementary code paths.
// cd %T
// RUN: %clangxx_asan -fsanitize-coverage=1 %s -o %dynamiclib -DFOOBAR -shared -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=1 %s %dynamiclib -o %t -DMAIN
// RUN: LIBNAME=`basename %dynamiclib`
// RUN: rm -rf %T/coverage-missing
// RUN: mkdir -p %T/coverage-missing
// RUN: cd %T/coverage-missing
// RUN: ASAN_OPTIONS=$ASAN_OPTIONS %t x
// RUN: %sancov print $LIBNAME.*.sancov > foo.txt
// RUN: rm *.sancov
// RUN: [ $(cat foo.txt | wc -l) == 2 ]
// RUN: ASAN_OPTIONS=$ASAN_OPTIONS %t x x
// RUN: %sancov print $LIBNAME.*.sancov > bar.txt
// RUN: rm *.sancov
// RUN: [ $(cat bar.txt | wc -l) == 3 ]
// RUN: %sancov missing %dynamiclib < foo.txt > foo-missing.txt
// RUN: diff bar.txt foo-missing.txt

// REQUIRES: x86_64-supported-target, i386-supported-target
// XFAIL: android

#include <stdio.h>

void foo1();
void foo2();
void bar1();
void bar2();
void bar3();

#if defined(FOOBAR)
void foo1() { fprintf(stderr, "foo1\n"); }
void foo2() { fprintf(stderr, "foo2\n"); }

void bar1() { fprintf(stderr, "bar1\n"); }
void bar2() { fprintf(stderr, "bar2\n"); }
void bar3() { fprintf(stderr, "bar3\n"); }
#endif

#if defined(MAIN)
int main(int argc, char **argv) {
  switch (argc) {
    case 1:
      break;
    case 2:
      foo1();
      foo2();
      break;
    case 3:
      bar1();
      bar2();
      bar3();
      break;
  }
}
#endif
