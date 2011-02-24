// RUN: %clang_cc1 -triple i386-apple-darwin9 -analyze -analyzer-checker=core.experimental -analyzer-check-objc-mem -analyzer-store=region -verify -fblocks -analyzer-opt-analyze-nested-blocks %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -analyze -analyzer-checker=core.experimental -analyzer-check-objc-mem -analyzer-store=region -verify -fblocks   -analyzer-opt-analyze-nested-blocks %s

//===------------------------------------------------------------------------------------------===//
// This files tests our path-sensitive handling of Objective-c++ files.
//===------------------------------------------------------------------------------------------===//

// Test basic handling of references.
char &test1_aux();
char *test1() {
  return &test1_aux();
}

// Test test1_aux() evaluates to char &.
char test1_as_rvalue() {
  return test1_aux();
}

// Test basic handling of references with Objective-C classes.
@interface Test1
- (char&) foo;
@end

char* Test1_harness(Test1 *p) {
  return &[p foo];
}

char Test1_harness_b(Test1 *p) {
  return [p foo];
}

