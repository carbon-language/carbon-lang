// RUN: %clang_cc1 -triple i386-apple-darwin9 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify -fblocks -analyzer-opt-analyze-nested-blocks %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify -fblocks   -analyzer-opt-analyze-nested-blocks %s
// XFAIL: *

// Test basic handling of references.
char &test1_aux();
char *test1() {
  return &test1_aux();
}

// This test currently crasehs because test1_aux() evaluates to a 'char' instead of a char& in CFRefCount.cpp.
char test1_as_rvalue() {
  return test1_aux();
}

