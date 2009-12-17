// RUN: %clang_cc1 -triple i386-apple-darwin9 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify -fblocks -analyzer-opt-analyze-nested-blocks %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify -fblocks   -analyzer-opt-analyze-nested-blocks %s
// XFAIL: *

// This test case currently crashes because of an assertion failure.
char &test1_aux();
char *test1() {
  return &test1_aux();
}
