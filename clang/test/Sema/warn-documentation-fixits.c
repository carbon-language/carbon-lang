// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

/// \param ZZZZZZZZZZ Blah blah. expected-warning {{parameter 'ZZZZZZZZZZ' not found in the function declaration}}  expected-note {{did you mean 'a'?}}
int test1(int a);

/// \param aab Blah blah. expected-warning {{parameter 'aab' not found in the function declaration}}  expected-note {{did you mean 'aaa'?}}
int test2(int aaa, int bbb);

// CHECK: fix-it:"{{.*}}":{4:12-4:22}:"a"
// CHECK: fix-it:"{{.*}}":{7:12-7:15}:"aaa"

