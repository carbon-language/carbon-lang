// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

/// \param ZZZZZZZZZZ Blah blah. expected-warning {{parameter 'ZZZZZZZZZZ' not found in the function declaration}}  expected-note {{did you mean 'a'?}}
int test1(int a);

/// \param aab Blah blah. expected-warning {{parameter 'aab' not found in the function declaration}}  expected-note {{did you mean 'aaa'?}}
int test2(int aaa, int bbb);

// expected-warning@+1 {{template parameter 'ZZZZZZZZZZ' not found in the template declaration}} expected-note@+1 {{did you mean 'T'?}}
/// \tparam ZZZZZZZZZZ Aaa
template<typename T>
void test3(T aaa);

// expected-warning@+1 {{template parameter 'SomTy' not found in the template declaration}} expected-note@+1 {{did you mean 'SomeTy'?}}
/// \tparam SomTy Aaa
/// \tparam OtherTy Bbb
template<typename SomeTy, typename OtherTy>
void test4(SomeTy aaa, OtherTy bbb);

// CHECK: fix-it:"{{.*}}":{4:12-4:22}:"a"
// CHECK: fix-it:"{{.*}}":{7:12-7:15}:"aaa"
// CHECK: fix-it:"{{.*}}":{11:13-11:23}:"T"
// CHECK: fix-it:"{{.*}}":{16:13-16:18}:"SomeTy"

