// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// expected-warning@+1 {{parameter 'ZZZZZZZZZZ' not found in the function declaration}} expected-note@+1 {{did you mean 'a'?}}
/// \param ZZZZZZZZZZ Blah blah.
int test1(int a);

// expected-warning@+1 {{parameter 'aab' not found in the function declaration}} expected-note@+1 {{did you mean 'aaa'?}}
/// \param aab Blah blah.
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

// CHECK: fix-it:"{{.*}}":{5:12-5:22}:"a"
// CHECK: fix-it:"{{.*}}":{9:12-9:15}:"aaa"
// CHECK: fix-it:"{{.*}}":{13:13-13:23}:"T"
// CHECK: fix-it:"{{.*}}":{18:13-18:18}:"SomeTy"

