// RUN: %clang_cc1 -std=c++11 -g -emit-llvm -g -triple x86_64-apple-darwin %s -o %t
// RUN: cat %t | FileCheck %s -check-prefix=CHECK0
// RUN: cat %t | FileCheck %s -check-prefix=CHECK1
// RUN: cat %t | FileCheck %s -check-prefix=CHECK2
//
// This test ensures that we associate a declaration with the
// definition of the constructor for OuterClass. The declaration is
// necessary so the backend can emit the DW_AT_specification attribute
// for the definition.
//
// rdar://problem/13116508

class Foo;
class OuterClass
{
  static class InnerClass {
  public:
    InnerClass(); // Here createContextChain() generates a limited type for OuterClass.
  } theInnerClass;
// CHECK0: [[DECL:[0-9]+]] = {{.*}} ; [ DW_TAG_subprogram ] [line [[@LINE+1]]] [OuterClass]
  OuterClass(const Foo *); // line 10
};
OuterClass::InnerClass OuterClass::theInnerClass; // This toplevel decl causes InnerClass to be generated.
// CHECK0: !"0x2e\00OuterClass\00{{.*}}\00[[@LINE+1]]"{{.*}}, ![[DECL]], {{![0-9]+}}} ; [ DW_TAG_subprogram ] [line [[@LINE+1]]] [def] [OuterClass]
OuterClass::OuterClass(const Foo *meta) { } // line 13






class Foo1;
class OuterClass1
{
  static class InnerClass1 {
  public:
    InnerClass1();
  } theInnerClass1;
// CHECK1: [[DECL:[0-9]+]] = {{.*}} ; [ DW_TAG_subprogram ] [line [[@LINE+2]]] [Bar]
// CHECK1: !"0x2e\00Bar\00{{.*}}\00[[@LINE+4]]"{{.*}}, ![[DECL]], {{![0-9]+}}} ; [ DW_TAG_subprogram ] [line [[@LINE+4]]] [def] [Bar]
  void Bar(const Foo1 *);
};
OuterClass1::InnerClass1 OuterClass1::theInnerClass1;
void OuterClass1::Bar(const Foo1 *meta) { }





class Foo2;
class OuterClass2
{
  static class InnerClass2 {
  public:
    InnerClass2();
  } theInnerClass2;
// CHECK2: [[DECL:[0-9]+]] = {{.*}} ; [ DW_TAG_subprogram ] [line [[@LINE+1]]] [~OuterClass2]
  ~OuterClass2(); // line 10
};
OuterClass2::InnerClass2 OuterClass2::theInnerClass2;
// CHECK2: !"0x2e\00~OuterClass2\00{{.*}}\00[[@LINE+1]]"{{.*}}, ![[DECL]], {{.*}}} ; [ DW_TAG_subprogram ] [line [[@LINE+1]]] [def] [~OuterClass2]
OuterClass2::~OuterClass2() { }
