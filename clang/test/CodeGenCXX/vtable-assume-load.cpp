// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o %t.ll -O1 -disable-llvm-optzns -fms-extensions
// RUN: %clang_cc1 %s -triple i686-pc-win32 -emit-llvm -o %t.ms.ll -O1 -disable-llvm-optzns -fms-extensions

// RUN: FileCheck --check-prefix=CHECK1 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK2 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK3 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK4 --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=CHECK-MS --input-file=%t.ms.ll %s

namespace test1 {

struct A {
  A();
  virtual void foo();
};

struct B : A {
  virtual void foo();
};

void g(A *a) { a->foo(); }

// CHECK1-LABEL: define void @_ZN5test14fooAEv()
// CHECK1: call void @_ZN5test11AC1Ev(%"struct.test1::A"* %a)
// CHECK1: %vtable = load i8**, i8*** %1
// CHECK1: %cmp.vtables = icmp eq i8** %vtable, getelementptr inbounds ([3 x i8*], [3 x i8*]* @_ZTVN5test11AE, i64 0, i64 2)
// CHECK1: call void @llvm.assume(i1 %cmp.vtables)
// CHECK1-LABEL: }

void fooA() {
  A a;
  g(&a);
}

// CHECK1-LABEL: define void @_ZN5test14fooBEv()
// CHECK1: call void @_ZN5test11BC1Ev(%"struct.test1::B"* %b)
// CHECK1: %vtable = load i8**, i8*** %1
// CHECK1: %cmp.vtables = icmp eq i8** %vtable, getelementptr inbounds ([3 x i8*], [3 x i8*]* @_ZTVN5test11BE, i64 0, i64 2)
// CHECK1: call void @llvm.assume(i1 %cmp.vtables)
// CHECK1-LABEL: }

void fooB() {
  B b;
  g(&b);
}
// there should not be any assumes in the ctor that calls base ctor
// CHECK1-LABEL: define linkonce_odr void @_ZN5test11BC2Ev(%"struct.test1::B"* %this)
// CHECK1-NOT: @llvm.assume(
// CHECK1-LABEL: }
}
namespace test2 {
struct A {
  A();
  virtual void foo();
};

struct B {
  B();
  virtual void bar();
};

struct C : A, B {
  C();
  virtual void foo();
};
void g(A *a) { a->foo(); }
void h(B *b) { b->bar(); }

// CHECK2-LABEL: define void @_ZN5test24testEv()
// CHECK2: call void @_ZN5test21CC1Ev(%"struct.test2::C"* %c)
// CHECK2: vtable = load i8**, i8*** %1
// CHECK2: %cmp.vtables = icmp eq i8** %vtable, getelementptr inbounds ([6 x i8*], [6 x i8*]* @_ZTVN5test21CE, i64 0, i64 2)
// CHECK2: call void @llvm.assume(i1 %cmp.vtables)

// CHECK2: %2 = bitcast %"struct.test2::C"* %c to i8*
// CHECK2: %add.ptr = getelementptr inbounds i8, i8* %2, i64 8
// CHECK2: %3 = bitcast i8* %add.ptr to i8***
// CHECK2: %vtable1 = load i8**, i8*** %3
// CHECK2: %cmp.vtables2 = icmp eq i8** %vtable1, getelementptr inbounds ([6 x i8*], [6 x i8*]* @_ZTVN5test21CE, i64 0, i64 5)
// CHECK2: call void @llvm.assume(i1 %cmp.vtables2)

// CHECK2: call void @_ZN5test21gEPNS_1AE(
// CHECK2-LABEL: }

void test() {
  C c;
  g(&c);
  h(&c);
}
}

namespace test3 {
struct A {
  A();
};

struct B : A {
  B();
  virtual void foo();
};

struct C : virtual A, B {
  C();
  virtual void foo();
};
void g(B *a) { a->foo(); }

// CHECK3-LABEL: define void @_ZN5test34testEv()
// CHECK3: call void @_ZN5test31CC1Ev(%"struct.test3::C"* %c)
// CHECK3: %cmp.vtables = icmp eq i8** %vtable, getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTVN5test31CE, i64 0, i64 3)
// CHECK3: call void @llvm.assume(i1 %cmp.vtables)
// CHECK3-LABLEL: }
void test() {
  C c;
  g(&c);
}
} // test3

namespace test4 {
struct A {
  A();
  virtual void foo();
};

struct B : virtual A {
  B();
  virtual void foo();
};
struct C : B {
  C();
  virtual void foo();
};

void g(C *c) { c->foo(); }

// CHECK4-LABEL: define void @_ZN5test44testEv()
// CHECK4: call void @_ZN5test41CC1Ev(%"struct.test4::C"* %c)
// CHECK4: %vtable = load i8**, i8*** %1
// CHECK4: %cmp.vtables = icmp eq i8** %vtable, getelementptr inbounds ([5 x i8*], [5 x i8*]* @_ZTVN5test41CE, i64 0, i64 4)
// CHECK4: call void @llvm.assume(i1 %cmp.vtables)

// CHECK4: %2 = bitcast %"struct.test4::C"* %c to i8***
// CHECK4: %vtable1 = load i8**, i8*** %2
// CHECK4: %cmp.vtables2 = icmp eq i8** %vtable1, getelementptr inbounds ([5 x i8*], [5 x i8*]* @_ZTVN5test41CE, i64 0, i64 4)
// CHECK4: call void @llvm.assume(i1 %cmp.vtables2)
// CHECK4-LABEL: }

void test() {
  C c;
  g(&c);
}
} // test4

namespace test5 {

struct __declspec(novtable) S {
  virtual void foo();
};

void g(S &s) { s.foo(); }

// if struct has novtable specifier, then we can't generate assumes
// CHECK-MS-LABEL: define void @"\01?test@test5@@YAXXZ"()
// CHECK-MS: call x86_thiscallcc %"struct.test5::S"* @"\01??0S@test5@@QAE@XZ"(
// CHECK-MS-NOT: @llvm.assume
// CHECK-MS-LABEL: }

void test() {
  S s;
  g(s);
}

} // test5
