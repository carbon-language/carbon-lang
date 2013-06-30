//RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-linux | FileCheck --check-prefix=CHECKGEN %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=thumbv7-apple-ios3.0 -target-abi apcs-gnu | FileCheck --check-prefix=CHECKARM %s
//RUN: %clang_cc1 %s -emit-llvm -o - -DPR12784_WORKAROUND -triple=x86_64-pc-win32 -cxx-abi microsoft | FileCheck --check-prefix=CHECKMS %s

// FIXME: Add checks to ensure that Microsoft destructors do not return 'this'
// once PR12784 is resolved

// Make sure we attach the 'returned' attribute to the 'this' parameter of
// constructors and destructors which return this (and only these cases)

class A {
public:
  A();
  ~A();

private:
  int x_;
};

class B : public A {
public:
  B(int *i);
  ~B();

private:
  int *i_;
};

B::B(int *i) : i_(i) { }
#ifndef PR12784_WORKAROUND
B::~B() { }
#endif

// CHECKGEN: define void @_ZN1BC1EPi(%class.B* %this, i32* %i)
// CHECKGEN: define void @_ZN1BC2EPi(%class.B* %this, i32* %i)
// CHECKGEN: define void @_ZN1BD1Ev(%class.B* %this)
// CHECKGEN: define void @_ZN1BD2Ev(%class.B* %this)

// CHECKARM: define %class.B* @_ZN1BC1EPi(%class.B* returned %this, i32* %i)
// CHECKARM: define %class.B* @_ZN1BC2EPi(%class.B* returned %this, i32* %i)
// CHECKARM: define %class.B* @_ZN1BD1Ev(%class.B* returned %this)
// CHECKARM: define %class.B* @_ZN1BD2Ev(%class.B* returned %this)

// CHECKMS: define %class.B* @"\01??0B@@QEAA@PEAH@Z"(%class.B* returned %this, i32* %i)

class C : public A, public B {
public:
  C(int *i, char *c);
  virtual ~C();
private:
  char *c_;
};

C::C(int *i, char *c) : B(i), c_(c) { }
#ifndef PR12784_WORKAROUND
C::~C() { }
#endif

// CHECKGEN: define void @_ZN1CC1EPiPc(%class.C* %this, i32* %i, i8* %c)
// CHECKGEN: define void @_ZN1CC2EPiPc(%class.C* %this, i32* %i, i8* %c)
// CHECKGEN: define void @_ZN1CD0Ev(%class.C* %this)
// CHECKGEN: define void @_ZN1CD1Ev(%class.C* %this)
// CHECKGEN: define void @_ZN1CD2Ev(%class.C* %this)

// CHECKARM: define %class.C* @_ZN1CC1EPiPc(%class.C* returned %this, i32* %i, i8* %c)
// CHECKARM: define %class.C* @_ZN1CC2EPiPc(%class.C* returned %this, i32* %i, i8* %c)
// CHECKARM: define void @_ZN1CD0Ev(%class.C* %this)
// CHECKARM: define %class.C* @_ZN1CD1Ev(%class.C* returned %this)
// CHECKARM: define %class.C* @_ZN1CD2Ev(%class.C* returned %this)

// CHECKMS: define %class.C* @"\01??0C@@QEAA@PEAHPEAD@Z"(%class.C* returned %this, i32* %i, i8* %c)

class D : public virtual A {
public:
  D();
  ~D();
};

#ifndef PR12784_WORKAROUND
D::D() { }
D::~D() { }
#endif

// CHECKGEN: define void @_ZN1DC1Ev(%class.D* %this)
// CHECKGEN: define void @_ZN1DC2Ev(%class.D* %this, i8** %vtt)
// CHECKGEN: define void @_ZN1DD1Ev(%class.D* %this)
// CHECKGEN: define void @_ZN1DD2Ev(%class.D* %this, i8** %vtt)

// CHECKARM: define %class.D* @_ZN1DC1Ev(%class.D* returned %this)
// CHECKARM: define %class.D* @_ZN1DC2Ev(%class.D* returned %this, i8** %vtt)
// CHECKARM: define %class.D* @_ZN1DD1Ev(%class.D* returned %this)
// CHECKARM: define %class.D* @_ZN1DD2Ev(%class.D* returned %this, i8** %vtt)

class E {
public:
  E();
  virtual ~E();
};

E* gete();

void test_destructor() {
  const E& e1 = E();
  E* e2 = gete();
  e2->~E();
}

// CHECKARM: define void @_Z15test_destructorv()

// Verify that virtual calls to destructors are not marked with a 'returned'
// this parameter at the call site...
// CHECKARM: [[VFN:%.*]] = getelementptr inbounds %class.E* (%class.E*)**
// CHECKARM: [[THUNK:%.*]] = load %class.E* (%class.E*)** [[VFN]]
// CHECKARM: call %class.E* [[THUNK]](%class.E* %

// ...but static calls create declarations with 'returned' this
// CHECKARM: {{%.*}} = call %class.E* @_ZN1ED1Ev(%class.E* %

// CHECKARM: declare %class.E* @_ZN1ED1Ev(%class.E* returned)
