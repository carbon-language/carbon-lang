//RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-linux | FileCheck --check-prefix=CHECKGEN %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=thumbv7-apple-ios3.0 -target-abi apcs-gnu | FileCheck --check-prefix=CHECKARM %s
//RUN: %clang_cc1 %s -emit-llvm -o - -triple=i386-pc-win32 -cxx-abi microsoft -fno-rtti | FileCheck --check-prefix=CHECKMS %s
// FIXME: these tests crash on the bots when run with -triple=x86_64-pc-win32

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
B::~B() { }

// CHECKGEN-LABEL: define void @_ZN1BC1EPi(%class.B* %this, i32* %i)
// CHECKGEN-LABEL: define void @_ZN1BC2EPi(%class.B* %this, i32* %i)
// CHECKGEN-LABEL: define void @_ZN1BD1Ev(%class.B* %this)
// CHECKGEN-LABEL: define void @_ZN1BD2Ev(%class.B* %this)

// CHECKARM-LABEL: define %class.B* @_ZN1BC1EPi(%class.B* returned %this, i32* %i)
// CHECKARM-LABEL: define %class.B* @_ZN1BC2EPi(%class.B* returned %this, i32* %i)
// CHECKARM-LABEL: define %class.B* @_ZN1BD1Ev(%class.B* returned %this)
// CHECKARM-LABEL: define %class.B* @_ZN1BD2Ev(%class.B* returned %this)

// CHECKMS-LABEL: define x86_thiscallcc %class.B* @"\01??0B@@QAE@PAH@Z"(%class.B* returned %this, i32* %i)
// CHECKMS-LABEL: define x86_thiscallcc void @"\01??1B@@QAE@XZ"(%class.B* %this)

class C : public A, public B {
public:
  C(int *i, char *c);
  virtual ~C();
private:
  char *c_;
};

C::C(int *i, char *c) : B(i), c_(c) { }
C::~C() { }

// CHECKGEN-LABEL: define void @_ZN1CC1EPiPc(%class.C* %this, i32* %i, i8* %c)
// CHECKGEN-LABEL: define void @_ZN1CC2EPiPc(%class.C* %this, i32* %i, i8* %c)
// CHECKGEN-LABEL: define void @_ZN1CD0Ev(%class.C* %this)
// CHECKGEN-LABEL: define void @_ZN1CD1Ev(%class.C* %this)
// CHECKGEN-LABEL: define void @_ZN1CD2Ev(%class.C* %this)

// CHECKARM-LABEL: define %class.C* @_ZN1CC1EPiPc(%class.C* returned %this, i32* %i, i8* %c)
// CHECKARM-LABEL: define %class.C* @_ZN1CC2EPiPc(%class.C* returned %this, i32* %i, i8* %c)
// CHECKARM-LABEL: define void @_ZN1CD0Ev(%class.C* %this)
// CHECKARM-LABEL: define %class.C* @_ZN1CD1Ev(%class.C* returned %this)
// CHECKARM-LABEL: define %class.C* @_ZN1CD2Ev(%class.C* returned %this)

// CHECKMS-LABEL: define x86_thiscallcc %class.C* @"\01??0C@@QAE@PAHPAD@Z"(%class.C* returned %this, i32* %i, i8* %c)
// CHECKMS-LABEL: define x86_thiscallcc void @"\01??1C@@UAE@XZ"(%class.C* %this)

class D : public virtual A {
public:
  D();
  ~D();
};

D::D() { }
D::~D() { }

// CHECKGEN-LABEL: define void @_ZN1DC1Ev(%class.D* %this)
// CHECKGEN-LABEL: define void @_ZN1DC2Ev(%class.D* %this, i8** %vtt)
// CHECKGEN-LABEL: define void @_ZN1DD1Ev(%class.D* %this)
// CHECKGEN-LABEL: define void @_ZN1DD2Ev(%class.D* %this, i8** %vtt)

// CHECKARM-LABEL: define %class.D* @_ZN1DC1Ev(%class.D* returned %this)
// CHECKARM-LABEL: define %class.D* @_ZN1DC2Ev(%class.D* returned %this, i8** %vtt)
// CHECKARM-LABEL: define %class.D* @_ZN1DD1Ev(%class.D* returned %this)
// CHECKARM-LABEL: define %class.D* @_ZN1DD2Ev(%class.D* returned %this, i8** %vtt)

// CHECKMS-LABEL: define x86_thiscallcc %class.D* @"\01??0D@@QAE@XZ"(%class.D* returned %this, i32 %is_most_derived)
// CHECKMS-LABEL: define x86_thiscallcc void @"\01??1D@@QAE@XZ"(%class.D* %this)

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

// CHECKARM-LABEL: define void @_Z15test_destructorv()

// Verify that virtual calls to destructors are not marked with a 'returned'
// this parameter at the call site...
// CHECKARM: [[VFN:%.*]] = getelementptr inbounds %class.E* (%class.E*)**
// CHECKARM: [[THUNK:%.*]] = load %class.E* (%class.E*)** [[VFN]]
// CHECKARM: call %class.E* [[THUNK]](%class.E* %

// ...but static calls create declarations with 'returned' this
// CHECKARM: {{%.*}} = call %class.E* @_ZN1ED1Ev(%class.E* %

// CHECKARM: declare %class.E* @_ZN1ED1Ev(%class.E* returned)
