// RUN: %clang_cc1 -triple i386-apple-darwin9 -emit-llvm -o - %s | FileCheck %s

// Non-trivial dtors, should both be passed indirectly.
struct S {
  ~S();
  short s;
};

// CHECK: define void @_Z1fv(%struct.S* sret %
S f() { return S(); }
// CHECK: define void @_Z1f1S(%struct.S*)
void f(S) { }

// Non-trivial dtors, should both be passed indirectly.
class C {
public:
  ~C();
  double c;
};

// CHECK: define void @_Z1gv(%class.C* sret %
C g() { return C(); }

// CHECK: define void @_Z1f1C(%class.C*) 
void f(C) { }




// PR7058 - Missing byval on MI thunk definition.

// CHECK: define void @_ZThn4_N18BasicAliasAnalysis13getModRefInfoE8CallSite
// ...
// CHECK: %struct.CallSite* byval %CS)
struct CallSite {
  unsigned Ptr;
  CallSite(unsigned XX) : Ptr(XX) {}
};

struct AliasAnalysis {
  virtual void xyz();
  virtual void getModRefInfo(CallSite CS) = 0;
};

struct ModulePass {
  virtual void xx();
};

struct BasicAliasAnalysis : public ModulePass, public AliasAnalysis {
  void getModRefInfo(CallSite CS);
};

void BasicAliasAnalysis::getModRefInfo(CallSite CS) {
}

// Check various single element struct type conditions.
//
// PR7098.

// CHECK: define i64 @_Z2f0v()
struct s0_0 { int x; };
struct s0_1 : s0_0 { int* y; };
s0_1 f0() { return s0_1(); }

// CHECK: define i32 @_Z2f1v()
struct s1_0 { int x; };
struct s1_1 : s1_0 { };
s1_1 f1() { return s1_1(); }

// CHECK: define double @_Z2f2v()
struct s2_0 { double x; };
struct s2_1 : s2_0 { };
s2_1 f2() { return s2_1(); }

// CHECK: define double @_Z2f3v()
struct s3_0 { };
struct s3_1 { double x; };
struct s3_2 : s3_0, s3_1 { };
s3_2 f3() { return s3_2(); }

// CHECK: define i64 @_Z2f4v()
struct s4_0 { float x; };
struct s4_1 { float x; };
struct s4_2 : s4_0, s4_1 { };
s4_2 f4() { return s4_2(); }
