// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s

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
