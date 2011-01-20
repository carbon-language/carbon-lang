// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -emit-llvm -o - %s | FileCheck %s

struct Base { 
  virtual void abc1(void) const; 
  virtual void abc2(void) const; 
  virtual void abc(void) const; 
};

void Base::abc(void) const {}

void FUNC(Base* p) {
  p->Base::abc();
}

// CHECK-NOT: call void @_ZNK4Base3abcEv
