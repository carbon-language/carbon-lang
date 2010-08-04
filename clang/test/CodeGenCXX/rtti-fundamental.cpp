// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

#include <typeinfo>

std::type_info foo() {
  return typeid(void);
}

namespace __cxxabiv1 {
  struct __fundamental_type_info {
    virtual ~__fundamental_type_info();
  };

  __fundamental_type_info::~__fundamental_type_info() { }
}

// CHECK: @_ZTIv = constant
// CHECK: @_ZTIPv = constant
// CHECK: @_ZTIPKv = constant
// CHECK: @_ZTIDi = constant
// CHECK: @_ZTIPDi = constant
// CHECK: @_ZTIPKDi = constant
// CHECK: @_ZTIDs = constant
// CHECK: @_ZTIPDs = constant
// CHECK: @_ZTIPKDs = constant
// CHECK: @_ZTIy = constant
// CHECK: @_ZTIPy = constant
// CHECK: @_ZTIPKy = constant
// CHECK: @_ZTIx = constant
// CHECK: @_ZTIPx = constant
// CHECK: @_ZTIPKx = constant
// CHECK: @_ZTIw = constant
// CHECK: @_ZTIPw = constant
// CHECK: @_ZTIPKw = constant
// CHECK: @_ZTIt = constant
// CHECK: @_ZTIPt = constant
// CHECK: @_ZTIPKt = constant
// CHECK: @_ZTIs = constant
// CHECK: @_ZTIPs = constant
// CHECK: @_ZTIPKs = constant
// CHECK: @_ZTIm = constant
// CHECK: @_ZTIPm = constant
// CHECK: @_ZTIPKm = constant
// CHECK: @_ZTIl = constant
// CHECK: @_ZTIPl = constant
// CHECK: @_ZTIPKl = constant
// CHECK: @_ZTIj = constant
// CHECK: @_ZTIPj = constant
// CHECK: @_ZTIPKj = constant
// CHECK: @_ZTIi = constant
// CHECK: @_ZTIPi = constant
// CHECK: @_ZTIPKi = constant
// CHECK: @_ZTIh = constant
// CHECK: @_ZTIPh = constant
// CHECK: @_ZTIPKh = constant
// CHECK: @_ZTIf = constant
// CHECK: @_ZTIPf = constant
// CHECK: @_ZTIPKf = constant
// CHECK: @_ZTIe = constant
// CHECK: @_ZTIPe = constant
// CHECK: @_ZTIPKe = constant
// CHECK: @_ZTId = constant
// CHECK: @_ZTIPd = constant
// CHECK: @_ZTIPKd = constant
// CHECK: @_ZTIc = constant
// CHECK: @_ZTIPc = constant
// CHECK: @_ZTIPKc = constant
// CHECK: @_ZTIb = constant
// CHECK: @_ZTIPb = constant
// CHECK: @_ZTIPKb = constant
// CHECK: @_ZTIa = constant
// CHECK: @_ZTIPa = constant
// CHECK: @_ZTIPKa = constant
