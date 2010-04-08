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

// CHECK: @_ZTIv = weak_odr constant
// CHECK: @_ZTIPv = weak_odr constant
// CHECK: @_ZTIPKv = weak_odr constant
// CHECK: @_ZTIDi = weak_odr constant
// CHECK: @_ZTIPDi = weak_odr constant
// CHECK: @_ZTIPKDi = weak_odr constant
// CHECK: @_ZTIDs = weak_odr constant
// CHECK: @_ZTIPDs = weak_odr constant
// CHECK: @_ZTIPKDs = weak_odr constant
// CHECK: @_ZTIy = weak_odr constant
// CHECK: @_ZTIPy = weak_odr constant
// CHECK: @_ZTIPKy = weak_odr constant
// CHECK: @_ZTIx = weak_odr constant
// CHECK: @_ZTIPx = weak_odr constant
// CHECK: @_ZTIPKx = weak_odr constant
// CHECK: @_ZTIw = weak_odr constant
// CHECK: @_ZTIPw = weak_odr constant
// CHECK: @_ZTIPKw = weak_odr constant
// CHECK: @_ZTIt = weak_odr constant
// CHECK: @_ZTIPt = weak_odr constant
// CHECK: @_ZTIPKt = weak_odr constant
// CHECK: @_ZTIs = weak_odr constant
// CHECK: @_ZTIPs = weak_odr constant
// CHECK: @_ZTIPKs = weak_odr constant
// CHECK: @_ZTIm = weak_odr constant
// CHECK: @_ZTIPm = weak_odr constant
// CHECK: @_ZTIPKm = weak_odr constant
// CHECK: @_ZTIl = weak_odr constant
// CHECK: @_ZTIPl = weak_odr constant
// CHECK: @_ZTIPKl = weak_odr constant
// CHECK: @_ZTIj = weak_odr constant
// CHECK: @_ZTIPj = weak_odr constant
// CHECK: @_ZTIPKj = weak_odr constant
// CHECK: @_ZTIi = weak_odr constant
// CHECK: @_ZTIPi = weak_odr constant
// CHECK: @_ZTIPKi = weak_odr constant
// CHECK: @_ZTIh = weak_odr constant
// CHECK: @_ZTIPh = weak_odr constant
// CHECK: @_ZTIPKh = weak_odr constant
// CHECK: @_ZTIf = weak_odr constant
// CHECK: @_ZTIPf = weak_odr constant
// CHECK: @_ZTIPKf = weak_odr constant
// CHECK: @_ZTIe = weak_odr constant
// CHECK: @_ZTIPe = weak_odr constant
// CHECK: @_ZTIPKe = weak_odr constant
// CHECK: @_ZTId = weak_odr constant
// CHECK: @_ZTIPd = weak_odr constant
// CHECK: @_ZTIPKd = weak_odr constant
// CHECK: @_ZTIc = weak_odr constant
// CHECK: @_ZTIPc = weak_odr constant
// CHECK: @_ZTIPKc = weak_odr constant
// CHECK: @_ZTIb = weak_odr constant
// CHECK: @_ZTIPb = weak_odr constant
// CHECK: @_ZTIPKb = weak_odr constant
// CHECK: @_ZTIa = weak_odr constant
// CHECK: @_ZTIPa = weak_odr constant
// CHECK: @_ZTIPKa = weak_odr constant
