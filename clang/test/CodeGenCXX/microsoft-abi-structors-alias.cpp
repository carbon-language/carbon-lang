// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 -fno-rtti -mconstructor-aliases | FileCheck %s

namespace test1 {
template <typename T> class A {
  ~A() {}
};
template class A<char>;
// CHECK: define weak_odr x86_thiscallcc void @"\01??1?$A@D@test1@@AAE@XZ"
}
