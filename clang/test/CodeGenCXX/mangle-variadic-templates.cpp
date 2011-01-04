// RUN: %clang_cc1 -std=c++0x -emit-llvm -o - %s | FileCheck %s

template<unsigned I, typename ...Types>
struct X { };

// CHECK: define weak_odr void @_Z2f0IJEEv1XIXsZT_EJspRT_EE
template<typename ...Types>
void f0(X<sizeof...(Types), Types&...>) { }

template void f0(X<0>);

// CHECK: define weak_odr void @_Z2f0IJifdEEv1XIXsZT_EJspRT_EE
template void f0<int, float, double>(X<3, int&, float&, double&>);
