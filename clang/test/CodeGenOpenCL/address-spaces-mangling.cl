// RUN: %clang_cc1 %s -ffake-address-space-map -faddress-space-map-mangling=yes -emit-llvm -o - | FileCheck -check-prefix=ASMANG %s
// RUN: %clang_cc1 %s -ffake-address-space-map -faddress-space-map-mangling=no -emit-llvm -o - | FileCheck -check-prefix=NOASMANG %s

// We can't name this f as private is equivalent to default
// no specifier given address space so we get multiple definition
// warnings, but we do want it for comparison purposes.
__attribute__((overloadable))
void ff(int *arg) { }
// ASMANG: @_Z2ffPi
// NOASMANG: @_Z2ffPi

__attribute__((overloadable))
void f(private int *arg) { }
// ASMANG: @_Z1fPi
// NOASMANG: @_Z1fPi

__attribute__((overloadable))
void f(global int *arg) { }
// ASMANG: @_Z1fPU3AS1i
// NOASMANG: @_Z1fPU8CLglobali

__attribute__((overloadable))
void f(local int *arg) { }
// ASMANG: @_Z1fPU3AS2i
// NOASMANG: @_Z1fPU7CLlocali

__attribute__((overloadable))
void f(constant int *arg) { }
// ASMANG: @_Z1fPU3AS3i
// NOASMANG: @_Z1fPU10CLconstanti
