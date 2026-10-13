// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/infinite_recursion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/infinite_recursion.carbon

import Cpp;

fn ManualMultiply[template T: type](x: T) -> T;

inline Cpp '''c++
template<typename T>
auto ManualMultiply(T x) -> T {
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+3]]:13: error: no matching function for call to 'ManualMultiply' [CppInteropParseError]
    // CHECK:STDERR:    23 |     return *Carbon::ManualMultiply(&x);
    // CHECK:STDERR:       |             ^~~~~~~~~~~~~~~~~~~~~~
    return *Carbon::ManualMultiply(&x);
  }
''';

fn ManualMultiply[template T: type](x: T) -> T {
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+115]]:32: note: in instantiation of function template specialization 'ManualMultiply<int ********>' requested here [CppInteropParseNote]
    // CHECK:STDERR:   143 |     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:       |                                ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-16]]:47: note: candidate template ignored: substitution failure [with T = int *********] [CppInteropParseNote]
    // CHECK:STDERR:    15 | fn ManualMultiply[template T: type](x: T) -> T;
    // CHECK:STDERR:       |                                               ^
    // CHECK:STDERR:
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-20]]:1: error: unable to monomorphize specific `ManualMultiply(Core.Optional(i32* as Core.OptionalStorage))` [ResolvingSpecificHere]
    // CHECK:STDERR: fn ManualMultiply[template T: type](x: T) -> T;
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+105]]:31: note: cannot implicitly convert expression of type `Core.Optional(i32* as Core.OptionalStorage)` to `i32*` [ConversionFailure]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+102]]:31: note: type `Core.Optional(i32* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(i32*)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-25]]:23: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: auto ManualMultiply(T x) -> T {
    // CHECK:STDERR:                       ^
    // CHECK:STDERR:
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-33]]:1: error: unable to monomorphize specific `ManualMultiply(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage))` [ResolvingSpecificHere]
    // CHECK:STDERR: fn ManualMultiply[template T: type](x: T) -> T;
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+92]]:31: note: cannot implicitly convert expression of type `Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)` to `Core.Optional(i32* as Core.OptionalStorage)*` [ConversionFailure]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+89]]:31: note: type `Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Core.Optional(i32* as Core.OptionalStorage)*)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-38]]:23: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: auto ManualMultiply(T x) -> T {
    // CHECK:STDERR:                       ^
    // CHECK:STDERR:
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-46]]:1: error: unable to monomorphize specific `ManualMultiply(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage))` [ResolvingSpecificHere]
    // CHECK:STDERR: fn ManualMultiply[template T: type](x: T) -> T;
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+79]]:31: note: cannot implicitly convert expression of type `Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` to `Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)*` [ConversionFailure]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+76]]:31: note: type `Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)*)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-51]]:23: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: auto ManualMultiply(T x) -> T {
    // CHECK:STDERR:                       ^
    // CHECK:STDERR:
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-59]]:1: error: unable to monomorphize specific `ManualMultiply(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage))` [ResolvingSpecificHere]
    // CHECK:STDERR: fn ManualMultiply[template T: type](x: T) -> T;
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+66]]:31: note: cannot implicitly convert expression of type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` to `Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*` [ConversionFailure]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+63]]:31: note: type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-64]]:23: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: auto ManualMultiply(T x) -> T {
    // CHECK:STDERR:                       ^
    // CHECK:STDERR:
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-72]]:1: error: unable to monomorphize specific `ManualMultiply(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage))` [ResolvingSpecificHere]
    // CHECK:STDERR: fn ManualMultiply[template T: type](x: T) -> T;
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+53]]:31: note: cannot implicitly convert expression of type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` to `Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*` [ConversionFailure]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+50]]:31: note: type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-77]]:23: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: auto ManualMultiply(T x) -> T {
    // CHECK:STDERR:                       ^
    // CHECK:STDERR:
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-85]]:1: error: unable to monomorphize specific `ManualMultiply(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage))` [ResolvingSpecificHere]
    // CHECK:STDERR: fn ManualMultiply[template T: type](x: T) -> T;
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+40]]:31: note: cannot implicitly convert expression of type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` to `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*` [ConversionFailure]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+37]]:31: note: type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-90]]:23: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: auto ManualMultiply(T x) -> T {
    // CHECK:STDERR:                       ^
    // CHECK:STDERR:
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-98]]:1: error: unable to monomorphize specific `ManualMultiply(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage))` [ResolvingSpecificHere]
    // CHECK:STDERR: fn ManualMultiply[template T: type](x: T) -> T;
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+27]]:31: note: cannot implicitly convert expression of type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` to `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*` [ConversionFailure]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+24]]:31: note: type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-103]]:23: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: auto ManualMultiply(T x) -> T {
    // CHECK:STDERR:                       ^
    // CHECK:STDERR:
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-111]]:1: error: unable to monomorphize specific `ManualMultiply(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage))` [ResolvingSpecificHere]
    // CHECK:STDERR: fn ManualMultiply[template T: type](x: T) -> T;
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+14]]:31: note: cannot implicitly convert expression of type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` to `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*` [ConversionFailure]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE+11]]:31: note: type `Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)` does not implement interface `Core.ImplicitAs(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(Core.Optional(i32* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)* as Core.OptionalStorage)*)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     return Cpp.ManualMultiply(x);
    // CHECK:STDERR:                               ^
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-116]]:23: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: auto ManualMultiply(T x) -> T {
    // CHECK:STDERR:                       ^
    // CHECK:STDERR:
    // CHECK:STDERR: infinite_recursion.carbon:[[@LINE-138]]:1: error: semantics TODO: `maximum depth for bidirectional Carbon/C++ interop is 10` [SemanticsTodo]
    // CHECK:STDERR: // Part of the Carbon Language project, under the Apache License v2.0 with LLVM
    // CHECK:STDERR: ^
    // CHECK:STDERR:
    return Cpp.ManualMultiply(x);
}

fn Test() {
  ManualMultiply(5 as i32);
}
