// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_impl_bad_assoc_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_impl_bad_assoc_fn.carbon

interface I { fn F(); }

class NoF {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:3: error: missing implementation of F in impl of interface I [ImplMissingFunction]
  // CHECK:STDERR:   impl as I {}
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-6]]:15: note: associated function F declared here [AssociatedFunctionHere]
  // CHECK:STDERR: interface I { fn F(); }
  // CHECK:STDERR:               ^~~~~~~
  // CHECK:STDERR:
  impl as I {}
}

class FNotFunction {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: associated function F implemented by non-function [ImplFunctionWithNonFunction]
    // CHECK:STDERR:     class F;
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-18]]:15: note: associated function F declared here [AssociatedFunctionHere]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    class F;
  }
}

fn PossiblyF();

// TODO: Should this be permitted?
class FAlias {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:11: error: associated function F implemented by non-function [ImplFunctionWithNonFunction]
    // CHECK:STDERR:     alias F = PossiblyF;
    // CHECK:STDERR:           ^
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-34]]:15: note: associated function F declared here [AssociatedFunctionHere]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    alias F = PossiblyF;
  }
}

class FExtraParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: error: 0 arguments passed to function expecting 1 argument [CallArgCountMismatch]
    // CHECK:STDERR:     fn F(b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: calling function declared here [InCallToEntity]
    // CHECK:STDERR:     fn F(b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-50]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F(b: bool);
  }
}

class FExtraImplicitParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: error: missing object argument in method call [MissingObjectInMethodCall]
    // CHECK:STDERR:     fn F[self: Self]();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: calling function declared here [InCallToFunction]
    // CHECK:STDERR:     fn F[self: Self]();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-66]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F[self: Self]();
  }
}

// TODO: Should this be permitted?
class FExtraReturnType {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: function redeclaration differs because return type is `bool` [FunctionRedeclReturnTypeDiffers]
    // CHECK:STDERR:     fn F() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-80]]:15: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F() -> bool;
  }
}

interface J { fn F[self: bool](b: bool) -> bool; }

class FMissingParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
    // CHECK:STDERR:     fn F[self: bool]() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: calling function declared here [InCallToEntity]
    // CHECK:STDERR:     fn F[self: bool]() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-10]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool]() -> bool;
  }
}

class FMissingImplicitParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: member name of type `<type of F>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
    // CHECK:STDERR:     fn F(b: bool) -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-23]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(b: bool) -> bool;
  }
}

class FMissingReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+16]]:5: error: cannot implicitly convert expression of type `()` to `bool` [ConversionFailure]
    // CHECK:STDERR:     fn F[self: bool](b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+13]]:5: note: type `()` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     fn F[self: bool](b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-39]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-43]]:32: error: cannot implicitly convert expression of type `bool` to `FDifferentParamType` [ConversionFailure]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-46]]:32: note: type `bool` does not implement interface `Core.ImplicitAs(FDifferentParamType)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^~~~~~~
    fn F[self: bool](b: bool);
  }
}

class FDifferentParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+13]]:22: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F[self: bool](b: Self) -> bool;
    // CHECK:STDERR:                      ^~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-58]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-62]]:20: error: cannot implicitly convert expression of type `bool` to `FDifferentImplicitParamType` [ConversionFailure]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                    ^~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-65]]:20: note: type `bool` does not implement interface `Core.ImplicitAs(FDifferentImplicitParamType)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                    ^~~~~~~~~~
    fn F[self: bool](b: Self) -> bool;
  }
}

class FDifferentImplicitParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F[self: Self](b: bool) -> bool;
    // CHECK:STDERR:          ^~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-77]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: Self](b: bool) -> bool;
  }
}

class FDifferentReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: error: cannot implicitly convert expression of type `FDifferentReturnType` to `bool` [ConversionFailure]
    // CHECK:STDERR:     fn F[self: bool](b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: type `FDifferentReturnType` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     fn F[self: bool](b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-93]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](b: bool) -> Self;
  }
}

interface SelfNested {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:8: error: cannot implicitly convert expression of type `SelfNestedBadParam` to `i32` [ConversionFailure]
  // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+3]]:8: note: type `SelfNestedBadParam` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
}

class SelfNestedBadParam {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-8]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
  }
}

class SelfNestedBadReturnType {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: error: cannot implicitly convert expression of type `array(SelfNestedBadParam, 4)` to `array(SelfNestedBadReturnType, 4)` [ConversionFailure]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: type `array(SelfNestedBadParam, 4)` does not implement interface `Core.ImplicitAs(array(SelfNestedBadReturnType, 4))` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-24]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> array(SelfNestedBadParam, 4);
  }
}

// CHECK:STDOUT: --- fail_impl_bad_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.ab9: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.08c: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.ab9) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.705: %I.WithSelf.F.type.08c = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.18e: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [concrete]
// CHECK:STDOUT:   %I.impl_witness.187: <witness> = impl_witness @NoF.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.fc8: %I.type = facet_value %NoF, (%I.impl_witness.187) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.cec: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.fc8) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.733: %I.WithSelf.F.type.cec = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [concrete]
// CHECK:STDOUT:   %I.impl_witness.cf4: <witness> = impl_witness @FNotFunction.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %F: type = class_type @F [concrete]
// CHECK:STDOUT:   %I.facet.c1a: %I.type = facet_value %FNotFunction, (%I.impl_witness.cf4) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.1b8: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.c1a) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.7e3: %I.WithSelf.F.type.1b8 = struct_value () [concrete]
// CHECK:STDOUT:   %PossiblyF.type: type = fn_type @PossiblyF [concrete]
// CHECK:STDOUT:   %PossiblyF: %PossiblyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [concrete]
// CHECK:STDOUT:   %I.impl_witness.109: <witness> = impl_witness @FAlias.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.574: %I.type = facet_value %FAlias, (%I.impl_witness.109) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.f26: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.574) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.5e6: %I.WithSelf.F.type.f26 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.627: <witness> = impl_witness @FExtraParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.531438.1: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.1 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.33e1fe.1: %FExtraParam.as.I.impl.F.type.531438.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.b53: %I.type = facet_value %FExtraParam, (%I.impl_witness.627) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.99d: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.b53) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.eab: %I.WithSelf.F.type.99d = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.531438.2: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.2 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.33e1fe.2: %FExtraParam.as.I.impl.F.type.531438.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.e9e: <witness> = impl_witness @FExtraImplicitParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.235: type = pattern_type %FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.9dd145.1: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_23.1 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.99eae8.1: %FExtraImplicitParam.as.I.impl.F.type.9dd145.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.5a5: %I.type = facet_value %FExtraImplicitParam, (%I.impl_witness.e9e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.e46: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.5a5) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.5bc: %I.WithSelf.F.type.e46 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.9dd145.2: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_23.2 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.99eae8.2: %FExtraImplicitParam.as.I.impl.F.type.9dd145.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [concrete]
// CHECK:STDOUT:   %I.impl_witness.247: <witness> = impl_witness @FExtraReturnType.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.type: type = fn_type @FExtraReturnType.as.I.impl.F [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F: %FExtraReturnType.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.1ed: %I.type = facet_value %FExtraReturnType, (%I.impl_witness.247) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.cd5: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.1ed) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.aee: %I.WithSelf.F.type.cd5 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.8a1: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.adb: type = fn_type @J.WithSelf.F, @J.WithSelf(%Self.8a1) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.569: %J.WithSelf.F.type.adb = struct_value () [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.481: %J.assoc_type = assoc_entity element0, @J.WithSelf.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.5e2: <witness> = impl_witness @FMissingParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.0f71d8.1: type = fn_type @FMissingParam.as.J.impl.F.loc117_31.1 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.7540f6.1: %FMissingParam.as.J.impl.F.type.0f71d8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.567: %J.type = facet_value %FMissingParam, (%J.impl_witness.5e2) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.54f: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.567) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.7e8: %J.WithSelf.F.type.54f = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.0f71d8.2: type = fn_type @FMissingParam.as.J.impl.F.loc117_31.2 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.7540f6.2: %FMissingParam.as.J.impl.F.type.0f71d8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.1e4: <witness> = impl_witness @FMissingImplicitParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.80a56b.1: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.c6a0b7.1: %FMissingImplicitParam.as.J.impl.F.type.80a56b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.266: %J.type = facet_value %FMissingImplicitParam, (%J.impl_witness.1e4) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.56e: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.266) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.d17: %J.WithSelf.F.type.56e = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.80a56b.2: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.c6a0b7.2: %FMissingImplicitParam.as.J.impl.F.type.80a56b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.7c7: <witness> = impl_witness @FMissingReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.ce0d21.1: type = fn_type @FMissingReturnType.as.J.impl.F.loc152_30.1 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.5b613a.1: %FMissingReturnType.as.J.impl.F.type.ce0d21.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.a9e: %J.type = facet_value %FMissingReturnType, (%J.impl_witness.7c7) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.8be: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.a9e) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.2a5: %J.WithSelf.F.type.8be = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.ce0d21.2: type = fn_type @FMissingReturnType.as.J.impl.F.loc152_30.2 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.5b613a.2: %FMissingReturnType.as.J.impl.F.type.ce0d21.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.cd7: <witness> = impl_witness @FDifferentParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.b79: type = pattern_type %FDifferentParamType [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.10926d.1: type = fn_type @FDifferentParamType.as.J.impl.F.loc171_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.f1245a.1: %FDifferentParamType.as.J.impl.F.type.10926d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.bd2: %J.type = facet_value %FDifferentParamType, (%J.impl_witness.cd7) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.06a: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.bd2) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.cad: %J.WithSelf.F.type.06a = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.10926d.2: type = fn_type @FDifferentParamType.as.J.impl.F.loc171_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.f1245a.2: %FDifferentParamType.as.J.impl.F.type.10926d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.4d2: <witness> = impl_witness @FDifferentImplicitParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.b8f: type = pattern_type %FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.955391.1: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.a9ece4.1: %FDifferentImplicitParamType.as.J.impl.F.type.955391.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.030: %J.type = facet_value %FDifferentImplicitParamType, (%J.impl_witness.4d2) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.f29: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.030) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.5b7: %J.WithSelf.F.type.f29 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.955391.2: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.a9ece4.2: %FDifferentImplicitParamType.as.J.impl.F.type.955391.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.a49: <witness> = impl_witness @FDifferentReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %.2a9: Core.Form = init_form %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.5c8: type = pattern_type %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.7bcb67.1: type = fn_type @FDifferentReturnType.as.J.impl.F.loc200_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.a709b1.1: %FDifferentReturnType.as.J.impl.F.type.7bcb67.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.4a1: %J.type = facet_value %FDifferentReturnType, (%J.impl_witness.a49) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.8e3: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.4a1) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.182: %J.WithSelf.F.type.8e3 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.7bcb67.2: type = fn_type @FDifferentReturnType.as.J.impl.F.loc200_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.a709b1.2: %FDifferentReturnType.as.J.impl.F.type.7bcb67.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.aed: %SelfNested.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b90: type = facet_access_type %Self.aed [symbolic]
// CHECK:STDOUT:   %ptr.ad7: type = ptr_type %Self.as_type.b90 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.8bf: type = struct_type {.x: %Self.as_type.b90, .y: %i32} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.f6b: %tuple.type.24b = tuple_value (%ptr.ad7, %struct_type.x.y.8bf) [symbolic]
// CHECK:STDOUT:   %tuple.type.873: type = tuple_type (%ptr.ad7, %struct_type.x.y.8bf) [symbolic]
// CHECK:STDOUT:   %pattern_type.5a6: type = pattern_type %tuple.type.873 [symbolic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.a6c: type = array_type %int_4, %Self.as_type.b90 [symbolic]
// CHECK:STDOUT:   %.725: Core.Form = init_form %array_type.a6c [symbolic]
// CHECK:STDOUT:   %pattern_type.28e: type = pattern_type %array_type.a6c [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.430: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%Self.aed) [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.257: %SelfNested.WithSelf.F.type.430 = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.da4: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.7e8: <witness> = impl_witness @SelfNestedBadParam.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.cf2: type = ptr_type %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %struct_type.x.y.871: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.bb3: %tuple.type.24b = tuple_value (%ptr.cf2, %struct_type.x.y.871) [concrete]
// CHECK:STDOUT:   %tuple.type.70f: type = tuple_type (%ptr.cf2, %struct_type.x.y.871) [concrete]
// CHECK:STDOUT:   %pattern_type.395: type = pattern_type %tuple.type.70f [concrete]
// CHECK:STDOUT:   %array_type.388: type = array_type %int_4, %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %.0cb: Core.Form = init_form %array_type.388 [concrete]
// CHECK:STDOUT:   %pattern_type.ec1: type = pattern_type %array_type.388 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.789632.1: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.6e1661.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.789632.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.b9f: %SelfNested.type = facet_value %SelfNestedBadParam, (%SelfNested.impl_witness.7e8) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.4de: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.b9f) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.5e9: %SelfNested.WithSelf.F.type.4de = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.62a: type = struct_type {.x: %SelfNestedBadParam, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.73b: %tuple.type.24b = tuple_value (%ptr.cf2, %struct_type.x.y.62a) [concrete]
// CHECK:STDOUT:   %tuple.type.7ea: type = tuple_type (%ptr.cf2, %struct_type.x.y.62a) [concrete]
// CHECK:STDOUT:   %pattern_type.8d7: type = pattern_type %tuple.type.7ea [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.789632.2: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.6e1661.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.789632.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.d9c: <witness> = impl_witness @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.cbe: type = ptr_type %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %struct_type.x.y.cfa: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.c42: %tuple.type.24b = tuple_value (%ptr.cbe, %struct_type.x.y.cfa) [concrete]
// CHECK:STDOUT:   %tuple.type.064: type = tuple_type (%ptr.cbe, %struct_type.x.y.cfa) [concrete]
// CHECK:STDOUT:   %pattern_type.1f1: type = pattern_type %tuple.type.064 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.75d128.1: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.587f77.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.75d128.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.23f: %SelfNested.type = facet_value %SelfNestedBadReturnType, (%SelfNested.impl_witness.d9c) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.156: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.23f) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.edc: %SelfNested.WithSelf.F.type.156 = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.31b: type = array_type %int_4, %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %.369: Core.Form = init_form %array_type.31b [concrete]
// CHECK:STDOUT:   %pattern_type.1fd: type = pattern_type %array_type.31b [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.75d128.2: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.587f77.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.75d128.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .NoF = %NoF.decl
// CHECK:STDOUT:     .FNotFunction = %FNotFunction.decl
// CHECK:STDOUT:     .PossiblyF = %PossiblyF.decl
// CHECK:STDOUT:     .FAlias = %FAlias.decl
// CHECK:STDOUT:     .FExtraParam = %FExtraParam.decl
// CHECK:STDOUT:     .FExtraImplicitParam = %FExtraImplicitParam.decl
// CHECK:STDOUT:     .FExtraReturnType = %FExtraReturnType.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .FMissingParam = %FMissingParam.decl
// CHECK:STDOUT:     .FMissingImplicitParam = %FMissingImplicitParam.decl
// CHECK:STDOUT:     .FMissingReturnType = %FMissingReturnType.decl
// CHECK:STDOUT:     .FDifferentParamType = %FDifferentParamType.decl
// CHECK:STDOUT:     .FDifferentImplicitParamType = %FDifferentImplicitParamType.decl
// CHECK:STDOUT:     .FDifferentReturnType = %FDifferentReturnType.decl
// CHECK:STDOUT:     .SelfNested = %SelfNested.decl
// CHECK:STDOUT:     .SelfNestedBadParam = %SelfNestedBadParam.decl
// CHECK:STDOUT:     .SelfNestedBadReturnType = %SelfNestedBadReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %NoF.decl: type = class_decl @NoF [concrete = constants.%NoF] {} {}
// CHECK:STDOUT:   %FNotFunction.decl: type = class_decl @FNotFunction [concrete = constants.%FNotFunction] {} {}
// CHECK:STDOUT:   %PossiblyF.decl: %PossiblyF.type = fn_decl @PossiblyF [concrete = constants.%PossiblyF] {} {}
// CHECK:STDOUT:   %FAlias.decl: type = class_decl @FAlias [concrete = constants.%FAlias] {} {}
// CHECK:STDOUT:   %FExtraParam.decl: type = class_decl @FExtraParam [concrete = constants.%FExtraParam] {} {}
// CHECK:STDOUT:   %FExtraImplicitParam.decl: type = class_decl @FExtraImplicitParam [concrete = constants.%FExtraImplicitParam] {} {}
// CHECK:STDOUT:   %FExtraReturnType.decl: type = class_decl @FExtraReturnType [concrete = constants.%FExtraReturnType] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %FMissingParam.decl: type = class_decl @FMissingParam [concrete = constants.%FMissingParam] {} {}
// CHECK:STDOUT:   %FMissingImplicitParam.decl: type = class_decl @FMissingImplicitParam [concrete = constants.%FMissingImplicitParam] {} {}
// CHECK:STDOUT:   %FMissingReturnType.decl: type = class_decl @FMissingReturnType [concrete = constants.%FMissingReturnType] {} {}
// CHECK:STDOUT:   %FDifferentParamType.decl: type = class_decl @FDifferentParamType [concrete = constants.%FDifferentParamType] {} {}
// CHECK:STDOUT:   %FDifferentImplicitParamType.decl: type = class_decl @FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType] {} {}
// CHECK:STDOUT:   %FDifferentReturnType.decl: type = class_decl @FDifferentReturnType [concrete = constants.%FDifferentReturnType] {} {}
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [concrete = constants.%SelfNested.type] {} {}
// CHECK:STDOUT:   %SelfNestedBadParam.decl: type = class_decl @SelfNestedBadParam [concrete = constants.%SelfNestedBadParam] {} {}
// CHECK:STDOUT:   %SelfNestedBadReturnType.decl: type = class_decl @SelfNestedBadReturnType [concrete = constants.%SelfNestedBadReturnType] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab9]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.08c) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.705)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.18e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.8a1]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.WithSelf.%J.WithSelf.F.type (%J.WithSelf.F.type.adb) = fn_decl @J.WithSelf.F [symbolic = @J.WithSelf.%J.WithSelf.F (constants.%J.WithSelf.F.569)] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc103_44.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc103_44.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc103_44.2: Core.Form = init_form %.loc103_44.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc103_26: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc103_35: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.WithSelf.F.decl [concrete = constants.%assoc0.481]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @J.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@J.WithSelf.%J.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = constants.%Self.aed]
// CHECK:STDOUT:   %SelfNested.WithSelf.decl = interface_with_self_decl @SelfNested [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %SelfNested.WithSelf.F.decl: @SelfNested.WithSelf.%SelfNested.WithSelf.F.type (%SelfNested.WithSelf.F.type.430) = fn_decl @SelfNested.WithSelf.F [symbolic = @SelfNested.WithSelf.%SelfNested.WithSelf.F (constants.%SelfNested.WithSelf.F.257)] {
// CHECK:STDOUT:     %x.param_patt: @SelfNested.WithSelf.F.%pattern_type.loc211_9 (%pattern_type.5a6) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @SelfNested.WithSelf.F.%pattern_type.loc211_9 (%pattern_type.5a6) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @SelfNested.WithSelf.F.%pattern_type.loc211_57 (%pattern_type.28e) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @SelfNested.WithSelf.F.%pattern_type.loc211_57 (%pattern_type.28e) = return_slot_pattern %return.param_patt, %array_type.loc211_57.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc211_50: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.aed)]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Self.as_type.loc211_50: type = facet_access_type %Self.ref.loc211_50 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b90)]
// CHECK:STDOUT:     %.loc211_50: type = converted %Self.ref.loc211_50, %Self.as_type.loc211_50 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b90)]
// CHECK:STDOUT:     %array_type.loc211_57.2: type = array_type %int_4, %.loc211_50 [symbolic = %array_type.loc211_57.1 (constants.%array_type.a6c)]
// CHECK:STDOUT:     %.loc211_57.2: Core.Form = init_form %array_type.loc211_57.2 [symbolic = %.loc211_57.1 (constants.%.725)]
// CHECK:STDOUT:     %x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.873) = value_param call_param0
// CHECK:STDOUT:     %.loc211_38.1: type = splice_block %.loc211_38.3 [symbolic = %tuple.type (constants.%tuple.type.873)] {
// CHECK:STDOUT:       %Self.ref.loc211_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.aed)]
// CHECK:STDOUT:       %Self.as_type.loc211_16.2: type = facet_access_type %Self.ref.loc211_12 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b90)]
// CHECK:STDOUT:       %.loc211_16: type = converted %Self.ref.loc211_12, %Self.as_type.loc211_16.2 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b90)]
// CHECK:STDOUT:       %ptr.loc211_16.2: type = ptr_type %.loc211_16 [symbolic = %ptr.loc211_16.1 (constants.%ptr.ad7)]
// CHECK:STDOUT:       %Self.ref.loc211_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.aed)]
// CHECK:STDOUT:       %Self.as_type.loc211_24: type = facet_access_type %Self.ref.loc211_24 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b90)]
// CHECK:STDOUT:       %.loc211_24: type = converted %Self.ref.loc211_24, %Self.as_type.loc211_24 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b90)]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y.loc211_37.2: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc211_16.1 (%Self.as_type.b90), .y: %i32} [symbolic = %struct_type.x.y.loc211_37.1 (constants.%struct_type.x.y.8bf)]
// CHECK:STDOUT:       %.loc211_38.2: %tuple.type.24b = tuple_literal (%ptr.loc211_16.2, %struct_type.x.y.loc211_37.2) [symbolic = %tuple (constants.%tuple.f6b)]
// CHECK:STDOUT:       %.loc211_38.3: type = converted %.loc211_38.2, constants.%tuple.type.873 [symbolic = %tuple.type (constants.%tuple.type.873)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.873) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @SelfNested.WithSelf.F.%array_type.loc211_57.1 (%array_type.a6c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @SelfNested.WithSelf.F.%array_type.loc211_57.1 (%array_type.a6c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %SelfNested.WithSelf.F.decl [concrete = constants.%assoc0.da4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @SelfNested.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@SelfNested.WithSelf.%SelfNested.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @NoF.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @NoF.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.187]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FNotFunction.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: type = class_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FNotFunction.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.cf4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FAlias.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %PossiblyF.ref: %PossiblyF.type = name_ref PossiblyF, file.%PossiblyF.decl [concrete = constants.%PossiblyF]
// CHECK:STDOUT:   %F: %PossiblyF.type = alias_binding F, file.%PossiblyF.decl [concrete = constants.%PossiblyF]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FAlias.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.109]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .PossiblyF = <poisoned>
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.1: %FExtraParam.as.I.impl.F.type.531438.1 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.33e1fe.1] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc69: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.2: %FExtraParam.as.I.impl.F.type.531438.2 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.2 [concrete = constants.%FExtraParam.as.I.impl.F.33e1fe.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraParam.as.I.impl.F.decl.loc69_18.2), @FExtraParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.627]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraParam.as.I.impl.F.decl.loc69_18.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraImplicitParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1: %FExtraImplicitParam.as.I.impl.F.type.9dd145.1 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_23.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.99eae8.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.235 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.235 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self: %FExtraImplicitParam = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.2: %FExtraImplicitParam.as.I.impl.F.type.9dd145.2 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_23.2 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.99eae8.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.2), @FExtraImplicitParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.e9e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraReturnType.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.decl: %FExtraReturnType.as.I.impl.F.type = fn_decl @FExtraReturnType.as.I.impl.F [concrete = constants.%FExtraReturnType.as.I.impl.F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc99_15.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc99_15.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc99_15.2: Core.Form = init_form %.loc99_15.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FExtraReturnType.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.247]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraReturnType.as.I.impl.F.decl
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_31.1: %FMissingParam.as.J.impl.F.type.0f71d8.1 = fn_decl @FMissingParam.as.J.impl.F.loc117_31.1 [concrete = constants.%FMissingParam.as.J.impl.F.7540f6.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc117_27.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc117_27.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc117_27.2: Core.Form = init_form %.loc117_27.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc117_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.567) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.567) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_31.2: %FMissingParam.as.J.impl.F.type.0f71d8.2 = fn_decl @FMissingParam.as.J.impl.F.loc117_31.2 [concrete = constants.%FMissingParam.as.J.impl.F.7540f6.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FMissingParam.as.J.impl.F.decl.loc117_31.2), @FMissingParam.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.5e2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingParam.as.J.impl.F.decl.loc117_31.1
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingImplicitParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1: %FMissingImplicitParam.as.J.impl.F.type.80a56b.1 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.c6a0b7.1] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc130_22.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc130_22.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc130_22.2: Core.Form = init_form %.loc130_22.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc130_13: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.266) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.266) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.2: %FMissingImplicitParam.as.J.impl.F.type.80a56b.2 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.c6a0b7.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.2), @FMissingImplicitParam.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.1e4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc152_30.1: %FMissingReturnType.as.J.impl.F.type.ce0d21.1 = fn_decl @FMissingReturnType.as.J.impl.F.loc152_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.5b613a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc152_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc152_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.a9e) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.a9e) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc152_30.2: %FMissingReturnType.as.J.impl.F.type.ce0d21.2 = fn_decl @FMissingReturnType.as.J.impl.F.loc152_30.2 [concrete = constants.%FMissingReturnType.as.J.impl.F.5b613a.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FMissingReturnType.as.J.impl.F.decl.loc152_30.2), @FMissingReturnType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.7c7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingReturnType.as.J.impl.F.decl.loc152_30.1
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc171_38.1: %FDifferentParamType.as.J.impl.F.type.10926d.1 = fn_decl @FDifferentParamType.as.J.impl.F.loc171_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.f1245a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.b79 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.b79 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc171_34.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc171_34.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc171_34.2: Core.Form = init_form %.loc171_34.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc171_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: %FDifferentParamType = value_param call_param1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [concrete = constants.%FDifferentParamType]
// CHECK:STDOUT:     %b: %FDifferentParamType = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.bd2) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.bd2) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc171_38.2: %FDifferentParamType.as.J.impl.F.type.10926d.2 = fn_decl @FDifferentParamType.as.J.impl.F.loc171_38.2 [concrete = constants.%FDifferentParamType.as.J.impl.F.f1245a.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FDifferentParamType.as.J.impl.F.decl.loc171_38.2), @FDifferentParamType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.cd7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentParamType.as.J.impl.F.decl.loc171_38.1
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentImplicitParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1: %FDifferentImplicitParamType.as.J.impl.F.type.955391.1 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.a9ece4.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b8f = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.b8f = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc184_34.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc184_34.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc184_34.2: Core.Form = init_form %.loc184_34.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: %FDifferentImplicitParamType = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self: %FDifferentImplicitParamType = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc184_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.030) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.030) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.2: %FDifferentImplicitParamType.as.J.impl.F.type.955391.2 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.a9ece4.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.2), @FDifferentImplicitParamType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.4d2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc200_38.1: %FDifferentReturnType.as.J.impl.F.type.7bcb67.1 = fn_decl @FDifferentReturnType.as.J.impl.F.loc200_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.a709b1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5c8 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5c8 = return_slot_pattern %return.param_patt, %Self.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %.loc200_34: Core.Form = init_form %Self.ref [concrete = constants.%.2a9]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc200_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc200_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param call_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.4a1) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.4a1) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc200_38.2: %FDifferentReturnType.as.J.impl.F.type.7bcb67.2 = fn_decl @FDifferentReturnType.as.J.impl.F.loc200_38.2 [concrete = constants.%FDifferentReturnType.as.J.impl.F.a709b1.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FDifferentReturnType.as.J.impl.F.decl.loc200_38.2), @FDifferentReturnType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.a49]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentReturnType.as.J.impl.F.decl.loc200_38.1
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadParam.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.789632.1 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.6e1661.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.395 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.395 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ec1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.ec1 = return_slot_pattern %return.param_patt, %array_type [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc223_65: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref.loc223_65 [concrete = constants.%array_type.388]
// CHECK:STDOUT:     %.loc223_86: Core.Form = init_form %array_type [concrete = constants.%.0cb]
// CHECK:STDOUT:     %x.param: %tuple.type.70f = value_param call_param0
// CHECK:STDOUT:     %.loc223_53.1: type = splice_block %.loc223_53.3 [concrete = constants.%tuple.type.70f] {
// CHECK:STDOUT:       %SelfNestedBadParam.ref.loc223_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadParam.ref.loc223_14 [concrete = constants.%ptr.cf2]
// CHECK:STDOUT:       %i32.loc223_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc223_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete = constants.%struct_type.x.y.871]
// CHECK:STDOUT:       %.loc223_53.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.bb3]
// CHECK:STDOUT:       %.loc223_53.3: type = converted %.loc223_53.2, constants.%tuple.type.70f [concrete = constants.%tuple.type.70f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.70f = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.388 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.388 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc211_57.1: type = specific_constant @SelfNested.WithSelf.F.%array_type.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.b9f) [concrete = constants.%array_type.388]
// CHECK:STDOUT:   %.loc211_57.2: type = specific_constant @SelfNested.WithSelf.F.%.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.b9f) [concrete = constants.%.0cb]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.789632.2 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.6e1661.2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.8d7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.8d7 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ec1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.ec1 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.7ea = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.7ea = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.388 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.388 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.2), @SelfNestedBadParam.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.7e8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadReturnType.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.75d128.1 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.587f77.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.1f1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.1f1 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ec1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.ec1 = return_slot_pattern %return.param_patt, %array_type [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref [concrete = constants.%array_type.388]
// CHECK:STDOUT:     %.loc239_111: Core.Form = init_form %array_type [concrete = constants.%.0cb]
// CHECK:STDOUT:     %x.param: %tuple.type.064 = value_param call_param0
// CHECK:STDOUT:     %.loc239_78.1: type = splice_block %.loc239_78.3 [concrete = constants.%tuple.type.064] {
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc239_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadReturnType.ref.loc239_14 [concrete = constants.%ptr.cbe]
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc239_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete = constants.%struct_type.x.y.cfa]
// CHECK:STDOUT:       %.loc239_78.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.c42]
// CHECK:STDOUT:       %.loc239_78.3: type = converted %.loc239_78.2, constants.%tuple.type.064 [concrete = constants.%tuple.type.064]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.064 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.388 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.388 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc211_57.1: type = specific_constant @SelfNested.WithSelf.F.%array_type.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.23f) [concrete = constants.%array_type.31b]
// CHECK:STDOUT:   %.loc211_57.2: type = specific_constant @SelfNested.WithSelf.F.%.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.23f) [concrete = constants.%.369]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.75d128.2 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.587f77.2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.1f1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.1f1 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1fd = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1fd = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.064 = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.064 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.31b = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.31b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.2), @SelfNestedBadReturnType.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.d9c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadReturnType = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoF {
// CHECK:STDOUT:   impl_decl @NoF.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoF [concrete = constants.%NoF]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoF
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FNotFunction {
// CHECK:STDOUT:   impl_decl @FNotFunction.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FNotFunction [concrete = constants.%FNotFunction]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FNotFunction
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F;
// CHECK:STDOUT:
// CHECK:STDOUT: class @FAlias {
// CHECK:STDOUT:   impl_decl @FAlias.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FAlias [concrete = constants.%FAlias]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FAlias
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .PossiblyF = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraParam {
// CHECK:STDOUT:   impl_decl @FExtraParam.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraParam [concrete = constants.%FExtraParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraParam
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraImplicitParam {
// CHECK:STDOUT:   impl_decl @FExtraImplicitParam.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraImplicitParam
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraReturnType {
// CHECK:STDOUT:   impl_decl @FExtraReturnType.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraReturnType [concrete = constants.%FExtraReturnType]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraReturnType
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingParam {
// CHECK:STDOUT:   impl_decl @FMissingParam.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingParam [concrete = constants.%FMissingParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingParam
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingImplicitParam {
// CHECK:STDOUT:   impl_decl @FMissingImplicitParam.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingImplicitParam [concrete = constants.%FMissingImplicitParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingImplicitParam
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingReturnType {
// CHECK:STDOUT:   impl_decl @FMissingReturnType.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingReturnType [concrete = constants.%FMissingReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingReturnType
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamType {
// CHECK:STDOUT:   impl_decl @FDifferentParamType.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [concrete = constants.%FDifferentParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamType
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentImplicitParamType {
// CHECK:STDOUT:   impl_decl @FDifferentImplicitParamType.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentImplicitParamType
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentReturnType {
// CHECK:STDOUT:   impl_decl @FDifferentReturnType.as.J.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentReturnType
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadParam {
// CHECK:STDOUT:   impl_decl @SelfNestedBadParam.as.SelfNested.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadParam [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadParam
// CHECK:STDOUT:   .SelfNested = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadReturnType {
// CHECK:STDOUT:   impl_decl @SelfNestedBadReturnType.as.SelfNested.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadReturnType [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   .SelfNested = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadReturnType = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PossiblyF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraParam.as.I.impl.F.loc69_18.1(%b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraParam.as.I.impl.F.loc69_18.2() [thunk @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.b53)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraParam.as.I.impl.F.type.531438.1 = name_ref F, @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.33e1fe.1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_23.1(%self.param: %FExtraImplicitParam);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_23.2() [thunk @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.5a5)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraImplicitParam.as.I.impl.F.type.9dd145.1 = name_ref F, @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.99eae8.1]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.call: init %empty_tuple.type = call %F.ref(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraReturnType.as.I.impl.F() -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn(%self.param: bool, %b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc117_31.1(%self.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc117_31.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_31.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.567)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingParam.as.J.impl.F.type.0f71d8.1 = name_ref F, @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_31.1 [concrete = constants.%FMissingParam.as.J.impl.F.7540f6.1]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.1(%b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.266)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingImplicitParam.as.J.impl.F.type.80a56b.1 = name_ref F, @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.c6a0b7.1]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.call: init bool = call %F.ref(%b.param)
// CHECK:STDOUT:   return %FMissingImplicitParam.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc152_30.1(%self.param: bool, %b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc152_30.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc152_30.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.a9e)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingReturnType.as.J.impl.F.type.ce0d21.1 = name_ref F, @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc152_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.5b613a.1]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.call: init %empty_tuple.type = call %FMissingReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc152: bool = converted %FMissingReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc171_38.1(%self.param: bool, %b.param: %FDifferentParamType) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc171_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc171_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.bd2)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentParamType.as.J.impl.F.type.10926d.1 = name_ref F, @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc171_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.f1245a.1]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc103: %FDifferentParamType = converted %b.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.call: init bool = call %FDifferentParamType.as.J.impl.F.bound(%self.param, <error>)
// CHECK:STDOUT:   return %FDifferentParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1(%self.param: %FDifferentImplicitParamType, %b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.030)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentImplicitParamType.as.J.impl.F.type.955391.1 = name_ref F, @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.a9ece4.1]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc103: %FDifferentImplicitParamType = converted %self.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.call: init bool = call %FDifferentImplicitParamType.as.J.impl.F.bound(<error>, %b.param)
// CHECK:STDOUT:   return %FDifferentImplicitParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc200_38.1(%self.param: bool, %b.param: bool) -> out %return.param: %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc200_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc200_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.4a1)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentReturnType.as.J.impl.F.type.7bcb67.1 = name_ref F, @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc200_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.a709b1.1]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc200_38.1: ref %FDifferentReturnType = temporary_storage
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.call: init %FDifferentReturnType to %.loc200_38.1 = call %FDifferentReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc200_38.2: bool = converted %FDifferentReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SelfNested.WithSelf.F(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.aed)]
// CHECK:STDOUT:   %Self.as_type.loc211_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b90)]
// CHECK:STDOUT:   %ptr.loc211_16.1: type = ptr_type %Self.as_type.loc211_16.1 [symbolic = %ptr.loc211_16.1 (constants.%ptr.ad7)]
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc211_16.1 (%Self.as_type.b90), .y: %i32} [symbolic = %struct_type.x.y.loc211_37.1 (constants.%struct_type.x.y.8bf)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc211_16.1, %struct_type.x.y.loc211_37.1) [symbolic = %tuple (constants.%tuple.f6b)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc211_16.1, %struct_type.x.y.loc211_37.1) [symbolic = %tuple.type (constants.%tuple.type.873)]
// CHECK:STDOUT:   %pattern_type.loc211_9: type = pattern_type %tuple.type [symbolic = %pattern_type.loc211_9 (constants.%pattern_type.5a6)]
// CHECK:STDOUT:   %array_type.loc211_57.1: type = array_type constants.%int_4, %Self.as_type.loc211_16.1 [symbolic = %array_type.loc211_57.1 (constants.%array_type.a6c)]
// CHECK:STDOUT:   %.loc211_57.1: Core.Form = init_form %array_type.loc211_57.1 [symbolic = %.loc211_57.1 (constants.%.725)]
// CHECK:STDOUT:   %pattern_type.loc211_57: type = pattern_type %array_type.loc211_57.1 [symbolic = %pattern_type.loc211_57 (constants.%pattern_type.28e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.873)) -> out %return.param: @SelfNested.WithSelf.F.%array_type.loc211_57.1 (%array_type.a6c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1(%x.param: %tuple.type.70f) -> out %return.param: %array_type.388;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2(%x.param: %tuple.type.7ea) -> out %return.param: %array_type.388 [thunk @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.b9f)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadParam.as.SelfNested.impl.F.type.789632.1 = name_ref F, @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.6e1661.1]
// CHECK:STDOUT:   %.loc211_57: ref %array_type.388 = splice_block %return.param {}
// CHECK:STDOUT:   %tuple.elem0: %ptr.cf2 = tuple_access %x.param, element0
// CHECK:STDOUT:   %tuple.elem1: %struct_type.x.y.62a = tuple_access %x.param, element1
// CHECK:STDOUT:   %.loc211_9.1: %SelfNestedBadParam = struct_access %tuple.elem1, element0
// CHECK:STDOUT:   %.loc211_9.2: %i32 = converted %.loc211_9.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.call: init %array_type.388 to %.loc211_57 = call %F.ref(<error>)
// CHECK:STDOUT:   return %SelfNestedBadParam.as.SelfNested.impl.F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1(%x.param: %tuple.type.064) -> out %return.param: %array_type.388;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2(%x.param: %tuple.type.064) -> out %return.param: %array_type.31b [thunk @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.23f)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.75d128.1 = name_ref F, @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.587f77.1]
// CHECK:STDOUT:   %.loc239_112.1: ref %array_type.388 = temporary_storage
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.call: init %array_type.388 to %.loc239_112.1 = call %F.ref(%x.param)
// CHECK:STDOUT:   %.loc239_112.2: %array_type.31b = converted %SelfNestedBadReturnType.as.SelfNested.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.ab9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.ab9
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.08c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.705
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.ab9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.fc8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.fc8
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.cec
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.733
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.c1a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.c1a
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.1b8
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.7e3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.574) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.574
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.f26
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.5e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.b53) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.b53
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.99d
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.eab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.b53) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.5a5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.5a5
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.e46
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.5bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.5a5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.1ed) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.1ed
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.cd5
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.aee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.1ed) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.8a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.8a1
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.adb
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.569
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.8a1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.567) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.567
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.54f
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.7e8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.567) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.266) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.266
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.56e
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.d17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.266) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.a9e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.a9e
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.8be
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.2a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.a9e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.bd2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.bd2
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.06a
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.cad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.bd2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.030) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.030
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.f29
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.5b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.030) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.4a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.4a1
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.8e3
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.4a1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%Self.aed) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.aed
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.430
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.257
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%Self.aed) {
// CHECK:STDOUT:   %Self => constants.%Self.aed
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%Self.as_type.b90
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.ad7
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.8bf
// CHECK:STDOUT:   %tuple => constants.%tuple.f6b
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.873
// CHECK:STDOUT:   %pattern_type.loc211_9 => constants.%pattern_type.5a6
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.a6c
// CHECK:STDOUT:   %.loc211_57.1 => constants.%.725
// CHECK:STDOUT:   %pattern_type.loc211_57 => constants.%pattern_type.28e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.b9f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.b9f
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.4de
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.5e9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.b9f) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.b9f
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.cf2
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.62a
// CHECK:STDOUT:   %tuple => constants.%tuple.73b
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.7ea
// CHECK:STDOUT:   %pattern_type.loc211_9 => constants.%pattern_type.8d7
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.388
// CHECK:STDOUT:   %.loc211_57.1 => constants.%.0cb
// CHECK:STDOUT:   %pattern_type.loc211_57 => constants.%pattern_type.ec1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.23f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.23f
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.156
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.edc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.23f) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.23f
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.cbe
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.cfa
// CHECK:STDOUT:   %tuple => constants.%tuple.c42
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.064
// CHECK:STDOUT:   %pattern_type.loc211_9 => constants.%pattern_type.1f1
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.31b
// CHECK:STDOUT:   %.loc211_57.1 => constants.%.369
// CHECK:STDOUT:   %pattern_type.loc211_57 => constants.%pattern_type.1fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
