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
// CHECK:STDOUT:   %Self.005: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.4dd: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.005) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.18f: %I.WithSelf.F.type.4dd = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.8bc: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [concrete]
// CHECK:STDOUT:   %I.impl_witness.7b6: <witness> = impl_witness @NoF.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.4c9: %I.type = facet_value %NoF, (%I.impl_witness.7b6) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.8d2: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.4c9) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.3f2: %I.WithSelf.F.type.8d2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [concrete]
// CHECK:STDOUT:   %I.impl_witness.da8: <witness> = impl_witness @FNotFunction.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %F: type = class_type @F [concrete]
// CHECK:STDOUT:   %I.facet.b57: %I.type = facet_value %FNotFunction, (%I.impl_witness.da8) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.6f0: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.b57) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.f52: %I.WithSelf.F.type.6f0 = struct_value () [concrete]
// CHECK:STDOUT:   %PossiblyF.type: type = fn_type @PossiblyF [concrete]
// CHECK:STDOUT:   %PossiblyF: %PossiblyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [concrete]
// CHECK:STDOUT:   %I.impl_witness.bae: <witness> = impl_witness @FAlias.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.f8e: %I.type = facet_value %FAlias, (%I.impl_witness.bae) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.379: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.f8e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.97e: %I.WithSelf.F.type.379 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.7b8: <witness> = impl_witness @FExtraParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.f79c0b.1: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.1 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.d99edf.1: %FExtraParam.as.I.impl.F.type.f79c0b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.7d0: %I.type = facet_value %FExtraParam, (%I.impl_witness.7b8) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.bb0: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.7d0) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.64a: %I.WithSelf.F.type.bb0 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.f79c0b.2: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.2 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.d99edf.2: %FExtraParam.as.I.impl.F.type.f79c0b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.001: <witness> = impl_witness @FExtraImplicitParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d16: type = pattern_type %FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.683510.1: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_23.1 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.862c08.1: %FExtraImplicitParam.as.I.impl.F.type.683510.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.21e: %I.type = facet_value %FExtraImplicitParam, (%I.impl_witness.001) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.b01: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.21e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.4ff: %I.WithSelf.F.type.b01 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.683510.2: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_23.2 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.862c08.2: %FExtraImplicitParam.as.I.impl.F.type.683510.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [concrete]
// CHECK:STDOUT:   %I.impl_witness.e8a: <witness> = impl_witness @FExtraReturnType.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.type: type = fn_type @FExtraReturnType.as.I.impl.F [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F: %FExtraReturnType.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.00b: %I.type = facet_value %FExtraReturnType, (%I.impl_witness.e8a) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.923: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.00b) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.d19: %I.WithSelf.F.type.923 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.932: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.98d: type = fn_type @J.WithSelf.F, @J.WithSelf(%Self.932) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.85b: %J.WithSelf.F.type.98d = struct_value () [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.830: %J.assoc_type = assoc_entity element0, @J.WithSelf.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.34c: <witness> = impl_witness @FMissingParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.55db91.1: type = fn_type @FMissingParam.as.J.impl.F.loc117_31.1 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.5f8454.1: %FMissingParam.as.J.impl.F.type.55db91.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.f48: %J.type = facet_value %FMissingParam, (%J.impl_witness.34c) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.193: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.f48) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.9d1: %J.WithSelf.F.type.193 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.55db91.2: type = fn_type @FMissingParam.as.J.impl.F.loc117_31.2 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.5f8454.2: %FMissingParam.as.J.impl.F.type.55db91.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.26c: <witness> = impl_witness @FMissingImplicitParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.969d7c.1: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.010b18.1: %FMissingImplicitParam.as.J.impl.F.type.969d7c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.6d7: %J.type = facet_value %FMissingImplicitParam, (%J.impl_witness.26c) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.a15: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.6d7) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.e8a: %J.WithSelf.F.type.a15 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.969d7c.2: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.010b18.2: %FMissingImplicitParam.as.J.impl.F.type.969d7c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.9b6: <witness> = impl_witness @FMissingReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.2c5d48.1: type = fn_type @FMissingReturnType.as.J.impl.F.loc152_30.1 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.26fa30.1: %FMissingReturnType.as.J.impl.F.type.2c5d48.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.db8: %J.type = facet_value %FMissingReturnType, (%J.impl_witness.9b6) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.457: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.db8) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.b76: %J.WithSelf.F.type.457 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.2c5d48.2: type = fn_type @FMissingReturnType.as.J.impl.F.loc152_30.2 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.26fa30.2: %FMissingReturnType.as.J.impl.F.type.2c5d48.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.2cd: <witness> = impl_witness @FDifferentParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %FDifferentParamType [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.a80439.1: type = fn_type @FDifferentParamType.as.J.impl.F.loc171_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.591ebb.1: %FDifferentParamType.as.J.impl.F.type.a80439.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.930: %J.type = facet_value %FDifferentParamType, (%J.impl_witness.2cd) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.7c8: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.930) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.367: %J.WithSelf.F.type.7c8 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.a80439.2: type = fn_type @FDifferentParamType.as.J.impl.F.loc171_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.591ebb.2: %FDifferentParamType.as.J.impl.F.type.a80439.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.93f: <witness> = impl_witness @FDifferentImplicitParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.f4f: type = pattern_type %FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.bcdb66.1: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.6a695d.1: %FDifferentImplicitParamType.as.J.impl.F.type.bcdb66.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.fc2: %J.type = facet_value %FDifferentImplicitParamType, (%J.impl_witness.93f) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.853: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.fc2) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.646: %J.WithSelf.F.type.853 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.bcdb66.2: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.6a695d.2: %FDifferentImplicitParamType.as.J.impl.F.type.bcdb66.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.65c: <witness> = impl_witness @FDifferentReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %.71e: Core.Form = init_form %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.be4: type = pattern_type %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.b06b09.1: type = fn_type @FDifferentReturnType.as.J.impl.F.loc200_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.6734ae.1: %FDifferentReturnType.as.J.impl.F.type.b06b09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.aef: %J.type = facet_value %FDifferentReturnType, (%J.impl_witness.65c) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.ac3: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.aef) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.1f0: %J.WithSelf.F.type.ac3 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.b06b09.2: type = fn_type @FDifferentReturnType.as.J.impl.F.loc200_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.6734ae.2: %FDifferentReturnType.as.J.impl.F.type.b06b09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.b20: %SelfNested.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7fd: type = facet_access_type %Self.b20 [symbolic]
// CHECK:STDOUT:   %ptr.845: type = ptr_type %Self.as_type.7fd [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.293: type = struct_type {.x: %Self.as_type.7fd, .y: %i32} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.28e: %tuple.type.24b = tuple_value (%ptr.845, %struct_type.x.y.293) [symbolic]
// CHECK:STDOUT:   %tuple.type.6dd: type = tuple_type (%ptr.845, %struct_type.x.y.293) [symbolic]
// CHECK:STDOUT:   %pattern_type.ce1: type = pattern_type %tuple.type.6dd [symbolic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.d6a: type = array_type %int_4, %Self.as_type.7fd [symbolic]
// CHECK:STDOUT:   %.82e: Core.Form = init_form %array_type.d6a [symbolic]
// CHECK:STDOUT:   %pattern_type.7f1: type = pattern_type %array_type.d6a [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.3c8: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%Self.b20) [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.3e1: %SelfNested.WithSelf.F.type.3c8 = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.831: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.a29: <witness> = impl_witness @SelfNestedBadParam.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.178: type = ptr_type %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %struct_type.x.y.9f1: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.e54: %tuple.type.24b = tuple_value (%ptr.178, %struct_type.x.y.9f1) [concrete]
// CHECK:STDOUT:   %tuple.type.a64: type = tuple_type (%ptr.178, %struct_type.x.y.9f1) [concrete]
// CHECK:STDOUT:   %pattern_type.abe: type = pattern_type %tuple.type.a64 [concrete]
// CHECK:STDOUT:   %array_type.efc: type = array_type %int_4, %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %.209: Core.Form = init_form %array_type.efc [concrete]
// CHECK:STDOUT:   %pattern_type.d59: type = pattern_type %array_type.efc [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.6dcd2a.1: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.145046.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.6dcd2a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.30b: %SelfNested.type = facet_value %SelfNestedBadParam, (%SelfNested.impl_witness.a29) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.2bf: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.30b) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.ac3: %SelfNested.WithSelf.F.type.2bf = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.d6a: type = struct_type {.x: %SelfNestedBadParam, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.15d: %tuple.type.24b = tuple_value (%ptr.178, %struct_type.x.y.d6a) [concrete]
// CHECK:STDOUT:   %tuple.type.aa7: type = tuple_type (%ptr.178, %struct_type.x.y.d6a) [concrete]
// CHECK:STDOUT:   %pattern_type.91e: type = pattern_type %tuple.type.aa7 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.6dcd2a.2: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.145046.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.6dcd2a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.786: <witness> = impl_witness @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.d9b: type = ptr_type %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %struct_type.x.y.b3a: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.368: %tuple.type.24b = tuple_value (%ptr.d9b, %struct_type.x.y.b3a) [concrete]
// CHECK:STDOUT:   %tuple.type.8c5: type = tuple_type (%ptr.d9b, %struct_type.x.y.b3a) [concrete]
// CHECK:STDOUT:   %pattern_type.e48: type = pattern_type %tuple.type.8c5 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.490975.1: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.5b61eb.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.490975.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.0de: %SelfNested.type = facet_value %SelfNestedBadReturnType, (%SelfNested.impl_witness.786) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.689: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.0de) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.123: %SelfNested.WithSelf.F.type.689 = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.894: type = array_type %int_4, %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %.b4c: Core.Form = init_form %array_type.894 [concrete]
// CHECK:STDOUT:   %pattern_type.748: type = pattern_type %array_type.894 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.490975.2: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.5b61eb.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.490975.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.b23 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.005]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.4dd) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.18f)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.8bc]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @I.WithSelf.%assoc0
// CHECK:STDOUT:   .PossiblyF = <poisoned>
// CHECK:STDOUT:   witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.932]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.WithSelf.%J.WithSelf.F.type (%J.WithSelf.F.type.98d) = fn_decl @J.WithSelf.F [symbolic = @J.WithSelf.%J.WithSelf.F (constants.%J.WithSelf.F.85b)] {
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
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.WithSelf.F.decl [concrete = constants.%assoc0.830]
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
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = constants.%Self.b20]
// CHECK:STDOUT:   %SelfNested.WithSelf.decl = interface_with_self_decl @SelfNested [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %SelfNested.WithSelf.F.decl: @SelfNested.WithSelf.%SelfNested.WithSelf.F.type (%SelfNested.WithSelf.F.type.3c8) = fn_decl @SelfNested.WithSelf.F [symbolic = @SelfNested.WithSelf.%SelfNested.WithSelf.F (constants.%SelfNested.WithSelf.F.3e1)] {
// CHECK:STDOUT:     %x.param_patt: @SelfNested.WithSelf.F.%pattern_type.loc211_9 (%pattern_type.ce1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @SelfNested.WithSelf.F.%pattern_type.loc211_9 (%pattern_type.ce1) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @SelfNested.WithSelf.F.%pattern_type.loc211_57 (%pattern_type.7f1) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @SelfNested.WithSelf.F.%pattern_type.loc211_57 (%pattern_type.7f1) = return_slot_pattern %return.param_patt, %array_type.loc211_57.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc211_50: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.b20)]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Self.as_type.loc211_50: type = facet_access_type %Self.ref.loc211_50 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.7fd)]
// CHECK:STDOUT:     %.loc211_50: type = converted %Self.ref.loc211_50, %Self.as_type.loc211_50 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.7fd)]
// CHECK:STDOUT:     %array_type.loc211_57.2: type = array_type %int_4, %.loc211_50 [symbolic = %array_type.loc211_57.1 (constants.%array_type.d6a)]
// CHECK:STDOUT:     %.loc211_57.2: Core.Form = init_form %array_type.loc211_57.2 [symbolic = %.loc211_57.1 (constants.%.82e)]
// CHECK:STDOUT:     %x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.6dd) = value_param call_param0
// CHECK:STDOUT:     %.loc211_38.1: type = splice_block %.loc211_38.3 [symbolic = %tuple.type (constants.%tuple.type.6dd)] {
// CHECK:STDOUT:       %Self.ref.loc211_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.b20)]
// CHECK:STDOUT:       %Self.as_type.loc211_16.2: type = facet_access_type %Self.ref.loc211_12 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.7fd)]
// CHECK:STDOUT:       %.loc211_16: type = converted %Self.ref.loc211_12, %Self.as_type.loc211_16.2 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.7fd)]
// CHECK:STDOUT:       %ptr.loc211_16.2: type = ptr_type %.loc211_16 [symbolic = %ptr.loc211_16.1 (constants.%ptr.845)]
// CHECK:STDOUT:       %Self.ref.loc211_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.b20)]
// CHECK:STDOUT:       %Self.as_type.loc211_24: type = facet_access_type %Self.ref.loc211_24 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.7fd)]
// CHECK:STDOUT:       %.loc211_24: type = converted %Self.ref.loc211_24, %Self.as_type.loc211_24 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.7fd)]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y.loc211_37.2: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc211_16.1 (%Self.as_type.7fd), .y: %i32} [symbolic = %struct_type.x.y.loc211_37.1 (constants.%struct_type.x.y.293)]
// CHECK:STDOUT:       %.loc211_38.2: %tuple.type.24b = tuple_literal (%ptr.loc211_16.2, %struct_type.x.y.loc211_37.2) [symbolic = %tuple (constants.%tuple.28e)]
// CHECK:STDOUT:       %.loc211_38.3: type = converted %.loc211_38.2, constants.%tuple.type.6dd [symbolic = %tuple.type (constants.%tuple.type.6dd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.6dd) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @SelfNested.WithSelf.F.%array_type.loc211_57.1 (%array_type.d6a) = out_param call_param1
// CHECK:STDOUT:     %return: ref @SelfNested.WithSelf.F.%array_type.loc211_57.1 (%array_type.d6a) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %SelfNested.WithSelf.F.decl [concrete = constants.%assoc0.831]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @SelfNested.WithSelf.%assoc0
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadReturnType = <poisoned>
// CHECK:STDOUT:   witness = (@SelfNested.WithSelf.%SelfNested.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @NoF.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @NoF.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.7b6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FNotFunction.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: type = class_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FNotFunction.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.da8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FAlias.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %PossiblyF.ref: %PossiblyF.type = name_ref PossiblyF, file.%PossiblyF.decl [concrete = constants.%PossiblyF]
// CHECK:STDOUT:   %F: %PossiblyF.type = alias_binding F, %PossiblyF.ref [concrete = constants.%PossiblyF]
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FAlias.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.bae]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .PossiblyF = <poisoned>
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.1: %FExtraParam.as.I.impl.F.type.f79c0b.1 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.d99edf.1] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc69: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.2: %FExtraParam.as.I.impl.F.type.f79c0b.2 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.2 [concrete = constants.%FExtraParam.as.I.impl.F.d99edf.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraParam.as.I.impl.F.decl.loc69_18.2), @FExtraParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.7b8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraParam.as.I.impl.F.decl.loc69_18.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraImplicitParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1: %FExtraImplicitParam.as.I.impl.F.type.683510.1 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_23.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.862c08.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d16 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.d16 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self: %FExtraImplicitParam = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.2: %FExtraImplicitParam.as.I.impl.F.type.683510.2 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_23.2 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.862c08.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.2), @FExtraImplicitParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.001]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1
// CHECK:STDOUT:   extend %I.ref
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.e8a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraReturnType.as.I.impl.F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_31.1: %FMissingParam.as.J.impl.F.type.55db91.1 = fn_decl @FMissingParam.as.J.impl.F.loc117_31.1 [concrete = constants.%FMissingParam.as.J.impl.F.5f8454.1] {
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.f48) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.f48) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_31.2: %FMissingParam.as.J.impl.F.type.55db91.2 = fn_decl @FMissingParam.as.J.impl.F.loc117_31.2 [concrete = constants.%FMissingParam.as.J.impl.F.5f8454.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.34c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingParam.as.J.impl.F.decl.loc117_31.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingImplicitParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1: %FMissingImplicitParam.as.J.impl.F.type.969d7c.1 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.010b18.1] {
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.6d7) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.6d7) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.2: %FMissingImplicitParam.as.J.impl.F.type.969d7c.2 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.010b18.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.26c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc152_30.1: %FMissingReturnType.as.J.impl.F.type.2c5d48.1 = fn_decl @FMissingReturnType.as.J.impl.F.loc152_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.26fa30.1] {
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.db8) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.db8) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc152_30.2: %FMissingReturnType.as.J.impl.F.type.2c5d48.2 = fn_decl @FMissingReturnType.as.J.impl.F.loc152_30.2 [concrete = constants.%FMissingReturnType.as.J.impl.F.26fa30.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.9b6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingReturnType.as.J.impl.F.decl.loc152_30.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc171_38.1: %FDifferentParamType.as.J.impl.F.type.a80439.1 = fn_decl @FDifferentParamType.as.J.impl.F.loc171_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.591ebb.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.4a0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.4a0 = at_binding_pattern b, %b.param_patt [concrete]
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.930) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.930) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc171_38.2: %FDifferentParamType.as.J.impl.F.type.a80439.2 = fn_decl @FDifferentParamType.as.J.impl.F.loc171_38.2 [concrete = constants.%FDifferentParamType.as.J.impl.F.591ebb.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.2cd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentParamType.as.J.impl.F.decl.loc171_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentImplicitParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1: %FDifferentImplicitParamType.as.J.impl.F.type.bcdb66.1 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.6a695d.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f4f = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.f4f = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.fc2) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.fc2) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.2: %FDifferentImplicitParamType.as.J.impl.F.type.bcdb66.2 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.6a695d.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.93f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc200_38.1: %FDifferentReturnType.as.J.impl.F.type.b06b09.1 = fn_decl @FDifferentReturnType.as.J.impl.F.loc200_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.6734ae.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.be4 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.be4 = return_slot_pattern %return.param_patt, %Self.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %.loc200_34: Core.Form = init_form %Self.ref [concrete = constants.%.71e]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc200_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc200_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param call_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.aef) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.aef) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc200_38.2: %FDifferentReturnType.as.J.impl.F.type.b06b09.2 = fn_decl @FDifferentReturnType.as.J.impl.F.loc200_38.2 [concrete = constants.%FDifferentReturnType.as.J.impl.F.6734ae.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.65c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentReturnType.as.J.impl.F.decl.loc200_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadParam.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.6dcd2a.1 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.145046.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.abe = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.abe = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d59 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d59 = return_slot_pattern %return.param_patt, %array_type [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc223_65: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref.loc223_65 [concrete = constants.%array_type.efc]
// CHECK:STDOUT:     %.loc223_86: Core.Form = init_form %array_type [concrete = constants.%.209]
// CHECK:STDOUT:     %x.param: %tuple.type.a64 = value_param call_param0
// CHECK:STDOUT:     %.loc223_53.1: type = splice_block %.loc223_53.3 [concrete = constants.%tuple.type.a64] {
// CHECK:STDOUT:       %SelfNestedBadParam.ref.loc223_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadParam.ref.loc223_14 [concrete = constants.%ptr.178]
// CHECK:STDOUT:       %i32.loc223_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc223_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete = constants.%struct_type.x.y.9f1]
// CHECK:STDOUT:       %.loc223_53.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.e54]
// CHECK:STDOUT:       %.loc223_53.3: type = converted %.loc223_53.2, constants.%tuple.type.a64 [concrete = constants.%tuple.type.a64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a64 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.efc = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.efc = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc211_57.1: type = specific_constant @SelfNested.WithSelf.F.%array_type.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.30b) [concrete = constants.%array_type.efc]
// CHECK:STDOUT:   %.loc211_57.2: type = specific_constant @SelfNested.WithSelf.F.%.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.30b) [concrete = constants.%.209]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.6dcd2a.2 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.145046.2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.91e = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.91e = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d59 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d59 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.aa7 = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.aa7 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.efc = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.efc = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.2), @SelfNestedBadParam.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.a29]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1
// CHECK:STDOUT:   extend %SelfNested.ref
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadReturnType.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.490975.1 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.5b61eb.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.e48 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.e48 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d59 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d59 = return_slot_pattern %return.param_patt, %array_type [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref [concrete = constants.%array_type.efc]
// CHECK:STDOUT:     %.loc239_111: Core.Form = init_form %array_type [concrete = constants.%.209]
// CHECK:STDOUT:     %x.param: %tuple.type.8c5 = value_param call_param0
// CHECK:STDOUT:     %.loc239_78.1: type = splice_block %.loc239_78.3 [concrete = constants.%tuple.type.8c5] {
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc239_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadReturnType.ref.loc239_14 [concrete = constants.%ptr.d9b]
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc239_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete = constants.%struct_type.x.y.b3a]
// CHECK:STDOUT:       %.loc239_78.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.368]
// CHECK:STDOUT:       %.loc239_78.3: type = converted %.loc239_78.2, constants.%tuple.type.8c5 [concrete = constants.%tuple.type.8c5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.8c5 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.efc = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.efc = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc211_57.1: type = specific_constant @SelfNested.WithSelf.F.%array_type.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.0de) [concrete = constants.%array_type.894]
// CHECK:STDOUT:   %.loc211_57.2: type = specific_constant @SelfNested.WithSelf.F.%.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.0de) [concrete = constants.%.b4c]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.490975.2 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.5b61eb.2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.e48 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.e48 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.748 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.748 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.8c5 = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.8c5 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.894 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.894 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.2), @SelfNestedBadReturnType.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.786]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadReturnType = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1
// CHECK:STDOUT:   extend %SelfNested.ref
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
// CHECK:STDOUT: fn @FExtraParam.as.I.impl.F.loc69_18.2() [thunk @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.7d0)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraParam.as.I.impl.F.type.f79c0b.1 = name_ref F, @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.d99edf.1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_23.1(%self.param: %FExtraImplicitParam);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_23.2() [thunk @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.21e)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraImplicitParam.as.I.impl.F.type.683510.1 = name_ref F, @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.862c08.1]
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
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc117_31.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_31.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.f48)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingParam.as.J.impl.F.type.55db91.1 = name_ref F, @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_31.1 [concrete = constants.%FMissingParam.as.J.impl.F.5f8454.1]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.1(%b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.6d7)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingImplicitParam.as.J.impl.F.type.969d7c.1 = name_ref F, @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.010b18.1]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.call: init bool = call %F.ref(%b.param)
// CHECK:STDOUT:   return %FMissingImplicitParam.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc152_30.1(%self.param: bool, %b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc152_30.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc152_30.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.db8)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingReturnType.as.J.impl.F.type.2c5d48.1 = name_ref F, @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc152_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.26fa30.1]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.call: init %empty_tuple.type = call %FMissingReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc152: bool = converted %FMissingReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc171_38.1(%self.param: bool, %b.param: %FDifferentParamType) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc171_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc171_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.930)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentParamType.as.J.impl.F.type.a80439.1 = name_ref F, @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc171_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.591ebb.1]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc103: %FDifferentParamType = converted %b.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.call: init bool = call %FDifferentParamType.as.J.impl.F.bound(%self.param, <error>)
// CHECK:STDOUT:   return %FDifferentParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1(%self.param: %FDifferentImplicitParamType, %b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.fc2)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentImplicitParamType.as.J.impl.F.type.bcdb66.1 = name_ref F, @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.6a695d.1]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc103: %FDifferentImplicitParamType = converted %self.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.call: init bool = call %FDifferentImplicitParamType.as.J.impl.F.bound(<error>, %b.param)
// CHECK:STDOUT:   return %FDifferentImplicitParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc200_38.1(%self.param: bool, %b.param: bool) -> out %return.param: %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc200_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc200_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.aef)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentReturnType.as.J.impl.F.type.b06b09.1 = name_ref F, @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc200_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.6734ae.1]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc200_38.1: ref %FDifferentReturnType = temporary_storage
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.call: init %FDifferentReturnType to %.loc200_38.1 = call %FDifferentReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc200_38.2: bool = converted %FDifferentReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SelfNested.WithSelf.F(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.b20)]
// CHECK:STDOUT:   %Self.as_type.loc211_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.7fd)]
// CHECK:STDOUT:   %ptr.loc211_16.1: type = ptr_type %Self.as_type.loc211_16.1 [symbolic = %ptr.loc211_16.1 (constants.%ptr.845)]
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc211_16.1 (%Self.as_type.7fd), .y: %i32} [symbolic = %struct_type.x.y.loc211_37.1 (constants.%struct_type.x.y.293)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc211_16.1, %struct_type.x.y.loc211_37.1) [symbolic = %tuple (constants.%tuple.28e)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc211_16.1, %struct_type.x.y.loc211_37.1) [symbolic = %tuple.type (constants.%tuple.type.6dd)]
// CHECK:STDOUT:   %pattern_type.loc211_9: type = pattern_type %tuple.type [symbolic = %pattern_type.loc211_9 (constants.%pattern_type.ce1)]
// CHECK:STDOUT:   %array_type.loc211_57.1: type = array_type constants.%int_4, %Self.as_type.loc211_16.1 [symbolic = %array_type.loc211_57.1 (constants.%array_type.d6a)]
// CHECK:STDOUT:   %.loc211_57.1: Core.Form = init_form %array_type.loc211_57.1 [symbolic = %.loc211_57.1 (constants.%.82e)]
// CHECK:STDOUT:   %pattern_type.loc211_57: type = pattern_type %array_type.loc211_57.1 [symbolic = %pattern_type.loc211_57 (constants.%pattern_type.7f1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.6dd)) -> out %return.param: @SelfNested.WithSelf.F.%array_type.loc211_57.1 (%array_type.d6a);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1(%x.param: %tuple.type.a64) -> out %return.param: %array_type.efc;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2(%x.param: %tuple.type.aa7) -> out %return.param: %array_type.efc [thunk @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.30b)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadParam.as.SelfNested.impl.F.type.6dcd2a.1 = name_ref F, @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.145046.1]
// CHECK:STDOUT:   %.loc211_57: ref %array_type.efc = splice_block %return.param {}
// CHECK:STDOUT:   %tuple.elem0: %ptr.178 = tuple_access %x.param, element0
// CHECK:STDOUT:   %tuple.elem1: %struct_type.x.y.d6a = tuple_access %x.param, element1
// CHECK:STDOUT:   %.loc211_9.1: %SelfNestedBadParam = struct_access %tuple.elem1, element0
// CHECK:STDOUT:   %.loc211_9.2: %i32 = converted %.loc211_9.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.call: init %array_type.efc to %.loc211_57 = call %F.ref(<error>)
// CHECK:STDOUT:   return %SelfNestedBadParam.as.SelfNested.impl.F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1(%x.param: %tuple.type.8c5) -> out %return.param: %array_type.efc;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2(%x.param: %tuple.type.8c5) -> out %return.param: %array_type.894 [thunk @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.0de)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.490975.1 = name_ref F, @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.5b61eb.1]
// CHECK:STDOUT:   %.loc239_112.1: ref %array_type.efc = temporary_storage
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.call: init %array_type.efc to %.loc239_112.1 = call %F.ref(%x.param)
// CHECK:STDOUT:   %.loc239_112.2: %array_type.894 = converted %SelfNestedBadReturnType.as.SelfNested.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.005) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.005
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.4dd
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.18f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.005) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.4c9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.4c9
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.8d2
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.3f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.b57) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.b57
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.6f0
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.f52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.f8e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.f8e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.379
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.97e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.7d0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.7d0
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.bb0
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.64a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.7d0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.21e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.21e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.b01
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.4ff
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.21e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.00b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.00b
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.923
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.d19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.00b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.932) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.932
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.98d
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.85b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.932) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.f48) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.f48
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.193
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.9d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.f48) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.6d7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.6d7
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.a15
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.e8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.6d7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.db8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.db8
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.457
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.b76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.db8) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.930) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.930
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.7c8
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.367
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.930) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.fc2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.fc2
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.853
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.646
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.fc2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.aef) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.aef
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.ac3
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.1f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.aef) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%Self.b20) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.b20
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.3c8
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.3e1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%Self.b20) {
// CHECK:STDOUT:   %Self => constants.%Self.b20
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%Self.as_type.7fd
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.845
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.293
// CHECK:STDOUT:   %tuple => constants.%tuple.28e
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.6dd
// CHECK:STDOUT:   %pattern_type.loc211_9 => constants.%pattern_type.ce1
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.d6a
// CHECK:STDOUT:   %.loc211_57.1 => constants.%.82e
// CHECK:STDOUT:   %pattern_type.loc211_57 => constants.%pattern_type.7f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.30b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.30b
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.2bf
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.ac3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.30b) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.30b
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.178
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.d6a
// CHECK:STDOUT:   %tuple => constants.%tuple.15d
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.aa7
// CHECK:STDOUT:   %pattern_type.loc211_9 => constants.%pattern_type.91e
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.efc
// CHECK:STDOUT:   %.loc211_57.1 => constants.%.209
// CHECK:STDOUT:   %pattern_type.loc211_57 => constants.%pattern_type.d59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.0de) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.0de
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.689
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.0de) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.0de
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.d9b
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.b3a
// CHECK:STDOUT:   %tuple => constants.%tuple.368
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.8c5
// CHECK:STDOUT:   %pattern_type.loc211_9 => constants.%pattern_type.e48
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.894
// CHECK:STDOUT:   %.loc211_57.1 => constants.%.b4c
// CHECK:STDOUT:   %pattern_type.loc211_57 => constants.%pattern_type.748
// CHECK:STDOUT: }
// CHECK:STDOUT:
