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
// CHECK:STDOUT:   %Self.cd6: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.831: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.cd6) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ddf: %I.WithSelf.F.type.831 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.918: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [concrete]
// CHECK:STDOUT:   %I.impl_witness.89a: <witness> = impl_witness @NoF.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.469: %I.type = facet_value %NoF, (%I.impl_witness.89a) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.9ea: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.469) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.528: %I.WithSelf.F.type.9ea = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [concrete]
// CHECK:STDOUT:   %I.impl_witness.377: <witness> = impl_witness @FNotFunction.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %F: type = class_type @F [concrete]
// CHECK:STDOUT:   %I.facet.1ce: %I.type = facet_value %FNotFunction, (%I.impl_witness.377) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.a4c: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.1ce) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.2f2: %I.WithSelf.F.type.a4c = struct_value () [concrete]
// CHECK:STDOUT:   %PossiblyF.type: type = fn_type @PossiblyF [concrete]
// CHECK:STDOUT:   %PossiblyF: %PossiblyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [concrete]
// CHECK:STDOUT:   %I.impl_witness.1ac: <witness> = impl_witness @FAlias.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.0a8: %I.type = facet_value %FAlias, (%I.impl_witness.1ac) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.c54: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.0a8) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.c60: %I.WithSelf.F.type.c54 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.efa: <witness> = impl_witness @FExtraParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.cb12e5.1: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.1 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.989bc3.1: %FExtraParam.as.I.impl.F.type.cb12e5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.50c: %I.type = facet_value %FExtraParam, (%I.impl_witness.efa) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.e8f: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.50c) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.df3: %I.WithSelf.F.type.e8f = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.cb12e5.2: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.2 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.989bc3.2: %FExtraParam.as.I.impl.F.type.cb12e5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.303: <witness> = impl_witness @FExtraImplicitParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ee: type = pattern_type %FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.f5adc6.1: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_23.1 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.8df3a7.1: %FExtraImplicitParam.as.I.impl.F.type.f5adc6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.f28: %I.type = facet_value %FExtraImplicitParam, (%I.impl_witness.303) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.ff9: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.f28) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.e60: %I.WithSelf.F.type.ff9 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.f5adc6.2: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_23.2 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.8df3a7.2: %FExtraImplicitParam.as.I.impl.F.type.f5adc6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [concrete]
// CHECK:STDOUT:   %I.impl_witness.0f7: <witness> = impl_witness @FExtraReturnType.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.type: type = fn_type @FExtraReturnType.as.I.impl.F [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F: %FExtraReturnType.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.d6e: %I.type = facet_value %FExtraReturnType, (%I.impl_witness.0f7) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.37b: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.d6e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.3ab: %I.WithSelf.F.type.37b = struct_value () [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.c93: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.type.66e: type = fn_type @J.WithSelf.F, @J.WithSelf(%Self.c93) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.876: %J.WithSelf.F.type.66e = struct_value () [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.7d2: %J.assoc_type = assoc_entity element0, @J.WithSelf.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.e6f: <witness> = impl_witness @FMissingParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.5ef2ac.1: type = fn_type @FMissingParam.as.J.impl.F.loc117_31.1 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.d8edee.1: %FMissingParam.as.J.impl.F.type.5ef2ac.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.4dc: %J.type = facet_value %FMissingParam, (%J.impl_witness.e6f) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.0f0: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.4dc) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.200: %J.WithSelf.F.type.0f0 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.5ef2ac.2: type = fn_type @FMissingParam.as.J.impl.F.loc117_31.2 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.d8edee.2: %FMissingParam.as.J.impl.F.type.5ef2ac.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.e96: <witness> = impl_witness @FMissingImplicitParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.f8427f.1: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.b398ca.1: %FMissingImplicitParam.as.J.impl.F.type.f8427f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.158: %J.type = facet_value %FMissingImplicitParam, (%J.impl_witness.e96) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.72a: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.158) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.5e6: %J.WithSelf.F.type.72a = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.f8427f.2: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.b398ca.2: %FMissingImplicitParam.as.J.impl.F.type.f8427f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.3a9: <witness> = impl_witness @FMissingReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.78fab6.1: type = fn_type @FMissingReturnType.as.J.impl.F.loc152_30.1 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.55cda9.1: %FMissingReturnType.as.J.impl.F.type.78fab6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.2d3: %J.type = facet_value %FMissingReturnType, (%J.impl_witness.3a9) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.d1d: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.2d3) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.eba: %J.WithSelf.F.type.d1d = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.78fab6.2: type = fn_type @FMissingReturnType.as.J.impl.F.loc152_30.2 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.55cda9.2: %FMissingReturnType.as.J.impl.F.type.78fab6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.eeb: <witness> = impl_witness @FDifferentParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.789: type = pattern_type %FDifferentParamType [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.a9aa82.1: type = fn_type @FDifferentParamType.as.J.impl.F.loc171_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.9d056e.1: %FDifferentParamType.as.J.impl.F.type.a9aa82.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.e11: %J.type = facet_value %FDifferentParamType, (%J.impl_witness.eeb) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.0de: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.e11) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.d73: %J.WithSelf.F.type.0de = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.a9aa82.2: type = fn_type @FDifferentParamType.as.J.impl.F.loc171_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.9d056e.2: %FDifferentParamType.as.J.impl.F.type.a9aa82.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.a56: <witness> = impl_witness @FDifferentImplicitParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.5a0: type = pattern_type %FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.2a4475.1: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.8c141a.1: %FDifferentImplicitParamType.as.J.impl.F.type.2a4475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.649: %J.type = facet_value %FDifferentImplicitParamType, (%J.impl_witness.a56) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.053: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.649) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.f0d: %J.WithSelf.F.type.053 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.2a4475.2: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.8c141a.2: %FDifferentImplicitParamType.as.J.impl.F.type.2a4475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.c5b: <witness> = impl_witness @FDifferentReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %.cf5: Core.Form = init_form %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.c2a: type = pattern_type %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.f060c0.1: type = fn_type @FDifferentReturnType.as.J.impl.F.loc200_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.30100a.1: %FDifferentReturnType.as.J.impl.F.type.f060c0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.37d: %J.type = facet_value %FDifferentReturnType, (%J.impl_witness.c5b) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.efc: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.37d) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.e15: %J.WithSelf.F.type.efc = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.f060c0.2: type = fn_type @FDifferentReturnType.as.J.impl.F.loc200_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.30100a.2: %FDifferentReturnType.as.J.impl.F.type.f060c0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.f72: %SelfNested.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.b0a: type = facet_access_type %Self.f72 [symbolic]
// CHECK:STDOUT:   %ptr.9b4: type = ptr_type %Self.as_type.b0a [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.1c2: type = struct_type {.x: %Self.as_type.b0a, .y: %i32} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.f8b: %tuple.type.24b = tuple_value (%ptr.9b4, %struct_type.x.y.1c2) [symbolic]
// CHECK:STDOUT:   %tuple.type.73c: type = tuple_type (%ptr.9b4, %struct_type.x.y.1c2) [symbolic]
// CHECK:STDOUT:   %pattern_type.695: type = pattern_type %tuple.type.73c [symbolic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.a3c: type = array_type %int_4, %Self.as_type.b0a [symbolic]
// CHECK:STDOUT:   %.21e: Core.Form = init_form %array_type.a3c [symbolic]
// CHECK:STDOUT:   %pattern_type.f41: type = pattern_type %array_type.a3c [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.a41: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%Self.f72) [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.004: %SelfNested.WithSelf.F.type.a41 = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.45a: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.026: <witness> = impl_witness @SelfNestedBadParam.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.bda: type = ptr_type %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %struct_type.x.y.88a: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.78d: %tuple.type.24b = tuple_value (%ptr.bda, %struct_type.x.y.88a) [concrete]
// CHECK:STDOUT:   %tuple.type.a2b: type = tuple_type (%ptr.bda, %struct_type.x.y.88a) [concrete]
// CHECK:STDOUT:   %pattern_type.a53: type = pattern_type %tuple.type.a2b [concrete]
// CHECK:STDOUT:   %array_type.3ee: type = array_type %int_4, %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %.de3: Core.Form = init_form %array_type.3ee [concrete]
// CHECK:STDOUT:   %pattern_type.072: type = pattern_type %array_type.3ee [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.c055d7.1: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.c20797.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.c055d7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.693: %SelfNested.type = facet_value %SelfNestedBadParam, (%SelfNested.impl_witness.026) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.4d2: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.693) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.b03: %SelfNested.WithSelf.F.type.4d2 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.ed7: type = struct_type {.x: %SelfNestedBadParam, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.be1: %tuple.type.24b = tuple_value (%ptr.bda, %struct_type.x.y.ed7) [concrete]
// CHECK:STDOUT:   %tuple.type.94e: type = tuple_type (%ptr.bda, %struct_type.x.y.ed7) [concrete]
// CHECK:STDOUT:   %pattern_type.a38: type = pattern_type %tuple.type.94e [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.c055d7.2: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.c20797.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.c055d7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.e62: <witness> = impl_witness @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.538: type = ptr_type %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %struct_type.x.y.e7e: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.a5a: %tuple.type.24b = tuple_value (%ptr.538, %struct_type.x.y.e7e) [concrete]
// CHECK:STDOUT:   %tuple.type.4db: type = tuple_type (%ptr.538, %struct_type.x.y.e7e) [concrete]
// CHECK:STDOUT:   %pattern_type.5f7: type = pattern_type %tuple.type.4db [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.6a241d.1: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.18a3b3.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.6a241d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.a9c: %SelfNested.type = facet_value %SelfNestedBadReturnType, (%SelfNested.impl_witness.e62) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.7c7: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.a9c) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.5d2: %SelfNested.WithSelf.F.type.7c7 = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.db9: type = array_type %int_4, %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %.0a9: Core.Form = init_form %array_type.db9 [concrete]
// CHECK:STDOUT:   %pattern_type.083: type = pattern_type %array_type.db9 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.6a241d.2: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.18a3b3.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.6a241d.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.cd6]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.831) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.ddf)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.918]
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
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c93]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.WithSelf.%J.WithSelf.F.type (%J.WithSelf.F.type.66e) = fn_decl @J.WithSelf.F [symbolic = @J.WithSelf.%J.WithSelf.F (constants.%J.WithSelf.F.876)] {
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
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.WithSelf.F.decl [concrete = constants.%assoc0.7d2]
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
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f72]
// CHECK:STDOUT:   %SelfNested.WithSelf.decl = interface_with_self_decl @SelfNested [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %SelfNested.WithSelf.F.decl: @SelfNested.WithSelf.%SelfNested.WithSelf.F.type (%SelfNested.WithSelf.F.type.a41) = fn_decl @SelfNested.WithSelf.F [symbolic = @SelfNested.WithSelf.%SelfNested.WithSelf.F (constants.%SelfNested.WithSelf.F.004)] {
// CHECK:STDOUT:     %x.param_patt: @SelfNested.WithSelf.F.%pattern_type.loc211_9 (%pattern_type.695) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @SelfNested.WithSelf.F.%pattern_type.loc211_9 (%pattern_type.695) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @SelfNested.WithSelf.F.%pattern_type.loc211_57 (%pattern_type.f41) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @SelfNested.WithSelf.F.%pattern_type.loc211_57 (%pattern_type.f41) = return_slot_pattern %return.param_patt, %array_type.loc211_57.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc211_50: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.f72)]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Self.as_type.loc211_50: type = facet_access_type %Self.ref.loc211_50 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b0a)]
// CHECK:STDOUT:     %.loc211_50: type = converted %Self.ref.loc211_50, %Self.as_type.loc211_50 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b0a)]
// CHECK:STDOUT:     %array_type.loc211_57.2: type = array_type %int_4, %.loc211_50 [symbolic = %array_type.loc211_57.1 (constants.%array_type.a3c)]
// CHECK:STDOUT:     %.loc211_57.2: Core.Form = init_form %array_type.loc211_57.2 [symbolic = %.loc211_57.1 (constants.%.21e)]
// CHECK:STDOUT:     %x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.73c) = value_param call_param0
// CHECK:STDOUT:     %.loc211_38.1: type = splice_block %.loc211_38.3 [symbolic = %tuple.type (constants.%tuple.type.73c)] {
// CHECK:STDOUT:       %Self.ref.loc211_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.f72)]
// CHECK:STDOUT:       %Self.as_type.loc211_16.2: type = facet_access_type %Self.ref.loc211_12 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b0a)]
// CHECK:STDOUT:       %.loc211_16: type = converted %Self.ref.loc211_12, %Self.as_type.loc211_16.2 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b0a)]
// CHECK:STDOUT:       %ptr.loc211_16.2: type = ptr_type %.loc211_16 [symbolic = %ptr.loc211_16.1 (constants.%ptr.9b4)]
// CHECK:STDOUT:       %Self.ref.loc211_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.f72)]
// CHECK:STDOUT:       %Self.as_type.loc211_24: type = facet_access_type %Self.ref.loc211_24 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b0a)]
// CHECK:STDOUT:       %.loc211_24: type = converted %Self.ref.loc211_24, %Self.as_type.loc211_24 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b0a)]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y.loc211_37.2: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc211_16.1 (%Self.as_type.b0a), .y: %i32} [symbolic = %struct_type.x.y.loc211_37.1 (constants.%struct_type.x.y.1c2)]
// CHECK:STDOUT:       %.loc211_38.2: %tuple.type.24b = tuple_literal (%ptr.loc211_16.2, %struct_type.x.y.loc211_37.2) [symbolic = %tuple (constants.%tuple.f8b)]
// CHECK:STDOUT:       %.loc211_38.3: type = converted %.loc211_38.2, constants.%tuple.type.73c [symbolic = %tuple.type (constants.%tuple.type.73c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.73c) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @SelfNested.WithSelf.F.%array_type.loc211_57.1 (%array_type.a3c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @SelfNested.WithSelf.F.%array_type.loc211_57.1 (%array_type.a3c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %SelfNested.WithSelf.F.decl [concrete = constants.%assoc0.45a]
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.89a]
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.377]
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.1ac]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .PossiblyF = <poisoned>
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.1: %FExtraParam.as.I.impl.F.type.cb12e5.1 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.989bc3.1] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc69: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.2: %FExtraParam.as.I.impl.F.type.cb12e5.2 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.2 [concrete = constants.%FExtraParam.as.I.impl.F.989bc3.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraParam.as.I.impl.F.decl.loc69_18.2), @FExtraParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.efa]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraParam.as.I.impl.F.decl.loc69_18.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraImplicitParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1: %FExtraImplicitParam.as.I.impl.F.type.f5adc6.1 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_23.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.8df3a7.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ee = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7ee = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self: %FExtraImplicitParam = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.2: %FExtraImplicitParam.as.I.impl.F.type.f5adc6.2 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_23.2 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.8df3a7.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.2), @FExtraImplicitParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.303]
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.0f7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraReturnType.as.I.impl.F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_31.1: %FMissingParam.as.J.impl.F.type.5ef2ac.1 = fn_decl @FMissingParam.as.J.impl.F.loc117_31.1 [concrete = constants.%FMissingParam.as.J.impl.F.d8edee.1] {
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.4dc) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.4dc) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_31.2: %FMissingParam.as.J.impl.F.type.5ef2ac.2 = fn_decl @FMissingParam.as.J.impl.F.loc117_31.2 [concrete = constants.%FMissingParam.as.J.impl.F.d8edee.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.e6f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingParam.as.J.impl.F.decl.loc117_31.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingImplicitParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1: %FMissingImplicitParam.as.J.impl.F.type.f8427f.1 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.b398ca.1] {
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.158) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.158) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.2: %FMissingImplicitParam.as.J.impl.F.type.f8427f.2 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.b398ca.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.e96]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc152_30.1: %FMissingReturnType.as.J.impl.F.type.78fab6.1 = fn_decl @FMissingReturnType.as.J.impl.F.loc152_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.55cda9.1] {
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.2d3) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.2d3) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc152_30.2: %FMissingReturnType.as.J.impl.F.type.78fab6.2 = fn_decl @FMissingReturnType.as.J.impl.F.loc152_30.2 [concrete = constants.%FMissingReturnType.as.J.impl.F.55cda9.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.3a9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingReturnType.as.J.impl.F.decl.loc152_30.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc171_38.1: %FDifferentParamType.as.J.impl.F.type.a9aa82.1 = fn_decl @FDifferentParamType.as.J.impl.F.loc171_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.9d056e.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.789 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.789 = at_binding_pattern b, %b.param_patt [concrete]
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.e11) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.e11) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc171_38.2: %FDifferentParamType.as.J.impl.F.type.a9aa82.2 = fn_decl @FDifferentParamType.as.J.impl.F.loc171_38.2 [concrete = constants.%FDifferentParamType.as.J.impl.F.9d056e.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.eeb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentParamType.as.J.impl.F.decl.loc171_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentImplicitParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1: %FDifferentImplicitParamType.as.J.impl.F.type.2a4475.1 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.8c141a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5a0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.5a0 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.649) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.649) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.2: %FDifferentImplicitParamType.as.J.impl.F.type.2a4475.2 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.8c141a.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.a56]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc200_38.1: %FDifferentReturnType.as.J.impl.F.type.f060c0.1 = fn_decl @FDifferentReturnType.as.J.impl.F.loc200_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.30100a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c2a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c2a = return_slot_pattern %return.param_patt, %Self.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %.loc200_34: Core.Form = init_form %Self.ref [concrete = constants.%.cf5]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc200_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = value_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc200_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param call_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.37d) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: type = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.37d) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc200_38.2: %FDifferentReturnType.as.J.impl.F.type.f060c0.2 = fn_decl @FDifferentReturnType.as.J.impl.F.loc200_38.2 [concrete = constants.%FDifferentReturnType.as.J.impl.F.30100a.2] {
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
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.c5b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentReturnType.as.J.impl.F.decl.loc200_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadParam.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.c055d7.1 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.c20797.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a53 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.a53 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.072 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.072 = return_slot_pattern %return.param_patt, %array_type [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc223_65: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref.loc223_65 [concrete = constants.%array_type.3ee]
// CHECK:STDOUT:     %.loc223_86: Core.Form = init_form %array_type [concrete = constants.%.de3]
// CHECK:STDOUT:     %x.param: %tuple.type.a2b = value_param call_param0
// CHECK:STDOUT:     %.loc223_53.1: type = splice_block %.loc223_53.3 [concrete = constants.%tuple.type.a2b] {
// CHECK:STDOUT:       %SelfNestedBadParam.ref.loc223_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadParam.ref.loc223_14 [concrete = constants.%ptr.bda]
// CHECK:STDOUT:       %i32.loc223_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc223_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete = constants.%struct_type.x.y.88a]
// CHECK:STDOUT:       %.loc223_53.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.78d]
// CHECK:STDOUT:       %.loc223_53.3: type = converted %.loc223_53.2, constants.%tuple.type.a2b [concrete = constants.%tuple.type.a2b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a2b = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.3ee = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.3ee = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc211_57.1: type = specific_constant @SelfNested.WithSelf.F.%array_type.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.693) [concrete = constants.%array_type.3ee]
// CHECK:STDOUT:   %.loc211_57.2: type = specific_constant @SelfNested.WithSelf.F.%.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.693) [concrete = constants.%.de3]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.c055d7.2 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.c20797.2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a38 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.a38 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.072 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.072 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.94e = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.94e = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.3ee = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.3ee = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.2), @SelfNestedBadParam.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.026]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1
// CHECK:STDOUT:   extend %SelfNested.ref
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadReturnType.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.6a241d.1 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.18a3b3.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.5f7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.5f7 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.072 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.072 = return_slot_pattern %return.param_patt, %array_type [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref [concrete = constants.%array_type.3ee]
// CHECK:STDOUT:     %.loc239_111: Core.Form = init_form %array_type [concrete = constants.%.de3]
// CHECK:STDOUT:     %x.param: %tuple.type.4db = value_param call_param0
// CHECK:STDOUT:     %.loc239_78.1: type = splice_block %.loc239_78.3 [concrete = constants.%tuple.type.4db] {
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc239_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadReturnType.ref.loc239_14 [concrete = constants.%ptr.538]
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc239_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete = constants.%struct_type.x.y.e7e]
// CHECK:STDOUT:       %.loc239_78.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.a5a]
// CHECK:STDOUT:       %.loc239_78.3: type = converted %.loc239_78.2, constants.%tuple.type.4db [concrete = constants.%tuple.type.4db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.4db = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.3ee = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.3ee = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc211_57.1: type = specific_constant @SelfNested.WithSelf.F.%array_type.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.a9c) [concrete = constants.%array_type.db9]
// CHECK:STDOUT:   %.loc211_57.2: type = specific_constant @SelfNested.WithSelf.F.%.loc211_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.a9c) [concrete = constants.%.0a9]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.6a241d.2 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.18a3b3.2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.5f7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.5f7 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.083 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.083 = return_slot_pattern %return.param_patt, invalid [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.4db = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.4db = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.db9 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.db9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.2), @SelfNestedBadReturnType.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.e62]
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
// CHECK:STDOUT: fn @FExtraParam.as.I.impl.F.loc69_18.2() [thunk @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.50c)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraParam.as.I.impl.F.type.cb12e5.1 = name_ref F, @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.989bc3.1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_23.1(%self.param: %FExtraImplicitParam);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_23.2() [thunk @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.f28)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraImplicitParam.as.I.impl.F.type.f5adc6.1 = name_ref F, @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.8df3a7.1]
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
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc117_31.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_31.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.4dc)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingParam.as.J.impl.F.type.5ef2ac.1 = name_ref F, @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_31.1 [concrete = constants.%FMissingParam.as.J.impl.F.d8edee.1]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.1(%b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.158)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingImplicitParam.as.J.impl.F.type.f8427f.1 = name_ref F, @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.b398ca.1]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.call: init bool = call %F.ref(%b.param)
// CHECK:STDOUT:   return %FMissingImplicitParam.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc152_30.1(%self.param: bool, %b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc152_30.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc152_30.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.2d3)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingReturnType.as.J.impl.F.type.78fab6.1 = name_ref F, @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc152_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.55cda9.1]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.call: init %empty_tuple.type = call %FMissingReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc152: bool = converted %FMissingReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc171_38.1(%self.param: bool, %b.param: %FDifferentParamType) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc171_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc171_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.e11)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentParamType.as.J.impl.F.type.a9aa82.1 = name_ref F, @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc171_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.9d056e.1]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc103: %FDifferentParamType = converted %b.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.call: init bool = call %FDifferentParamType.as.J.impl.F.bound(%self.param, <error>)
// CHECK:STDOUT:   return %FDifferentParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1(%self.param: %FDifferentImplicitParamType, %b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.649)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentImplicitParamType.as.J.impl.F.type.2a4475.1 = name_ref F, @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.8c141a.1]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc103: %FDifferentImplicitParamType = converted %self.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.call: init bool = call %FDifferentImplicitParamType.as.J.impl.F.bound(<error>, %b.param)
// CHECK:STDOUT:   return %FDifferentImplicitParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc200_38.1(%self.param: bool, %b.param: bool) -> out %return.param: %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc200_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc200_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.37d)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentReturnType.as.J.impl.F.type.f060c0.1 = name_ref F, @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc200_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.30100a.1]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc200_38.1: ref %FDifferentReturnType = temporary_storage
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.call: init %FDifferentReturnType to %.loc200_38.1 = call %FDifferentReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc200_38.2: bool = converted %FDifferentReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SelfNested.WithSelf.F(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.f72)]
// CHECK:STDOUT:   %Self.as_type.loc211_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.b0a)]
// CHECK:STDOUT:   %ptr.loc211_16.1: type = ptr_type %Self.as_type.loc211_16.1 [symbolic = %ptr.loc211_16.1 (constants.%ptr.9b4)]
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc211_16.1 (%Self.as_type.b0a), .y: %i32} [symbolic = %struct_type.x.y.loc211_37.1 (constants.%struct_type.x.y.1c2)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc211_16.1, %struct_type.x.y.loc211_37.1) [symbolic = %tuple (constants.%tuple.f8b)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc211_16.1, %struct_type.x.y.loc211_37.1) [symbolic = %tuple.type (constants.%tuple.type.73c)]
// CHECK:STDOUT:   %pattern_type.loc211_9: type = pattern_type %tuple.type [symbolic = %pattern_type.loc211_9 (constants.%pattern_type.695)]
// CHECK:STDOUT:   %array_type.loc211_57.1: type = array_type constants.%int_4, %Self.as_type.loc211_16.1 [symbolic = %array_type.loc211_57.1 (constants.%array_type.a3c)]
// CHECK:STDOUT:   %.loc211_57.1: Core.Form = init_form %array_type.loc211_57.1 [symbolic = %.loc211_57.1 (constants.%.21e)]
// CHECK:STDOUT:   %pattern_type.loc211_57: type = pattern_type %array_type.loc211_57.1 [symbolic = %pattern_type.loc211_57 (constants.%pattern_type.f41)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.73c)) -> out %return.param: @SelfNested.WithSelf.F.%array_type.loc211_57.1 (%array_type.a3c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1(%x.param: %tuple.type.a2b) -> out %return.param: %array_type.3ee;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2(%x.param: %tuple.type.94e) -> out %return.param: %array_type.3ee [thunk @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.693)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadParam.as.SelfNested.impl.F.type.c055d7.1 = name_ref F, @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.c20797.1]
// CHECK:STDOUT:   %.loc211_57: ref %array_type.3ee = splice_block %return.param {}
// CHECK:STDOUT:   %tuple.elem0: %ptr.bda = tuple_access %x.param, element0
// CHECK:STDOUT:   %tuple.elem1: %struct_type.x.y.ed7 = tuple_access %x.param, element1
// CHECK:STDOUT:   %.loc211_9.1: %SelfNestedBadParam = struct_access %tuple.elem1, element0
// CHECK:STDOUT:   %.loc211_9.2: %i32 = converted %.loc211_9.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.call: init %array_type.3ee to %.loc211_57 = call %F.ref(<error>)
// CHECK:STDOUT:   return %SelfNestedBadParam.as.SelfNested.impl.F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1(%x.param: %tuple.type.4db) -> out %return.param: %array_type.3ee;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2(%x.param: %tuple.type.4db) -> out %return.param: %array_type.db9 [thunk @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.a9c)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.6a241d.1 = name_ref F, @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.18a3b3.1]
// CHECK:STDOUT:   %.loc239_112.1: ref %array_type.3ee = temporary_storage
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.call: init %array_type.3ee to %.loc239_112.1 = call %F.ref(%x.param)
// CHECK:STDOUT:   %.loc239_112.2: %array_type.db9 = converted %SelfNestedBadReturnType.as.SelfNested.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.cd6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.cd6
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.831
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ddf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.cd6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.469) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.469
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.9ea
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.528
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.1ce) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.1ce
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.a4c
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.2f2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.0a8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.0a8
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.c54
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.c60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.50c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.50c
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.e8f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.df3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.50c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.f28) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.f28
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.ff9
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.e60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.f28) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.d6e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.d6e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.37b
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.3ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.d6e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.c93) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.c93
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.66e
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.876
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.c93) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.4dc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.4dc
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.0f0
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.4dc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.158) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.158
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.72a
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.5e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.158) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.2d3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.2d3
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.d1d
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.eba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.2d3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.e11) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.e11
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.0de
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.d73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.e11) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.649) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.649
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.053
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.f0d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.649) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.37d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.37d
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.efc
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.e15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.37d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%Self.f72) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.f72
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.a41
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.004
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%Self.f72) {
// CHECK:STDOUT:   %Self => constants.%Self.f72
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%Self.as_type.b0a
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.9b4
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.1c2
// CHECK:STDOUT:   %tuple => constants.%tuple.f8b
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.73c
// CHECK:STDOUT:   %pattern_type.loc211_9 => constants.%pattern_type.695
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.a3c
// CHECK:STDOUT:   %.loc211_57.1 => constants.%.21e
// CHECK:STDOUT:   %pattern_type.loc211_57 => constants.%pattern_type.f41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.693) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.693
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.4d2
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.b03
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.693) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.693
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.bda
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.ed7
// CHECK:STDOUT:   %tuple => constants.%tuple.be1
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.94e
// CHECK:STDOUT:   %pattern_type.loc211_9 => constants.%pattern_type.a38
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.3ee
// CHECK:STDOUT:   %.loc211_57.1 => constants.%.de3
// CHECK:STDOUT:   %pattern_type.loc211_57 => constants.%pattern_type.072
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.a9c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.a9c
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.7c7
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.5d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.a9c) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.a9c
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.538
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.e7e
// CHECK:STDOUT:   %tuple => constants.%tuple.a5a
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.4db
// CHECK:STDOUT:   %pattern_type.loc211_9 => constants.%pattern_type.5f7
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.db9
// CHECK:STDOUT:   %.loc211_57.1 => constants.%.0a9
// CHECK:STDOUT:   %pattern_type.loc211_57 => constants.%pattern_type.083
// CHECK:STDOUT: }
// CHECK:STDOUT:
