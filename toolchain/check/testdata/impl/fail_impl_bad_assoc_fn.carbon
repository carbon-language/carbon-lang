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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+13]]:5: note: type `()` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:     fn F[self: bool](b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-39]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-43]]:32: error: cannot implicitly convert expression of type `bool` to `FDifferentParamType` [ConversionFailure]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-46]]:32: note: type `bool` does not implement interface `Core.ImplicitAs(FDifferentParamType)` [MissingImplInMemberAccessNote]
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-65]]:20: note: type `bool` does not implement interface `Core.ImplicitAs(FDifferentImplicitParamType)` [MissingImplInMemberAccessNote]
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: type `FDifferentReturnType` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+3]]:8: note: type `SelfNestedBadParam` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: type `array(SelfNestedBadParam, 4)` does not implement interface `Core.ImplicitAs(array(SelfNestedBadReturnType, 4))` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.F: %I.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%I.F.decl [concrete]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [concrete]
// CHECK:STDOUT:   %I.impl_witness.901: <witness> = impl_witness @NoF.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.9b3: <witness> = impl_witness @NoF.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.0f9: type = ptr_type %NoF [concrete]
// CHECK:STDOUT:   %pattern_type.7b5: type = pattern_type %ptr.0f9 [concrete]
// CHECK:STDOUT:   %NoF.as.Destroy.impl.Op.type: type = fn_type @NoF.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %NoF.as.Destroy.impl.Op: %NoF.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [concrete]
// CHECK:STDOUT:   %I.impl_witness.44e: <witness> = impl_witness @FNotFunction.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %F: type = class_type @F [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.f9b: <witness> = impl_witness @FNotFunction.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.2b2: type = ptr_type %FNotFunction [concrete]
// CHECK:STDOUT:   %pattern_type.27c: type = pattern_type %ptr.2b2 [concrete]
// CHECK:STDOUT:   %FNotFunction.as.Destroy.impl.Op.type: type = fn_type @FNotFunction.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FNotFunction.as.Destroy.impl.Op: %FNotFunction.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %PossiblyF.type: type = fn_type @PossiblyF [concrete]
// CHECK:STDOUT:   %PossiblyF: %PossiblyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [concrete]
// CHECK:STDOUT:   %I.impl_witness.c80: <witness> = impl_witness @FAlias.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.740: <witness> = impl_witness @FAlias.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.4bf: type = ptr_type %FAlias [concrete]
// CHECK:STDOUT:   %pattern_type.310: type = pattern_type %ptr.4bf [concrete]
// CHECK:STDOUT:   %FAlias.as.Destroy.impl.Op.type: type = fn_type @FAlias.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FAlias.as.Destroy.impl.Op: %FAlias.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.1f3: <witness> = impl_witness @FExtraParam.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.44ef8c.1: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.1 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.424e9e.1: %FExtraParam.as.I.impl.F.type.44ef8c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.fa4: %I.type = facet_value %FExtraParam, (%I.impl_witness.1f3) [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.44ef8c.2: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.2 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.424e9e.2: %FExtraParam.as.I.impl.F.type.44ef8c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.44a: <witness> = impl_witness @FExtraParam.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.0f2: type = ptr_type %FExtraParam [concrete]
// CHECK:STDOUT:   %pattern_type.8d3: type = pattern_type %ptr.0f2 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.Destroy.impl.Op.type: type = fn_type @FExtraParam.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FExtraParam.as.Destroy.impl.Op: %FExtraParam.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.a7b: <witness> = impl_witness @FExtraImplicitParam.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.8ae: type = pattern_type %FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.e1abdd.1: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_23.1 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.6ff574.1: %FExtraImplicitParam.as.I.impl.F.type.e1abdd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.ee5: %I.type = facet_value %FExtraImplicitParam, (%I.impl_witness.a7b) [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.e1abdd.2: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_23.2 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.6ff574.2: %FExtraImplicitParam.as.I.impl.F.type.e1abdd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.a05: <witness> = impl_witness @FExtraImplicitParam.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.8c7: type = ptr_type %FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %pattern_type.ab8: type = pattern_type %ptr.8c7 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.Destroy.impl.Op.type: type = fn_type @FExtraImplicitParam.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.Destroy.impl.Op: %FExtraImplicitParam.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [concrete]
// CHECK:STDOUT:   %I.impl_witness.59a: <witness> = impl_witness @FExtraReturnType.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.type: type = fn_type @FExtraReturnType.as.I.impl.F [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F: %FExtraReturnType.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.d5c: %I.type = facet_value %FExtraReturnType, (%I.impl_witness.59a) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.3ea: <witness> = impl_witness @FExtraReturnType.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.147: type = ptr_type %FExtraReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.f1c: type = pattern_type %ptr.147 [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.Destroy.impl.Op.type: type = fn_type @FExtraReturnType.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.Destroy.impl.Op: %FExtraReturnType.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.F.type: type = fn_type @J.F [concrete]
// CHECK:STDOUT:   %J.F: %J.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.922: %J.assoc_type = assoc_entity element0, @J.%J.F.decl [concrete]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.4e9: <witness> = impl_witness @FMissingParam.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.69596c.1: type = fn_type @FMissingParam.as.J.impl.F.loc117_31.1 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.738f31.1: %FMissingParam.as.J.impl.F.type.69596c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.94c: %J.type = facet_value %FMissingParam, (%J.impl_witness.4e9) [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.69596c.2: type = fn_type @FMissingParam.as.J.impl.F.loc117_31.2 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.738f31.2: %FMissingParam.as.J.impl.F.type.69596c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.672: <witness> = impl_witness @FMissingParam.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.fc8: type = ptr_type %FMissingParam [concrete]
// CHECK:STDOUT:   %pattern_type.c8f: type = pattern_type %ptr.fc8 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.Destroy.impl.Op.type: type = fn_type @FMissingParam.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FMissingParam.as.Destroy.impl.Op: %FMissingParam.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.841: <witness> = impl_witness @FMissingImplicitParam.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.d97cef.1: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.01de92.1: %FMissingImplicitParam.as.J.impl.F.type.d97cef.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.e7e: %J.type = facet_value %FMissingImplicitParam, (%J.impl_witness.841) [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.d97cef.2: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.01de92.2: %FMissingImplicitParam.as.J.impl.F.type.d97cef.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ae6: <witness> = impl_witness @FMissingImplicitParam.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.c3c: type = ptr_type %FMissingImplicitParam [concrete]
// CHECK:STDOUT:   %pattern_type.575: type = pattern_type %ptr.c3c [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.Destroy.impl.Op.type: type = fn_type @FMissingImplicitParam.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.Destroy.impl.Op: %FMissingImplicitParam.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.e75: <witness> = impl_witness @FMissingReturnType.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.123d7a.1: type = fn_type @FMissingReturnType.as.J.impl.F.loc152_30.1 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.c7d02d.1: %FMissingReturnType.as.J.impl.F.type.123d7a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.974: %J.type = facet_value %FMissingReturnType, (%J.impl_witness.e75) [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.123d7a.2: type = fn_type @FMissingReturnType.as.J.impl.F.loc152_30.2 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.c7d02d.2: %FMissingReturnType.as.J.impl.F.type.123d7a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.8d8: <witness> = impl_witness @FMissingReturnType.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.41e: type = ptr_type %FMissingReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.585: type = pattern_type %ptr.41e [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.Destroy.impl.Op.type: type = fn_type @FMissingReturnType.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.Destroy.impl.Op: %FMissingReturnType.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.cbe: <witness> = impl_witness @FDifferentParamType.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.b90: type = pattern_type %FDifferentParamType [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.6b537d.1: type = fn_type @FDifferentParamType.as.J.impl.F.loc171_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.04313a.1: %FDifferentParamType.as.J.impl.F.type.6b537d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.37e: %J.type = facet_value %FDifferentParamType, (%J.impl_witness.cbe) [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.6b537d.2: type = fn_type @FDifferentParamType.as.J.impl.F.loc171_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.04313a.2: %FDifferentParamType.as.J.impl.F.type.6b537d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.443: <witness> = impl_witness @FDifferentParamType.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.f1f: type = ptr_type %FDifferentParamType [concrete]
// CHECK:STDOUT:   %pattern_type.e0c: type = pattern_type %ptr.f1f [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.Destroy.impl.Op.type: type = fn_type @FDifferentParamType.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.Destroy.impl.Op: %FDifferentParamType.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.af2: <witness> = impl_witness @FDifferentImplicitParamType.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9cc: type = pattern_type %FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.d6232a.1: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.886f70.1: %FDifferentImplicitParamType.as.J.impl.F.type.d6232a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.bf1: %J.type = facet_value %FDifferentImplicitParamType, (%J.impl_witness.af2) [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.d6232a.2: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.886f70.2: %FDifferentImplicitParamType.as.J.impl.F.type.d6232a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.c2a: <witness> = impl_witness @FDifferentImplicitParamType.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.dc5: type = ptr_type %FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %pattern_type.986: type = pattern_type %ptr.dc5 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.Destroy.impl.Op.type: type = fn_type @FDifferentImplicitParamType.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.Destroy.impl.Op: %FDifferentImplicitParamType.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.bc1: <witness> = impl_witness @FDifferentReturnType.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.8ee: type = pattern_type %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.d3b58f.1: type = fn_type @FDifferentReturnType.as.J.impl.F.loc200_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.be86c9.1: %FDifferentReturnType.as.J.impl.F.type.d3b58f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.67f: %J.type = facet_value %FDifferentReturnType, (%J.impl_witness.bc1) [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.d3b58f.2: type = fn_type @FDifferentReturnType.as.J.impl.F.loc200_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.be86c9.2: %FDifferentReturnType.as.J.impl.F.type.d3b58f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.6d2: <witness> = impl_witness @FDifferentReturnType.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.954: type = ptr_type %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.246: type = pattern_type %ptr.954 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.Destroy.impl.Op.type: type = fn_type @FDifferentReturnType.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.Destroy.impl.Op: %FDifferentReturnType.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.2ff: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.e1e: type = facet_access_type %Self.2ff [symbolic]
// CHECK:STDOUT:   %ptr.e87: type = ptr_type %Self.as_type.e1e [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.81e: type = struct_type {.x: %Self.as_type.e1e, .y: %i32} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.229: type = tuple_type (%ptr.e87, %struct_type.x.y.81e) [symbolic]
// CHECK:STDOUT:   %pattern_type.ced: type = pattern_type %tuple.type.229 [symbolic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.873: type = array_type %int_4, %Self.as_type.e1e [symbolic]
// CHECK:STDOUT:   %pattern_type.c7e: type = pattern_type %array_type.873 [symbolic]
// CHECK:STDOUT:   %SelfNested.F.type: type = fn_type @SelfNested.F [concrete]
// CHECK:STDOUT:   %SelfNested.F: %SelfNested.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.beb: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.%SelfNested.F.decl [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.e31: <witness> = impl_witness @SelfNestedBadParam.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.4cd: type = ptr_type %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %struct_type.x.y.871: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.a7d: type = tuple_type (%ptr.4cd, %struct_type.x.y.871) [concrete]
// CHECK:STDOUT:   %pattern_type.714: type = pattern_type %tuple.type.a7d [concrete]
// CHECK:STDOUT:   %array_type.a41: type = array_type %int_4, %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %pattern_type.1f1: type = pattern_type %array_type.a41 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.f90165.1: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.fa8d04.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.f90165.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.608: %SelfNested.type = facet_value %SelfNestedBadParam, (%SelfNested.impl_witness.e31) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.a89: type = struct_type {.x: %SelfNestedBadParam, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.9c9: type = tuple_type (%ptr.4cd, %struct_type.x.y.a89) [concrete]
// CHECK:STDOUT:   %pattern_type.a5c: type = pattern_type %tuple.type.9c9 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.f90165.2: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.fa8d04.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.f90165.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.908: <witness> = impl_witness @SelfNestedBadParam.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.2a7: type = pattern_type %ptr.4cd [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.Destroy.impl.Op.type: type = fn_type @SelfNestedBadParam.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.Destroy.impl.Op: %SelfNestedBadParam.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f40: type = ptr_type %array_type.a41 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.ba3: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%array_type.a41) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.d25: %T.as.Destroy.impl.Op.type.ba3 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.d25, @T.as.Destroy.impl.Op(%array_type.a41) [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.d5e: <witness> = impl_witness @SelfNestedBadReturnType.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.612: type = ptr_type %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %struct_type.x.y.ac5: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.eb9: type = tuple_type (%ptr.612, %struct_type.x.y.ac5) [concrete]
// CHECK:STDOUT:   %pattern_type.23f: type = pattern_type %tuple.type.eb9 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.0e7d1d.1: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.0bc78a.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.0e7d1d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.766: %SelfNested.type = facet_value %SelfNestedBadReturnType, (%SelfNested.impl_witness.d5e) [concrete]
// CHECK:STDOUT:   %array_type.126: type = array_type %int_4, %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.f56: type = pattern_type %array_type.126 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.0e7d1d.2: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.0bc78a.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.0e7d1d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.9f5: <witness> = impl_witness @SelfNestedBadReturnType.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4de: type = pattern_type %ptr.612 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.Destroy.impl.Op.type: type = fn_type @SelfNestedBadReturnType.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.Destroy.impl.Op: %SelfNestedBadReturnType.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %I.F.decl: %I.F.type = fn_decl @I.F [concrete = constants.%I.F] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.F.decl [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%I.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %J.F.decl: %J.F.type = fn_decl @J.F [concrete = constants.%J.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call.loc103_44: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc103_44.1: type = value_of_initializer %Bool.call.loc103_44 [concrete = bool]
// CHECK:STDOUT:     %.loc103_44.2: type = converted %Bool.call.loc103_44, %.loc103_44.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc103_26.1: type = splice_block %.loc103_26.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc103_26: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc103_26.2: type = value_of_initializer %Bool.call.loc103_26 [concrete = bool]
// CHECK:STDOUT:       %.loc103_26.3: type = converted %Bool.call.loc103_26, %.loc103_26.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc103_35.1: type = splice_block %.loc103_35.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc103_35: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc103_35.2: type = value_of_initializer %Bool.call.loc103_35 [concrete = bool]
// CHECK:STDOUT:       %.loc103_35.3: type = converted %Bool.call.loc103_35, %.loc103_35.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.F.decl [concrete = constants.%assoc0.922]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%J.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2ff]
// CHECK:STDOUT:   %SelfNested.F.decl: %SelfNested.F.type = fn_decl @SelfNested.F [concrete = constants.%SelfNested.F] {
// CHECK:STDOUT:     %x.patt: @SelfNested.F.%pattern_type.loc211_8 (%pattern_type.ced) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @SelfNested.F.%pattern_type.loc211_8 (%pattern_type.ced) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @SelfNested.F.%pattern_type.loc211_41 (%pattern_type.c7e) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @SelfNested.F.%pattern_type.loc211_41 (%pattern_type.c7e) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc211_50: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Self.as_type.loc211_50: type = facet_access_type %Self.ref.loc211_50 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:     %.loc211_50: type = converted %Self.ref.loc211_50, %Self.as_type.loc211_50 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:     %array_type.loc211_57.2: type = array_type %int_4, %.loc211_50 [symbolic = %array_type.loc211_57.1 (constants.%array_type.873)]
// CHECK:STDOUT:     %x.param: @SelfNested.F.%tuple.type (%tuple.type.229) = value_param call_param0
// CHECK:STDOUT:     %.loc211_38.1: type = splice_block %.loc211_38.3 [symbolic = %tuple.type (constants.%tuple.type.229)] {
// CHECK:STDOUT:       %Self.ref.loc211_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc211_16.2: type = facet_access_type %Self.ref.loc211_12 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc211_16: type = converted %Self.ref.loc211_12, %Self.as_type.loc211_16.2 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %ptr.loc211_16.2: type = ptr_type %.loc211_16 [symbolic = %ptr.loc211_16.1 (constants.%ptr.e87)]
// CHECK:STDOUT:       %Self.ref.loc211_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc211_24: type = facet_access_type %Self.ref.loc211_24 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc211_24: type = converted %Self.ref.loc211_24, %Self.as_type.loc211_24 [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y.loc211_37.2: type = struct_type {.x: @SelfNested.F.%Self.as_type.loc211_16.1 (%Self.as_type.e1e), .y: %i32} [symbolic = %struct_type.x.y.loc211_37.1 (constants.%struct_type.x.y.81e)]
// CHECK:STDOUT:       %.loc211_38.2: %tuple.type.24b = tuple_literal (%ptr.loc211_16.2, %struct_type.x.y.loc211_37.2)
// CHECK:STDOUT:       %.loc211_38.3: type = converted %.loc211_38.2, constants.%tuple.type.229 [symbolic = %tuple.type (constants.%tuple.type.229)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SelfNested.F.%tuple.type (%tuple.type.229) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @SelfNested.F.%array_type.loc211_57.1 (%array_type.873) = out_param call_param1
// CHECK:STDOUT:     %return: ref @SelfNested.F.%array_type.loc211_57.1 (%array_type.873) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %SelfNested.F.decl [concrete = constants.%assoc0.beb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%SelfNested.F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @NoF.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   witness = @NoF.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @NoF.as.Destroy.impl: constants.%NoF as constants.%Destroy.type {
// CHECK:STDOUT:   %NoF.as.Destroy.impl.Op.decl: %NoF.as.Destroy.impl.Op.type = fn_decl @NoF.as.Destroy.impl.Op [concrete = constants.%NoF.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.7b5 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7b5 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc17: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.0f9 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoF [concrete = constants.%NoF]
// CHECK:STDOUT:     %self: %ptr.0f9 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %NoF.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @NoF.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FNotFunction.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: type = class_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FNotFunction.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FNotFunction.as.Destroy.impl: constants.%FNotFunction as constants.%Destroy.type {
// CHECK:STDOUT:   %FNotFunction.as.Destroy.impl.Op.decl: %FNotFunction.as.Destroy.impl.Op.type = fn_decl @FNotFunction.as.Destroy.impl.Op [concrete = constants.%FNotFunction.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.27c = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.27c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc28: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.2b2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FNotFunction [concrete = constants.%FNotFunction]
// CHECK:STDOUT:     %self: %ptr.2b2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FNotFunction.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FNotFunction.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FAlias.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %PossiblyF.ref: %PossiblyF.type = name_ref PossiblyF, file.%PossiblyF.decl [concrete = constants.%PossiblyF]
// CHECK:STDOUT:   %F: %PossiblyF.type = bind_alias F, file.%PossiblyF.decl [concrete = constants.%PossiblyF]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .PossiblyF = <poisoned>
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = @FAlias.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FAlias.as.Destroy.impl: constants.%FAlias as constants.%Destroy.type {
// CHECK:STDOUT:   %FAlias.as.Destroy.impl.Op.decl: %FAlias.as.Destroy.impl.Op.type = fn_decl @FAlias.as.Destroy.impl.Op [concrete = constants.%FAlias.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.310 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.310 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc44: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.4bf = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FAlias [concrete = constants.%FAlias]
// CHECK:STDOUT:     %self: %ptr.4bf = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FAlias.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FAlias.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.1: %FExtraParam.as.I.impl.F.type.44ef8c.1 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.424e9e.1] {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc69_13.1: type = splice_block %.loc69_13.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc69_13.2: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:       %.loc69_13.3: type = converted %Bool.call, %.loc69_13.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.2: %FExtraParam.as.I.impl.F.type.44ef8c.2 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.2 [concrete = constants.%FExtraParam.as.I.impl.F.424e9e.2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraParam.as.I.impl.F.decl.loc69_18.1
// CHECK:STDOUT:   witness = @FExtraParam.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraParam.as.Destroy.impl: constants.%FExtraParam as constants.%Destroy.type {
// CHECK:STDOUT:   %FExtraParam.as.Destroy.impl.Op.decl: %FExtraParam.as.Destroy.impl.Op.type = fn_decl @FExtraParam.as.Destroy.impl.Op [concrete = constants.%FExtraParam.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.8d3 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8d3 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc57: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.0f2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraParam [concrete = constants.%FExtraParam]
// CHECK:STDOUT:     %self: %ptr.0f2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FExtraParam.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FExtraParam.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraImplicitParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1: %FExtraImplicitParam.as.I.impl.F.type.e1abdd.1 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_23.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.6ff574.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.8ae = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8ae = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self: %FExtraImplicitParam = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.2: %FExtraImplicitParam.as.I.impl.F.type.e1abdd.2 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_23.2 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.6ff574.2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1
// CHECK:STDOUT:   witness = @FExtraImplicitParam.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraImplicitParam.as.Destroy.impl: constants.%FExtraImplicitParam as constants.%Destroy.type {
// CHECK:STDOUT:   %FExtraImplicitParam.as.Destroy.impl.Op.decl: %FExtraImplicitParam.as.Destroy.impl.Op.type = fn_decl @FExtraImplicitParam.as.Destroy.impl.Op [concrete = constants.%FExtraImplicitParam.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.ab8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ab8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc73: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.8c7 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self: %ptr.8c7 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FExtraImplicitParam.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FExtraImplicitParam.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraReturnType.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.decl: %FExtraReturnType.as.I.impl.F.type = fn_decl @FExtraReturnType.as.I.impl.F [concrete = constants.%FExtraReturnType.as.I.impl.F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc99_15.1: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc99_15.2: type = converted %Bool.call, %.loc99_15.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraReturnType.as.I.impl.F.decl
// CHECK:STDOUT:   witness = @FExtraReturnType.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraReturnType.as.Destroy.impl: constants.%FExtraReturnType as constants.%Destroy.type {
// CHECK:STDOUT:   %FExtraReturnType.as.Destroy.impl.Op.decl: %FExtraReturnType.as.Destroy.impl.Op.type = fn_decl @FExtraReturnType.as.Destroy.impl.Op [concrete = constants.%FExtraReturnType.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f1c = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f1c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc90: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.147 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraReturnType [concrete = constants.%FExtraReturnType]
// CHECK:STDOUT:     %self: %ptr.147 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FExtraReturnType.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FExtraReturnType.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_31.1: %FMissingParam.as.J.impl.F.type.69596c.1 = fn_decl @FMissingParam.as.J.impl.F.loc117_31.1 [concrete = constants.%FMissingParam.as.J.impl.F.738f31.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call.loc117_27: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc117_27.1: type = value_of_initializer %Bool.call.loc117_27 [concrete = bool]
// CHECK:STDOUT:     %.loc117_27.2: type = converted %Bool.call.loc117_27, %.loc117_27.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc117_16.1: type = splice_block %.loc117_16.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc117_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc117_16.2: type = value_of_initializer %Bool.call.loc117_16 [concrete = bool]
// CHECK:STDOUT:       %.loc117_16.3: type = converted %Bool.call.loc117_16, %.loc117_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_31.2: %FMissingParam.as.J.impl.F.type.69596c.2 = fn_decl @FMissingParam.as.J.impl.F.loc117_31.2 [concrete = constants.%FMissingParam.as.J.impl.F.738f31.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingParam.as.J.impl.F.decl.loc117_31.1
// CHECK:STDOUT:   witness = @FMissingParam.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingParam.as.Destroy.impl: constants.%FMissingParam as constants.%Destroy.type {
// CHECK:STDOUT:   %FMissingParam.as.Destroy.impl.Op.decl: %FMissingParam.as.Destroy.impl.Op.type = fn_decl @FMissingParam.as.Destroy.impl.Op [concrete = constants.%FMissingParam.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c8f = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c8f = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc105: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.fc8 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingParam [concrete = constants.%FMissingParam]
// CHECK:STDOUT:     %self: %ptr.fc8 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FMissingParam.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FMissingParam.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingImplicitParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1: %FMissingImplicitParam.as.J.impl.F.type.d97cef.1 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.01de92.1] {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call.loc130_22: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc130_22.1: type = value_of_initializer %Bool.call.loc130_22 [concrete = bool]
// CHECK:STDOUT:     %.loc130_22.2: type = converted %Bool.call.loc130_22, %.loc130_22.1 [concrete = bool]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc130_13.1: type = splice_block %.loc130_13.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc130_13: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc130_13.2: type = value_of_initializer %Bool.call.loc130_13 [concrete = bool]
// CHECK:STDOUT:       %.loc130_13.3: type = converted %Bool.call.loc130_13, %.loc130_13.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.2: %FMissingImplicitParam.as.J.impl.F.type.d97cef.2 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.01de92.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1
// CHECK:STDOUT:   witness = @FMissingImplicitParam.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingImplicitParam.as.Destroy.impl: constants.%FMissingImplicitParam as constants.%Destroy.type {
// CHECK:STDOUT:   %FMissingImplicitParam.as.Destroy.impl.Op.decl: %FMissingImplicitParam.as.Destroy.impl.Op.type = fn_decl @FMissingImplicitParam.as.Destroy.impl.Op [concrete = constants.%FMissingImplicitParam.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.575 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.575 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc121: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.c3c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingImplicitParam [concrete = constants.%FMissingImplicitParam]
// CHECK:STDOUT:     %self: %ptr.c3c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FMissingImplicitParam.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FMissingImplicitParam.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc152_30.1: %FMissingReturnType.as.J.impl.F.type.123d7a.1 = fn_decl @FMissingReturnType.as.J.impl.F.loc152_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.c7d02d.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc152_16.1: type = splice_block %.loc152_16.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc152_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc152_16.2: type = value_of_initializer %Bool.call.loc152_16 [concrete = bool]
// CHECK:STDOUT:       %.loc152_16.3: type = converted %Bool.call.loc152_16, %.loc152_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc152_25.1: type = splice_block %.loc152_25.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc152_25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc152_25.2: type = value_of_initializer %Bool.call.loc152_25 [concrete = bool]
// CHECK:STDOUT:       %.loc152_25.3: type = converted %Bool.call.loc152_25, %.loc152_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc152_30.2: %FMissingReturnType.as.J.impl.F.type.123d7a.2 = fn_decl @FMissingReturnType.as.J.impl.F.loc152_30.2 [concrete = constants.%FMissingReturnType.as.J.impl.F.c7d02d.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingReturnType.as.J.impl.F.decl.loc152_30.1
// CHECK:STDOUT:   witness = @FMissingReturnType.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingReturnType.as.Destroy.impl: constants.%FMissingReturnType as constants.%Destroy.type {
// CHECK:STDOUT:   %FMissingReturnType.as.Destroy.impl.Op.decl: %FMissingReturnType.as.Destroy.impl.Op.type = fn_decl @FMissingReturnType.as.Destroy.impl.Op [concrete = constants.%FMissingReturnType.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.585 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.585 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc134: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.41e = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingReturnType [concrete = constants.%FMissingReturnType]
// CHECK:STDOUT:     %self: %ptr.41e = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FMissingReturnType.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FMissingReturnType.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc171_38.1: %FDifferentParamType.as.J.impl.F.type.6b537d.1 = fn_decl @FDifferentParamType.as.J.impl.F.loc171_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.04313a.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.b90 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.b90 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call.loc171_34: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc171_34.1: type = value_of_initializer %Bool.call.loc171_34 [concrete = bool]
// CHECK:STDOUT:     %.loc171_34.2: type = converted %Bool.call.loc171_34, %.loc171_34.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc171_16.1: type = splice_block %.loc171_16.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc171_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc171_16.2: type = value_of_initializer %Bool.call.loc171_16 [concrete = bool]
// CHECK:STDOUT:       %.loc171_16.3: type = converted %Bool.call.loc171_16, %.loc171_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: %FDifferentParamType = value_param call_param1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [concrete = constants.%FDifferentParamType]
// CHECK:STDOUT:     %b: %FDifferentParamType = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc171_38.2: %FDifferentParamType.as.J.impl.F.type.6b537d.2 = fn_decl @FDifferentParamType.as.J.impl.F.loc171_38.2 [concrete = constants.%FDifferentParamType.as.J.impl.F.04313a.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentParamType.as.J.impl.F.decl.loc171_38.1
// CHECK:STDOUT:   witness = @FDifferentParamType.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentParamType.as.Destroy.impl: constants.%FDifferentParamType as constants.%Destroy.type {
// CHECK:STDOUT:   %FDifferentParamType.as.Destroy.impl.Op.decl: %FDifferentParamType.as.Destroy.impl.Op.type = fn_decl @FDifferentParamType.as.Destroy.impl.Op [concrete = constants.%FDifferentParamType.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.e0c = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e0c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc156: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.f1f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [concrete = constants.%FDifferentParamType]
// CHECK:STDOUT:     %self: %ptr.f1f = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FDifferentParamType.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FDifferentParamType.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentImplicitParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1: %FDifferentImplicitParamType.as.J.impl.F.type.d6232a.1 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.886f70.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.9cc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9cc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Bool.call.loc184_34: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc184_34.1: type = value_of_initializer %Bool.call.loc184_34 [concrete = bool]
// CHECK:STDOUT:     %.loc184_34.2: type = converted %Bool.call.loc184_34, %.loc184_34.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %FDifferentImplicitParamType = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self: %FDifferentImplicitParamType = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc184_25.1: type = splice_block %.loc184_25.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc184_25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc184_25.2: type = value_of_initializer %Bool.call.loc184_25 [concrete = bool]
// CHECK:STDOUT:       %.loc184_25.3: type = converted %Bool.call.loc184_25, %.loc184_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.2: %FDifferentImplicitParamType.as.J.impl.F.type.d6232a.2 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.886f70.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1
// CHECK:STDOUT:   witness = @FDifferentImplicitParamType.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentImplicitParamType.as.Destroy.impl: constants.%FDifferentImplicitParamType as constants.%Destroy.type {
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.Destroy.impl.Op.decl: %FDifferentImplicitParamType.as.Destroy.impl.Op.type = fn_decl @FDifferentImplicitParamType.as.Destroy.impl.Op [concrete = constants.%FDifferentImplicitParamType.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.986 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.986 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc175: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.dc5 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self: %ptr.dc5 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FDifferentImplicitParamType.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FDifferentImplicitParamType.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc200_38.1: %FDifferentReturnType.as.J.impl.F.type.d3b58f.1 = fn_decl @FDifferentReturnType.as.J.impl.F.loc200_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.be86c9.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.8ee = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8ee = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc200_16.1: type = splice_block %.loc200_16.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc200_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc200_16.2: type = value_of_initializer %Bool.call.loc200_16 [concrete = bool]
// CHECK:STDOUT:       %.loc200_16.3: type = converted %Bool.call.loc200_16, %.loc200_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc200_25.1: type = splice_block %.loc200_25.3 [concrete = bool] {
// CHECK:STDOUT:       %Bool.call.loc200_25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc200_25.2: type = value_of_initializer %Bool.call.loc200_25 [concrete = bool]
// CHECK:STDOUT:       %.loc200_25.3: type = converted %Bool.call.loc200_25, %.loc200_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param call_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc200_38.2: %FDifferentReturnType.as.J.impl.F.type.d3b58f.2 = fn_decl @FDifferentReturnType.as.J.impl.F.loc200_38.2 [concrete = constants.%FDifferentReturnType.as.J.impl.F.be86c9.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentReturnType.as.J.impl.F.decl.loc200_38.1
// CHECK:STDOUT:   witness = @FDifferentReturnType.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentReturnType.as.Destroy.impl: constants.%FDifferentReturnType as constants.%Destroy.type {
// CHECK:STDOUT:   %FDifferentReturnType.as.Destroy.impl.Op.decl: %FDifferentReturnType.as.Destroy.impl.Op.type = fn_decl @FDifferentReturnType.as.Destroy.impl.Op [concrete = constants.%FDifferentReturnType.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.246 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.246 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc188: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.954 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %self: %ptr.954 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %FDifferentReturnType.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @FDifferentReturnType.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadParam.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.f90165.1 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.fa8d04.1] {
// CHECK:STDOUT:     %x.patt: %pattern_type.714 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.714 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1f1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1f1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc223_65: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref.loc223_65 [concrete = constants.%array_type.a41]
// CHECK:STDOUT:     %x.param: %tuple.type.a7d = value_param call_param0
// CHECK:STDOUT:     %.loc223_53.1: type = splice_block %.loc223_53.3 [concrete = constants.%tuple.type.a7d] {
// CHECK:STDOUT:       %SelfNestedBadParam.ref.loc223_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadParam.ref.loc223_14 [concrete = constants.%ptr.4cd]
// CHECK:STDOUT:       %int_32.loc223_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc223_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc223_49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc223_49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete = constants.%struct_type.x.y.871]
// CHECK:STDOUT:       %.loc223_53.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc223_53.3: type = converted %.loc223_53.2, constants.%tuple.type.a7d [concrete = constants.%tuple.type.a7d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a7d = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.a41 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.a41 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.f90165.2 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.fa8d04.2] {
// CHECK:STDOUT:     %x.patt: %pattern_type.a5c = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a5c = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1f1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1f1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.9c9 = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.9c9 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.a41 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.a41 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1
// CHECK:STDOUT:   witness = @SelfNestedBadParam.%SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadParam.as.Destroy.impl: constants.%SelfNestedBadParam as constants.%Destroy.type {
// CHECK:STDOUT:   %SelfNestedBadParam.as.Destroy.impl.Op.decl: %SelfNestedBadParam.as.Destroy.impl.Op.type = fn_decl @SelfNestedBadParam.as.Destroy.impl.Op [concrete = constants.%SelfNestedBadParam.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.2a7 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2a7 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc214: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.4cd = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadParam [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %self: %ptr.4cd = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %SelfNestedBadParam.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @SelfNestedBadParam.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadReturnType.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.0e7d1d.1 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.0bc78a.1] {
// CHECK:STDOUT:     %x.patt: %pattern_type.23f = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.23f = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1f1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1f1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref [concrete = constants.%array_type.a41]
// CHECK:STDOUT:     %x.param: %tuple.type.eb9 = value_param call_param0
// CHECK:STDOUT:     %.loc239_78.1: type = splice_block %.loc239_78.3 [concrete = constants.%tuple.type.eb9] {
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc239_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadReturnType.ref.loc239_14 [concrete = constants.%ptr.612]
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc239_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete = constants.%struct_type.x.y.ac5]
// CHECK:STDOUT:       %.loc239_78.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc239_78.3: type = converted %.loc239_78.2, constants.%tuple.type.eb9 [concrete = constants.%tuple.type.eb9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.eb9 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.a41 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.a41 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.0e7d1d.2 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.0bc78a.2] {
// CHECK:STDOUT:     %x.patt: %pattern_type.23f = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.23f = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.f56 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f56 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.eb9 = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.eb9 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.126 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.126 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadReturnType = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1
// CHECK:STDOUT:   witness = @SelfNestedBadReturnType.%SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadReturnType.as.Destroy.impl: constants.%SelfNestedBadReturnType as constants.%Destroy.type {
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.Destroy.impl.Op.decl: %SelfNestedBadReturnType.as.Destroy.impl.Op.type = fn_decl @SelfNestedBadReturnType.as.Destroy.impl.Op [concrete = constants.%SelfNestedBadReturnType.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.4de = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4de = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc227: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.612 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadReturnType [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %self: %ptr.612 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %SelfNestedBadReturnType.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @SelfNestedBadReturnType.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoF {
// CHECK:STDOUT:   impl_decl @NoF.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoF [concrete = constants.%NoF]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @NoF.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.901]
// CHECK:STDOUT:   impl_decl @NoF.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@NoF.as.Destroy.impl.%NoF.as.Destroy.impl.Op.decl), @NoF.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.9b3]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FNotFunction.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.44e]
// CHECK:STDOUT:   impl_decl @FNotFunction.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FNotFunction.as.Destroy.impl.%FNotFunction.as.Destroy.impl.Op.decl), @FNotFunction.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.f9b]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FAlias.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.c80]
// CHECK:STDOUT:   impl_decl @FAlias.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FAlias.as.Destroy.impl.%FAlias.as.Destroy.impl.Op.decl), @FAlias.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.740]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.2), @FExtraParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.1f3]
// CHECK:STDOUT:   impl_decl @FExtraParam.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FExtraParam.as.Destroy.impl.%FExtraParam.as.Destroy.impl.Op.decl), @FExtraParam.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.44a]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.2), @FExtraImplicitParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.a7b]
// CHECK:STDOUT:   impl_decl @FExtraImplicitParam.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FExtraImplicitParam.as.Destroy.impl.%FExtraImplicitParam.as.Destroy.impl.Op.decl), @FExtraImplicitParam.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.a05]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FExtraReturnType.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.59a]
// CHECK:STDOUT:   impl_decl @FExtraReturnType.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FExtraReturnType.as.Destroy.impl.%FExtraReturnType.as.Destroy.impl.Op.decl), @FExtraReturnType.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.3ea]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_31.2), @FMissingParam.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.4e9]
// CHECK:STDOUT:   impl_decl @FMissingParam.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FMissingParam.as.Destroy.impl.%FMissingParam.as.Destroy.impl.Op.decl), @FMissingParam.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.672]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.2), @FMissingImplicitParam.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.841]
// CHECK:STDOUT:   impl_decl @FMissingImplicitParam.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FMissingImplicitParam.as.Destroy.impl.%FMissingImplicitParam.as.Destroy.impl.Op.decl), @FMissingImplicitParam.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.ae6]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc152_30.2), @FMissingReturnType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.e75]
// CHECK:STDOUT:   impl_decl @FMissingReturnType.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FMissingReturnType.as.Destroy.impl.%FMissingReturnType.as.Destroy.impl.Op.decl), @FMissingReturnType.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.8d8]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc171_38.2), @FDifferentParamType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.cbe]
// CHECK:STDOUT:   impl_decl @FDifferentParamType.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FDifferentParamType.as.Destroy.impl.%FDifferentParamType.as.Destroy.impl.Op.decl), @FDifferentParamType.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.443]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.2), @FDifferentImplicitParamType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.af2]
// CHECK:STDOUT:   impl_decl @FDifferentImplicitParamType.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FDifferentImplicitParamType.as.Destroy.impl.%FDifferentImplicitParamType.as.Destroy.impl.Op.decl), @FDifferentImplicitParamType.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.c2a]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc200_38.2), @FDifferentReturnType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.bc1]
// CHECK:STDOUT:   impl_decl @FDifferentReturnType.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@FDifferentReturnType.as.Destroy.impl.%FDifferentReturnType.as.Destroy.impl.Op.decl), @FDifferentReturnType.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.6d2]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (@SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.2), @SelfNestedBadParam.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.e31]
// CHECK:STDOUT:   impl_decl @SelfNestedBadParam.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@SelfNestedBadParam.as.Destroy.impl.%SelfNestedBadParam.as.Destroy.impl.Op.decl), @SelfNestedBadParam.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.908]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (@SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.2), @SelfNestedBadReturnType.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.d5e]
// CHECK:STDOUT:   impl_decl @SelfNestedBadReturnType.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@SelfNestedBadReturnType.as.Destroy.impl.%SelfNestedBadReturnType.as.Destroy.impl.Op.decl), @SelfNestedBadReturnType.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.9f5]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   .SelfNested = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadReturnType = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoF.as.Destroy.impl.Op(%self.param: %ptr.0f9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FNotFunction.as.Destroy.impl.Op(%self.param: %ptr.2b2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PossiblyF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FAlias.as.Destroy.impl.Op(%self.param: %ptr.4bf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraParam.as.I.impl.F.loc69_18.1(%b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraParam.as.I.impl.F.loc69_18.2() [thunk @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraParam.as.I.impl.F.type.44ef8c.1 = name_ref F, @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.424e9e.1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraParam.as.Destroy.impl.Op(%self.param: %ptr.0f2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_23.1(%self.param: %FExtraImplicitParam);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_23.2() [thunk @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraImplicitParam.as.I.impl.F.type.e1abdd.1 = name_ref F, @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_23.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.6ff574.1]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.call: init %empty_tuple.type = call %F.ref(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.Destroy.impl.Op(%self.param: %ptr.8c7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraReturnType.as.I.impl.F() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraReturnType.as.Destroy.impl.Op(%self.param: %ptr.147) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn(%self.param: bool, %b.param: bool) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc117_31.1(%self.param: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc117_31.2(%self.param: bool, %b.param: bool) -> bool [thunk @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_31.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingParam.as.J.impl.F.type.69596c.1 = name_ref F, @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_31.1 [concrete = constants.%FMissingParam.as.J.impl.F.738f31.1]
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %return.ref: ref bool = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingParam.as.Destroy.impl.Op(%self.param: %ptr.fc8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.1(%b.param: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.2(%self.param: bool, %b.param: bool) -> bool [thunk @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingImplicitParam.as.J.impl.F.type.d97cef.1 = name_ref F, @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.01de92.1]
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %return.ref: ref bool = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.call: init bool = call %F.ref(%b.ref)
// CHECK:STDOUT:   %.loc130_26.1: bool = value_of_initializer %FMissingImplicitParam.as.J.impl.F.call
// CHECK:STDOUT:   %.loc130_26.2: bool = converted %FMissingImplicitParam.as.J.impl.F.call, %.loc130_26.1
// CHECK:STDOUT:   return %.loc130_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.Destroy.impl.Op(%self.param: %ptr.c3c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc152_30.1(%self.param: bool, %b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc152_30.2(%self.param: bool, %b.param: bool) -> bool [thunk @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc152_30.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingReturnType.as.J.impl.F.type.123d7a.1 = name_ref F, @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc152_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.c7d02d.1]
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %return.ref: ref bool = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.call: init %empty_tuple.type = call %FMissingReturnType.as.J.impl.F.bound(%self.ref, %b.ref)
// CHECK:STDOUT:   %.loc152: bool = converted %FMissingReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.Destroy.impl.Op(%self.param: %ptr.41e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc171_38.1(%self.param: bool, %b.param: %FDifferentParamType) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc171_38.2(%self.param: bool, %b.param: bool) -> bool [thunk @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc171_38.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentParamType.as.J.impl.F.type.6b537d.1 = name_ref F, @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc171_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.04313a.1]
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %return.ref: ref bool = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %.loc103: %FDifferentParamType = converted %b.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.call: init bool = call %FDifferentParamType.as.J.impl.F.bound(%self.ref, <error>)
// CHECK:STDOUT:   %.loc171_38.1: bool = value_of_initializer %FDifferentParamType.as.J.impl.F.call
// CHECK:STDOUT:   %.loc171_38.2: bool = converted %FDifferentParamType.as.J.impl.F.call, %.loc171_38.1
// CHECK:STDOUT:   return %.loc171_38.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.Destroy.impl.Op(%self.param: %ptr.f1f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc184_38.1(%self.param: %FDifferentImplicitParamType, %b.param: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc184_38.2(%self.param: bool, %b.param: bool) -> bool [thunk @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentImplicitParamType.as.J.impl.F.type.d6232a.1 = name_ref F, @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc184_38.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.886f70.1]
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %return.ref: ref bool = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %.loc103: %FDifferentImplicitParamType = converted %self.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.call: init bool = call %FDifferentImplicitParamType.as.J.impl.F.bound(<error>, %b.ref)
// CHECK:STDOUT:   %.loc184_38.1: bool = value_of_initializer %FDifferentImplicitParamType.as.J.impl.F.call
// CHECK:STDOUT:   %.loc184_38.2: bool = converted %FDifferentImplicitParamType.as.J.impl.F.call, %.loc184_38.1
// CHECK:STDOUT:   return %.loc184_38.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.Destroy.impl.Op(%self.param: %ptr.dc5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc200_38.1(%self.param: bool, %b.param: bool) -> %return.param: %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc200_38.2(%self.param: bool, %b.param: bool) -> bool [thunk @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc200_38.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentReturnType.as.J.impl.F.type.d3b58f.1 = name_ref F, @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc200_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.be86c9.1]
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %return.ref: ref bool = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   %.loc200_38.1: ref %FDifferentReturnType = temporary_storage
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.call: init %FDifferentReturnType = call %FDifferentReturnType.as.J.impl.F.bound(%self.ref, %b.ref) to %.loc200_38.1
// CHECK:STDOUT:   %.loc200_38.2: bool = converted %FDifferentReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentReturnType.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc200_38.1, constants.%FDifferentReturnType.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.954 = addr_of %.loc200_38.1
// CHECK:STDOUT:   %FDifferentReturnType.as.Destroy.impl.Op.call: init %empty_tuple.type = call %FDifferentReturnType.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.Destroy.impl.Op(%self.param: %ptr.954) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SelfNested.F(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:   %Self.as_type.loc211_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc211_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:   %ptr.loc211_16.1: type = ptr_type %Self.as_type.loc211_16.1 [symbolic = %ptr.loc211_16.1 (constants.%ptr.e87)]
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1: type = struct_type {.x: @SelfNested.F.%Self.as_type.loc211_16.1 (%Self.as_type.e1e), .y: %i32} [symbolic = %struct_type.x.y.loc211_37.1 (constants.%struct_type.x.y.81e)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc211_16.1, %struct_type.x.y.loc211_37.1) [symbolic = %tuple.type (constants.%tuple.type.229)]
// CHECK:STDOUT:   %pattern_type.loc211_8: type = pattern_type %tuple.type [symbolic = %pattern_type.loc211_8 (constants.%pattern_type.ced)]
// CHECK:STDOUT:   %array_type.loc211_57.1: type = array_type constants.%int_4, %Self.as_type.loc211_16.1 [symbolic = %array_type.loc211_57.1 (constants.%array_type.873)]
// CHECK:STDOUT:   %pattern_type.loc211_41: type = pattern_type %array_type.loc211_57.1 [symbolic = %pattern_type.loc211_41 (constants.%pattern_type.c7e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @SelfNested.F.%tuple.type (%tuple.type.229)) -> @SelfNested.F.%array_type.loc211_57.1 (%array_type.873);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.1(%x.param: %tuple.type.a7d) -> %return.param: %array_type.a41;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc223_87.2(%x.param: %tuple.type.9c9) -> %return.param: %array_type.a41 [thunk @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadParam.as.SelfNested.impl.F.type.f90165.1 = name_ref F, @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc223_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.fa8d04.1]
// CHECK:STDOUT:   %x.ref: %tuple.type.9c9 = name_ref x, %x.param
// CHECK:STDOUT:   %return.ref: ref %array_type.a41 = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %.loc211_41: ref %array_type.a41 = splice_block %return {}
// CHECK:STDOUT:   %tuple.elem0: %ptr.4cd = tuple_access %x.ref, element0
// CHECK:STDOUT:   %tuple.elem1: %struct_type.x.y.a89 = tuple_access %x.ref, element1
// CHECK:STDOUT:   %.loc211_9.1: %SelfNestedBadParam = struct_access %tuple.elem1, element0
// CHECK:STDOUT:   %.loc211_9.2: %i32 = converted %.loc211_9.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.call: init %array_type.a41 = call %F.ref(<error>) to %.loc211_41
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc211_41, constants.%T.as.Destroy.impl.Op.d25
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.d25, @T.as.Destroy.impl.Op(constants.%array_type.a41) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc211_41, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f40 = addr_of %.loc211_41
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %SelfNestedBadParam.as.SelfNested.impl.F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.Destroy.impl.Op(%self.param: %ptr.4cd) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.1(%x.param: %tuple.type.eb9) -> %return.param: %array_type.a41;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc239_112.2(%x.param: %tuple.type.eb9) -> %return.param: %array_type.126 [thunk @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.0e7d1d.1 = name_ref F, @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc239_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.0bc78a.1]
// CHECK:STDOUT:   %x.ref: %tuple.type.eb9 = name_ref x, %x.param
// CHECK:STDOUT:   %return.ref: ref %array_type.126 = name_ref <return slot>, %return.param
// CHECK:STDOUT:   %.loc239_112.1: ref %array_type.a41 = temporary_storage
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.call: init %array_type.a41 = call %F.ref(%x.ref) to %.loc239_112.1
// CHECK:STDOUT:   %.loc239_112.2: %array_type.126 = converted %SelfNestedBadReturnType.as.SelfNested.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc239_112.1, constants.%T.as.Destroy.impl.Op.d25
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.d25, @T.as.Destroy.impl.Op(constants.%array_type.a41) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc239_112.1, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f40 = addr_of %.loc239_112.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.Destroy.impl.Op(%self.param: %ptr.612) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet.fa4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet.ee5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%I.facet.d5c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet.94c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet.e7e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet.974) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet.bf1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.F(constants.%J.facet.67f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.F(constants.%Self.2ff) {
// CHECK:STDOUT:   %Self => constants.%Self.2ff
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%Self.as_type.e1e
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.e87
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.81e
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.229
// CHECK:STDOUT:   %pattern_type.loc211_8 => constants.%pattern_type.ced
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.873
// CHECK:STDOUT:   %pattern_type.loc211_41 => constants.%pattern_type.c7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.F(constants.%SelfNested.facet.608) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.608
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.4cd
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.a89
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.9c9
// CHECK:STDOUT:   %pattern_type.loc211_8 => constants.%pattern_type.a5c
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.a41
// CHECK:STDOUT:   %pattern_type.loc211_41 => constants.%pattern_type.1f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.F(constants.%SelfNested.facet.766) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.766
// CHECK:STDOUT:   %Self.as_type.loc211_16.1 => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %ptr.loc211_16.1 => constants.%ptr.612
// CHECK:STDOUT:   %struct_type.x.y.loc211_37.1 => constants.%struct_type.x.y.ac5
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.eb9
// CHECK:STDOUT:   %pattern_type.loc211_8 => constants.%pattern_type.23f
// CHECK:STDOUT:   %array_type.loc211_57.1 => constants.%array_type.126
// CHECK:STDOUT:   %pattern_type.loc211_41 => constants.%pattern_type.f56
// CHECK:STDOUT: }
// CHECK:STDOUT:
