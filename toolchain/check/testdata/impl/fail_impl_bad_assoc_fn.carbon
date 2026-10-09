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

class FExtraParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: error: 0 arguments passed to function expecting 1 argument [CallArgCountMismatch]
    // CHECK:STDERR:     fn F(b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: calling function declared here [InCallToEntity]
    // CHECK:STDERR:     fn F(b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-34]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F(b: bool);
  }
}

class FExtraImplicitParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: error: 0 arguments passed to function expecting 1 argument [CallArgCountMismatch]
    // CHECK:STDERR:     fn F(self);
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: calling function declared here [InCallToEntity]
    // CHECK:STDERR:     fn F(self);
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-50]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F(self);
  }
}

// TODO: Should this be permitted?
class FExtraReturnType {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: function redeclaration differs because return type is `bool` [FunctionRedeclReturnTypeDiffers]
    // CHECK:STDERR:     fn F() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-64]]:15: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F() -> bool;
  }
}

interface J { fn F(self: bool, b: bool) -> bool; }

class FMissingParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
    // CHECK:STDERR:     fn F(self: bool) -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: calling function declared here [InCallToEntity]
    // CHECK:STDERR:     fn F(self: bool) -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-10]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F(self: bool, b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(self: bool) -> bool;
  }
}

class FMissingImplicitParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: member name of type `<type of F>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
    // CHECK:STDERR:     fn F(b: bool) -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-23]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F(self: bool, b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(b: bool) -> bool;
  }
}

class FMissingReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+12]]:5: error: cannot implicitly convert expression of type `()` to `bool` [ConversionFailure]
    // CHECK:STDERR:     fn F(self: bool, b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+9]]:5: note: type `()` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     fn F(self: bool, b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-39]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F(self: bool, b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon: error: cannot implicitly convert expression of type `bool` to `FDifferentParamType` [ConversionFailure]
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon: note: type `bool` does not implement interface `Core.ImplicitAs(FDifferentParamType)` [MissingImplInMemberAccessInContext]
    fn F(self: bool, b: bool);
  }
}

class FDifferentParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+9]]:22: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(self: bool, b: Self) -> bool;
    // CHECK:STDERR:                      ^~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-54]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F(self: bool, b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon: error: cannot implicitly convert expression of type `bool` to `FDifferentImplicitParamType` [ConversionFailure]
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon: note: type `bool` does not implement interface `Core.ImplicitAs(FDifferentImplicitParamType)` [MissingImplInMemberAccessInContext]
    fn F(self: bool, b: Self) -> bool;
  }
}

class FDifferentImplicitParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(self, b: bool) -> bool;
    // CHECK:STDERR:          ^~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-69]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F(self: bool, b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(self, b: bool) -> bool;
  }
}

class FDifferentReturnType {
  impl as J {
    // TODO: Include an "aka" in this diagnostic to explain that the type `Self` is `FDifferentReturnType`.
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+12]]:5: error: cannot implicitly convert expression of type `Self` to `bool` [ConversionFailure]
    // CHECK:STDERR:     fn F(self: bool, b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+9]]:5: note: type `FDifferentReturnType` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     fn F(self: bool, b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-86]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F(self: bool, b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon: error: cannot implicitly convert expression of type `SelfNestedBadParam` to `i32` [ConversionFailure]
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon: note: type `SelfNestedBadParam` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
    fn F(self: bool, b: bool) -> Self;
  }
}

interface SelfNested {
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.de2: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.20b: %I.WithSelf.F.type.de2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.e24: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [concrete]
// CHECK:STDOUT:   %.6c7: <witness> = impl_self_witness %NoF, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.ff5: <witness> = impl_witness @NoF.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.eb1: %I.type = facet_value %NoF, (%I.impl_witness.ff5) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.37a: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.eb1) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.d9b: %I.WithSelf.F.type.37a = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [concrete]
// CHECK:STDOUT:   %.d4c: <witness> = impl_self_witness %FNotFunction, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.ef7: <witness> = impl_witness @FNotFunction.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %F: type = class_type @F [concrete]
// CHECK:STDOUT:   %I.facet.304: %I.type = facet_value %FNotFunction, (%I.impl_witness.ef7) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.a4f: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.304) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ee2: %I.WithSelf.F.type.a4f = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [concrete]
// CHECK:STDOUT:   %.44b: <witness> = impl_self_witness %FExtraParam, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.0d1: <witness> = impl_witness @FExtraParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %b.param_patt.792: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.d80: %pattern_type.831 = wrapper_binding_pattern b, %b.param_patt.792 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.ec7b0b.1: type = fn_type @FExtraParam.as.I.impl.F.loc53_18.1 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.ef9801.1: %FExtraParam.as.I.impl.F.type.ec7b0b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.476: %I.type = facet_value %FExtraParam, (%I.impl_witness.0d1) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.f2e: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.476) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.91f: %I.WithSelf.F.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.ec7b0b.2: type = fn_type @FExtraParam.as.I.impl.F.loc53_18.2 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.ef9801.2: %FExtraParam.as.I.impl.F.type.ec7b0b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %.77c: <witness> = impl_self_witness %FExtraImplicitParam, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.989: <witness> = impl_witness @FExtraImplicitParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.54d: type = pattern_type %FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %self.param_patt.f41: %pattern_type.54d = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cca: %pattern_type.54d = wrapper_binding_pattern self, %self.param_patt.f41 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.54cac2.1: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc69_15.1 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.a42f35.1: %FExtraImplicitParam.as.I.impl.F.type.54cac2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.7ed: %I.type = facet_value %FExtraImplicitParam, (%I.impl_witness.989) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.0ed: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.7ed) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ec4: %I.WithSelf.F.type.0ed = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.54cac2.2: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc69_15.2 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.a42f35.2: %FExtraImplicitParam.as.I.impl.F.type.54cac2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [concrete]
// CHECK:STDOUT:   %.ebd: <witness> = impl_self_witness %FExtraReturnType, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.5ee: <witness> = impl_witness @FExtraReturnType.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %return.param_patt.5a1: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.c0c: %pattern_type.831 = return_slot_pattern %return.param_patt.5a1, bool [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.type: type = fn_type @FExtraReturnType.as.I.impl.F [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F: %FExtraReturnType.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.6f2: %I.type = facet_value %FExtraReturnType, (%I.impl_witness.5ee) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.fa4: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.6f2) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.5ce: %I.WithSelf.F.type.fa4 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.210: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.3a9: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.210 [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.e09: type = fn_type @J.WithSelf.F, @J.WithSelf(%Self.653) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.e79: %J.WithSelf.F.type.e09 = struct_value () [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.bd8: %J.assoc_type = assoc_entity element0, @J.WithSelf.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [concrete]
// CHECK:STDOUT:   %.599: <witness> = impl_self_witness %FMissingParam, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness.24e: <witness> = impl_witness @FMissingParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.aad6ec.1: type = fn_type @FMissingParam.as.J.impl.F.loc101_29.1 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.dd09dc.1: %FMissingParam.as.J.impl.F.type.aad6ec.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.2a3: %J.type = facet_value %FMissingParam, (%J.impl_witness.24e) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.38e: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.2a3) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.433: %J.WithSelf.F.type.38e = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.aad6ec.2: type = fn_type @FMissingParam.as.J.impl.F.loc101_29.2 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.dd09dc.2: %FMissingParam.as.J.impl.F.type.aad6ec.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [concrete]
// CHECK:STDOUT:   %.f79: <witness> = impl_self_witness %FMissingImplicitParam, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness.0d0: <witness> = impl_witness @FMissingImplicitParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.f666a8.1: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc114_26.1 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.620ed1.1: %FMissingImplicitParam.as.J.impl.F.type.f666a8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.ba1: %J.type = facet_value %FMissingImplicitParam, (%J.impl_witness.0d0) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.5df: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.ba1) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.44b: %J.WithSelf.F.type.5df = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.f666a8.2: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc114_26.2 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.620ed1.2: %FMissingImplicitParam.as.J.impl.F.type.f666a8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [concrete]
// CHECK:STDOUT:   %.199: <witness> = impl_self_witness %FMissingReturnType, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness.50a: <witness> = impl_witness @FMissingReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.9cd56a.1: type = fn_type @FMissingReturnType.as.J.impl.F.loc132_30.1 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.6975be.1: %FMissingReturnType.as.J.impl.F.type.9cd56a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.b68: %J.type = facet_value %FMissingReturnType, (%J.impl_witness.50a) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.628: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.b68) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.70e: %J.WithSelf.F.type.628 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.9cd56a.2: type = fn_type @FMissingReturnType.as.J.impl.F.loc132_30.2 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.6975be.2: %FMissingReturnType.as.J.impl.F.type.9cd56a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.b02: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.862: %ImplicitAs.type.b02 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.07d: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.eae: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.862) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.699: %ImplicitAs.WithSelf.Convert.type.eae = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc0.3a5: %ImplicitAs.assoc_type.07d = assoc_entity element0, imports.%Core.import_ref.9e1 [symbolic]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [concrete]
// CHECK:STDOUT:   %.593: <witness> = impl_self_witness %FDifferentParamType, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness.9e2: <witness> = impl_witness @FDifferentParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.5d3: type = pattern_type %FDifferentParamType [concrete]
// CHECK:STDOUT:   %b.param_patt.0a8: %pattern_type.5d3 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.f7e: %pattern_type.5d3 = wrapper_binding_pattern b, %b.param_patt.0a8 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.4a2e9b.1: type = fn_type @FDifferentParamType.as.J.impl.F.loc147_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.28602a.1: %FDifferentParamType.as.J.impl.F.type.4a2e9b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.73e: %J.type = facet_value %FDifferentParamType, (%J.impl_witness.9e2) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.fec: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.73e) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.236: %J.WithSelf.F.type.fec = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.4a2e9b.2: type = fn_type @FDifferentParamType.as.J.impl.F.loc147_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.28602a.2: %FDifferentParamType.as.J.impl.F.type.4a2e9b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.85e: type = facet_type <@ImplicitAs, @ImplicitAs(%FDifferentParamType)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.a99: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%FDifferentParamType) [concrete]
// CHECK:STDOUT:   %assoc0.8f3: %ImplicitAs.assoc_type.a99 = assoc_entity element0, imports.%Core.import_ref.a03 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %.46c: <witness> = impl_self_witness %FDifferentImplicitParamType, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness.d1a: <witness> = impl_witness @FDifferentImplicitParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.1bc: type = pattern_type %FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %self.param_patt.ea6: %pattern_type.1bc = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.85b: %pattern_type.1bc = wrapper_binding_pattern self, %self.param_patt.ea6 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.1: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc160_32.1 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.792c78.1: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.00a: %J.type = facet_value %FDifferentImplicitParamType, (%J.impl_witness.d1a) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.fa8: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.00a) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.a0b: %J.WithSelf.F.type.fa8 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.2: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc160_32.2 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.792c78.2: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.444: type = facet_type <@ImplicitAs, @ImplicitAs(%FDifferentImplicitParamType)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.884: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%FDifferentImplicitParamType) [concrete]
// CHECK:STDOUT:   %assoc0.cb0: %ImplicitAs.assoc_type.884 = assoc_entity element0, imports.%Core.import_ref.a03 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [concrete]
// CHECK:STDOUT:   %.ce7: <witness> = impl_self_witness %FDifferentReturnType, @J [concrete]
// CHECK:STDOUT:   %J.impl_witness.6a4: <witness> = impl_witness @FDifferentReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %.1f5: Core.Form = init_form %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.b9f: type = pattern_type %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %return.param_patt.f37: %pattern_type.b9f = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4f: %pattern_type.b9f = return_slot_pattern %return.param_patt.f37, %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.ff55f4.1: type = fn_type @FDifferentReturnType.as.J.impl.F.loc179_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.9725a5.1: %FDifferentReturnType.as.J.impl.F.type.ff55f4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.6a7: %J.type = facet_value %FDifferentReturnType, (%J.impl_witness.6a4) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.b41: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.6a7) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.982: %J.WithSelf.F.type.b41 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.ff55f4.2: type = fn_type @FDifferentReturnType.as.J.impl.F.loc179_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.9725a5.2: %FDifferentReturnType.as.J.impl.F.type.ff55f4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.e81: %SelfNested.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.8ab: type = facet_access_type %Self.e81 [symbolic]
// CHECK:STDOUT:   %ptr.a16: type = ptr_type %Self.as_type.8ab [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.07c: type = struct_type {.x: %Self.as_type.8ab, .y: %i32} [symbolic]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.a60: %tuple.type.693 = tuple_value (%ptr.a16, %struct_type.x.y.07c) [symbolic]
// CHECK:STDOUT:   %tuple.type.823: type = tuple_type (%ptr.a16, %struct_type.x.y.07c) [symbolic]
// CHECK:STDOUT:   %pattern_type.1ac: type = pattern_type %tuple.type.823 [symbolic]
// CHECK:STDOUT:   %x.param_patt.be0: %pattern_type.1ac = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.971: %pattern_type.1ac = wrapper_binding_pattern x, %x.param_patt.be0 [symbolic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Array.type: type = generic_class_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.2fe: type = class_type @Array, @Array(%Self.as_type.8ab, %int_4) [symbolic]
// CHECK:STDOUT:   %.baf: Core.Form = init_form %Array.2fe [symbolic]
// CHECK:STDOUT:   %pattern_type.8ac: type = pattern_type %Array.2fe [symbolic]
// CHECK:STDOUT:   %return.param_patt.404: %pattern_type.8ac = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.472: %pattern_type.8ac = return_slot_pattern %return.param_patt.404, %Array.2fe [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.3a1: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%Self.e81) [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.796: %SelfNested.WithSelf.F.type.3a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.5a5: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %.83b: <witness> = impl_self_witness %SelfNestedBadParam, @SelfNested [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.09d: <witness> = impl_witness @SelfNestedBadParam.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.52e: type = ptr_type %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %struct_type.x.y.17f: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.e5a: %tuple.type.693 = tuple_value (%ptr.52e, %struct_type.x.y.17f) [concrete]
// CHECK:STDOUT:   %tuple.type.c98: type = tuple_type (%ptr.52e, %struct_type.x.y.17f) [concrete]
// CHECK:STDOUT:   %pattern_type.6af: type = pattern_type %tuple.type.c98 [concrete]
// CHECK:STDOUT:   %x.param_patt.3da: %pattern_type.6af = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.4cd: %pattern_type.6af = wrapper_binding_pattern x, %x.param_patt.3da [concrete]
// CHECK:STDOUT:   %Array.58c: type = class_type @Array, @Array(%SelfNestedBadParam, %int_4) [concrete]
// CHECK:STDOUT:   %.867: Core.Form = init_form %Array.58c [concrete]
// CHECK:STDOUT:   %pattern_type.443: type = pattern_type %Array.58c [concrete]
// CHECK:STDOUT:   %return.param_patt.a95: %pattern_type.443 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.cbd: %pattern_type.443 = return_slot_pattern %return.param_patt.a95, %Array.58c [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.1: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc196_87.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.5fd896.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.d0d: %SelfNested.type = facet_value %SelfNestedBadParam, (%SelfNested.impl_witness.09d) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.49e: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.d0d) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.4af: %SelfNested.WithSelf.F.type.49e = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.18a: type = struct_type {.x: %SelfNestedBadParam, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.ad6: %tuple.type.693 = tuple_value (%ptr.52e, %struct_type.x.y.18a) [concrete]
// CHECK:STDOUT:   %tuple.type.569: type = tuple_type (%ptr.52e, %struct_type.x.y.18a) [concrete]
// CHECK:STDOUT:   %pattern_type.9fd: type = pattern_type %tuple.type.569 [concrete]
// CHECK:STDOUT:   %x.param_patt.274: %pattern_type.9fd = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.3f0: %pattern_type.9fd = wrapper_binding_pattern x, %x.param_patt.274 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.2: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc196_87.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.5fd896.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.d38: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.a95: %ImplicitAs.assoc_type.d38 = assoc_entity element0, imports.%Core.import_ref.a03 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %.020: <witness> = impl_self_witness %SelfNestedBadReturnType, @SelfNested [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.849: <witness> = impl_witness @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.f55: type = ptr_type %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %struct_type.x.y.c16: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.f23: %tuple.type.693 = tuple_value (%ptr.f55, %struct_type.x.y.c16) [concrete]
// CHECK:STDOUT:   %tuple.type.0c2: type = tuple_type (%ptr.f55, %struct_type.x.y.c16) [concrete]
// CHECK:STDOUT:   %pattern_type.047: type = pattern_type %tuple.type.0c2 [concrete]
// CHECK:STDOUT:   %x.param_patt.4a3: %pattern_type.047 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.c39: %pattern_type.047 = wrapper_binding_pattern x, %x.param_patt.4a3 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.1: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc212_112.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.e71: %SelfNested.type = facet_value %SelfNestedBadReturnType, (%SelfNested.impl_witness.849) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.369: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.e71) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.29b: %SelfNested.WithSelf.F.type.369 = struct_value () [concrete]
// CHECK:STDOUT:   %Array.148: type = class_type @Array, @Array(%SelfNestedBadReturnType, %int_4) [concrete]
// CHECK:STDOUT:   %.0b9: Core.Form = init_form %Array.148 [concrete]
// CHECK:STDOUT:   %pattern_type.62c: type = pattern_type %Array.148 [concrete]
// CHECK:STDOUT:   %return.param_patt.287: %pattern_type.62c = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.86d: %pattern_type.62c = return_slot_pattern %return.param_patt.287, %Array.148 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.2: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc212_112.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.383: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.07d) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.3a5)]
// CHECK:STDOUT:   %Core.import_ref.a03: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.eae) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.699)]
// CHECK:STDOUT:   %Core.import_ref.9e1 = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .NoF = %NoF.decl
// CHECK:STDOUT:     .FNotFunction = %FNotFunction.decl
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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.de2) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.20b)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.e24]
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
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.WithSelf.%J.WithSelf.F.type (%J.WithSelf.F.type.e09) = fn_decl @J.WithSelf.F [symbolic = @J.WithSelf.%J.WithSelf.F (constants.%J.WithSelf.F.e79)] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc87_44.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc87_44.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc87_44.2: Core.Form = init_form %.loc87_44.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc87_26: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc87_35: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.WithSelf.F.decl [concrete = constants.%assoc0.bd8]
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
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = constants.%Self.e81]
// CHECK:STDOUT:   %SelfNested.WithSelf.decl = interface_with_self_decl @SelfNested [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %SelfNested.WithSelf.F.decl: @SelfNested.WithSelf.%SelfNested.WithSelf.F.type (%SelfNested.WithSelf.F.type.3a1) = fn_decl @SelfNested.WithSelf.F [symbolic = @SelfNested.WithSelf.%SelfNested.WithSelf.F (constants.%SelfNested.WithSelf.F.796)] {
// CHECK:STDOUT:     %x.param_patt.loc184_9.1: @SelfNested.WithSelf.F.%pattern_type.loc184_9 (%pattern_type.1ac) = value_param_pattern [symbolic = %x.param_patt.loc184_9.2 (constants.%x.param_patt.be0)]
// CHECK:STDOUT:     %x.patt.loc184_9.1: @SelfNested.WithSelf.F.%pattern_type.loc184_9 (%pattern_type.1ac) = wrapper_binding_pattern x, %x.param_patt.loc184_9.1 [symbolic = %x.patt.loc184_9.2 (constants.%x.patt.971)]
// CHECK:STDOUT:     %return.param_patt.loc184_57.1: @SelfNested.WithSelf.F.%pattern_type.loc184_57 (%pattern_type.8ac) = out_param_pattern [symbolic = %return.param_patt.loc184_57.2 (constants.%return.param_patt.404)]
// CHECK:STDOUT:     %return.patt.loc184_41.1: @SelfNested.WithSelf.F.%pattern_type.loc184_57 (%pattern_type.8ac) = return_slot_pattern %return.param_patt.loc184_57.1, %Array.loc184_57.2 [symbolic = %return.patt.loc184_41.2 (constants.%return.patt.472)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc184_50: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Self.as_type.loc184_50: type = facet_access_type %Self.ref.loc184_50 [symbolic = %Self.as_type.loc184_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:     %.loc184_50: type = converted %Self.ref.loc184_50, %Self.as_type.loc184_50 [symbolic = %Self.as_type.loc184_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:     %Array.loc184_57.2: type = class_type @Array, @Array(constants.%Self.as_type.8ab, constants.%int_4) [symbolic = %Array.loc184_57.1 (constants.%Array.2fe)]
// CHECK:STDOUT:     %.loc184_57.2: Core.Form = init_form %Array.loc184_57.2 [symbolic = %.loc184_57.1 (constants.%.baf)]
// CHECK:STDOUT:     %x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.823) = value_param call_param0
// CHECK:STDOUT:     %.loc184_38.1: type = splice_block %.loc184_38.3 [symbolic = %tuple.type (constants.%tuple.type.823)] {
// CHECK:STDOUT:       %Self.ref.loc184_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:       %Self.as_type.loc184_16.2: type = facet_access_type %Self.ref.loc184_12 [symbolic = %Self.as_type.loc184_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %.loc184_16: type = converted %Self.ref.loc184_12, %Self.as_type.loc184_16.2 [symbolic = %Self.as_type.loc184_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %ptr.loc184_16.2: type = ptr_type %.loc184_16 [symbolic = %ptr.loc184_16.1 (constants.%ptr.a16)]
// CHECK:STDOUT:       %Self.ref.loc184_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:       %Self.as_type.loc184_24: type = facet_access_type %Self.ref.loc184_24 [symbolic = %Self.as_type.loc184_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %.loc184_24: type = converted %Self.ref.loc184_24, %Self.as_type.loc184_24 [symbolic = %Self.as_type.loc184_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y.loc184_37.2: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc184_16.1 (%Self.as_type.8ab), .y: %i32} [symbolic = %struct_type.x.y.loc184_37.1 (constants.%struct_type.x.y.07c)]
// CHECK:STDOUT:       %.loc184_38.2: %tuple.type.693 = tuple_literal (%ptr.loc184_16.2, %struct_type.x.y.loc184_37.2) [symbolic = %tuple (constants.%tuple.a60)]
// CHECK:STDOUT:       %.loc184_38.3: type = converted %.loc184_38.2, constants.%tuple.type.823 [symbolic = %tuple.type (constants.%tuple.type.823)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.823) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @SelfNested.WithSelf.F.%Array.loc184_57.1 (%Array.2fe) = out_param call_param1
// CHECK:STDOUT:     %return: ref @SelfNested.WithSelf.F.%Array.loc184_57.1 (%Array.2fe) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %SelfNested.WithSelf.F.decl [concrete = constants.%assoc0.5a5]
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.ff5]
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.ef7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc53_18.1: %FExtraParam.as.I.impl.F.type.ec7b0b.1 = fn_decl @FExtraParam.as.I.impl.F.loc53_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.ef9801.1] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc53: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc53_18.2: %FExtraParam.as.I.impl.F.type.ec7b0b.2 = fn_decl @FExtraParam.as.I.impl.F.loc53_18.2 [concrete = constants.%FExtraParam.as.I.impl.F.ef9801.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraParam.as.I.impl.F.decl.loc53_18.2), @FExtraParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.0d1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraParam.as.I.impl.F.decl.loc53_18.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraImplicitParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc69_15.1: %FExtraImplicitParam.as.I.impl.F.type.54cac2.1 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc69_15.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.a42f35.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.54d = value_param_pattern [concrete = constants.%self.param_patt.f41]
// CHECK:STDOUT:     %self.patt: %pattern_type.54d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cca]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self: %FExtraImplicitParam = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc69_15.2: %FExtraImplicitParam.as.I.impl.F.type.54cac2.2 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc69_15.2 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.a42f35.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraImplicitParam.as.I.impl.F.decl.loc69_15.2), @FExtraImplicitParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.989]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraImplicitParam.as.I.impl.F.decl.loc69_15.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraReturnType.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.decl: %FExtraReturnType.as.I.impl.F.type = fn_decl @FExtraReturnType.as.I.impl.F [concrete = constants.%FExtraReturnType.as.I.impl.F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc83_15.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc83_15.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc83_15.2: Core.Form = init_form %.loc83_15.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FExtraReturnType.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.5ee]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraReturnType.as.I.impl.F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc101_29.1: %FMissingParam.as.J.impl.F.type.aad6ec.1 = fn_decl @FMissingParam.as.J.impl.F.loc101_29.1 [concrete = constants.%FMissingParam.as.J.impl.F.dd09dc.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc101_25.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc101_25.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc101_25.2: Core.Form = init_form %.loc101_25.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc101_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc87_44.1: type = specific_constant @J.WithSelf.F.%.loc87_44.1, @J.WithSelf.F(constants.%J.facet.2a3) [concrete = bool]
// CHECK:STDOUT:   %.loc87_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc87_44.2, @J.WithSelf.F(constants.%J.facet.2a3) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc101_29.2: %FMissingParam.as.J.impl.F.type.aad6ec.2 = fn_decl @FMissingParam.as.J.impl.F.loc101_29.2 [concrete = constants.%FMissingParam.as.J.impl.F.dd09dc.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.2a3) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %.1 [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.2a3) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %.2 [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.2a3) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FMissingParam.as.J.impl.F.decl.loc101_29.2), @FMissingParam.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.24e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingParam.as.J.impl.F.decl.loc101_29.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingImplicitParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc114_26.1: %FMissingImplicitParam.as.J.impl.F.type.f666a8.1 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc114_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.620ed1.1] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc114_22.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc114_22.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc114_22.2: Core.Form = init_form %.loc114_22.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc114_13: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc87_44.1: type = specific_constant @J.WithSelf.F.%.loc87_44.1, @J.WithSelf.F(constants.%J.facet.ba1) [concrete = bool]
// CHECK:STDOUT:   %.loc87_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc87_44.2, @J.WithSelf.F(constants.%J.facet.ba1) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc114_26.2: %FMissingImplicitParam.as.J.impl.F.type.f666a8.2 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc114_26.2 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.620ed1.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.ba1) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %.1 [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.ba1) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %.2 [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.ba1) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FMissingImplicitParam.as.J.impl.F.decl.loc114_26.2), @FMissingImplicitParam.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.0d0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingImplicitParam.as.J.impl.F.decl.loc114_26.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc132_30.1: %FMissingReturnType.as.J.impl.F.type.9cd56a.1 = fn_decl @FMissingReturnType.as.J.impl.F.loc132_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.6975be.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc132_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc132_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc87_44.1: type = specific_constant @J.WithSelf.F.%.loc87_44.1, @J.WithSelf.F(constants.%J.facet.b68) [concrete = bool]
// CHECK:STDOUT:   %.loc87_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc87_44.2, @J.WithSelf.F(constants.%J.facet.b68) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc132_30.2: %FMissingReturnType.as.J.impl.F.type.9cd56a.2 = fn_decl @FMissingReturnType.as.J.impl.F.loc132_30.2 [concrete = constants.%FMissingReturnType.as.J.impl.F.6975be.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.b68) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %.1 [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.b68) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %.2 [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.b68) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FMissingReturnType.as.J.impl.F.decl.loc132_30.2), @FMissingReturnType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.50a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingReturnType.as.J.impl.F.decl.loc132_30.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc147_38.1: %FDifferentParamType.as.J.impl.F.type.4a2e9b.1 = fn_decl @FDifferentParamType.as.J.impl.F.loc147_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.28602a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.5d3 = value_param_pattern [concrete = constants.%b.param_patt.0a8]
// CHECK:STDOUT:     %b.patt: %pattern_type.5d3 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.f7e]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc147_34.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc147_34.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc147_34.2: Core.Form = init_form %.loc147_34.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc147_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: %FDifferentParamType = value_param call_param1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [concrete = constants.%FDifferentParamType]
// CHECK:STDOUT:     %b: %FDifferentParamType = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc87_44.1: type = specific_constant @J.WithSelf.F.%.loc87_44.1, @J.WithSelf.F(constants.%J.facet.73e) [concrete = bool]
// CHECK:STDOUT:   %.loc87_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc87_44.2, @J.WithSelf.F(constants.%J.facet.73e) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc147_38.2: %FDifferentParamType.as.J.impl.F.type.4a2e9b.2 = fn_decl @FDifferentParamType.as.J.impl.F.loc147_38.2 [concrete = constants.%FDifferentParamType.as.J.impl.F.28602a.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.73e) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %.1 [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.73e) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %.2 [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.73e) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FDifferentParamType.as.J.impl.F.decl.loc147_38.2), @FDifferentParamType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.9e2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentParamType.as.J.impl.F.decl.loc147_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentImplicitParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc160_32.1: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.1 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc160_32.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.792c78.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1bc = value_param_pattern [concrete = constants.%self.param_patt.ea6]
// CHECK:STDOUT:     %self.patt: %pattern_type.1bc = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.85b]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc160_28.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc160_28.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc160_28.2: Core.Form = init_form %.loc160_28.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: %FDifferentImplicitParamType = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self: %FDifferentImplicitParamType = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc160_19: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc87_44.1: type = specific_constant @J.WithSelf.F.%.loc87_44.1, @J.WithSelf.F(constants.%J.facet.00a) [concrete = bool]
// CHECK:STDOUT:   %.loc87_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc87_44.2, @J.WithSelf.F(constants.%J.facet.00a) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc160_32.2: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.2 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc160_32.2 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.792c78.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.00a) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %.1 [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.00a) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %.2 [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.00a) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FDifferentImplicitParamType.as.J.impl.F.decl.loc160_32.2), @FDifferentImplicitParamType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.d1a]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentImplicitParamType.as.J.impl.F.decl.loc160_32.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc179_38.1: %FDifferentReturnType.as.J.impl.F.type.ff55f4.1 = fn_decl @FDifferentReturnType.as.J.impl.F.loc179_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.9725a5.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b9f = out_param_pattern [concrete = constants.%return.param_patt.f37]
// CHECK:STDOUT:     %return.patt: %pattern_type.b9f = return_slot_pattern %return.param_patt, %Self.ref [concrete = constants.%return.patt.e4f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %.loc179_34: Core.Form = init_form %Self.ref [concrete = constants.%.1f5]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc179_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc179_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param call_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc87_44.1: type = specific_constant @J.WithSelf.F.%.loc87_44.1, @J.WithSelf.F(constants.%J.facet.6a7) [concrete = bool]
// CHECK:STDOUT:   %.loc87_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc87_44.2, @J.WithSelf.F(constants.%J.facet.6a7) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc179_38.2: %FDifferentReturnType.as.J.impl.F.type.ff55f4.2 = fn_decl @FDifferentReturnType.as.J.impl.F.loc179_38.2 [concrete = constants.%FDifferentReturnType.as.J.impl.F.9725a5.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.6a7) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %.1 [concrete = constants.%self.patt.3a9]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.6a7) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = wrapper_binding_pattern b, %.2 [concrete = constants.%b.patt.d80]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.6a7) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FDifferentReturnType.as.J.impl.F.decl.loc179_38.2), @FDifferentReturnType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.6a4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentReturnType.as.J.impl.F.decl.loc179_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadParam.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc196_87.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.1 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc196_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.5fd896.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6af = value_param_pattern [concrete = constants.%x.param_patt.3da]
// CHECK:STDOUT:     %x.patt: %pattern_type.6af = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.4cd]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.443 = out_param_pattern [concrete = constants.%return.param_patt.a95]
// CHECK:STDOUT:     %return.patt: %pattern_type.443 = return_slot_pattern %return.param_patt, %Array [concrete = constants.%return.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc196_65: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%SelfNestedBadParam, constants.%int_4) [concrete = constants.%Array.58c]
// CHECK:STDOUT:     %.loc196_86: Core.Form = init_form %Array [concrete = constants.%.867]
// CHECK:STDOUT:     %x.param: %tuple.type.c98 = value_param call_param0
// CHECK:STDOUT:     %.loc196_53.1: type = splice_block %.loc196_53.3 [concrete = constants.%tuple.type.c98] {
// CHECK:STDOUT:       %SelfNestedBadParam.ref.loc196_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadParam.ref.loc196_14 [concrete = constants.%ptr.52e]
// CHECK:STDOUT:       %i32.loc196_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc196_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete = constants.%struct_type.x.y.17f]
// CHECK:STDOUT:       %.loc196_53.2: %tuple.type.693 = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.e5a]
// CHECK:STDOUT:       %.loc196_53.3: type = converted %.loc196_53.2, constants.%tuple.type.c98 [concrete = constants.%tuple.type.c98]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.c98 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %Array.58c = out_param call_param1
// CHECK:STDOUT:     %return: ref %Array.58c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc184_57.1: type = specific_constant @SelfNested.WithSelf.F.%Array.loc184_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.d0d) [concrete = constants.%Array.58c]
// CHECK:STDOUT:   %.loc184_57.2: Core.Form = specific_constant @SelfNested.WithSelf.F.%.loc184_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.d0d) [concrete = constants.%.867]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc196_87.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.2 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc196_87.2 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.5fd896.2] {
// CHECK:STDOUT:     %.1: %pattern_type.9fd = specific_constant constants.%x.param_patt.274, @SelfNested.WithSelf.F(constants.%SelfNested.facet.d0d) [concrete = constants.%x.param_patt.274]
// CHECK:STDOUT:     %x.patt: %pattern_type.9fd = wrapper_binding_pattern x, %.1 [concrete = constants.%x.patt.3f0]
// CHECK:STDOUT:     %.2: %pattern_type.443 = specific_constant constants.%return.patt.cbd, @SelfNested.WithSelf.F(constants.%SelfNested.facet.d0d) [concrete = constants.%return.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.569 = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.569 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %Array.58c = out_param call_param1
// CHECK:STDOUT:     %return: ref %Array.58c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc196_87.2), @SelfNestedBadParam.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.09d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc196_87.1
// CHECK:STDOUT:   extend %SelfNested.ref
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadReturnType.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc212_112.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.1 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc212_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.047 = value_param_pattern [concrete = constants.%x.param_patt.4a3]
// CHECK:STDOUT:     %x.patt: %pattern_type.047 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.c39]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.443 = out_param_pattern [concrete = constants.%return.param_patt.a95]
// CHECK:STDOUT:     %return.patt: %pattern_type.443 = return_slot_pattern %return.param_patt, %Array [concrete = constants.%return.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%SelfNestedBadParam, constants.%int_4) [concrete = constants.%Array.58c]
// CHECK:STDOUT:     %.loc212_111: Core.Form = init_form %Array [concrete = constants.%.867]
// CHECK:STDOUT:     %x.param: %tuple.type.0c2 = value_param call_param0
// CHECK:STDOUT:     %.loc212_78.1: type = splice_block %.loc212_78.3 [concrete = constants.%tuple.type.0c2] {
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc212_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadReturnType.ref.loc212_14 [concrete = constants.%ptr.f55]
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc212_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete = constants.%struct_type.x.y.c16]
// CHECK:STDOUT:       %.loc212_78.2: %tuple.type.693 = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.f23]
// CHECK:STDOUT:       %.loc212_78.3: type = converted %.loc212_78.2, constants.%tuple.type.0c2 [concrete = constants.%tuple.type.0c2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.0c2 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %Array.58c = out_param call_param1
// CHECK:STDOUT:     %return: ref %Array.58c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc184_57.1: type = specific_constant @SelfNested.WithSelf.F.%Array.loc184_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.e71) [concrete = constants.%Array.148]
// CHECK:STDOUT:   %.loc184_57.2: Core.Form = specific_constant @SelfNested.WithSelf.F.%.loc184_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.e71) [concrete = constants.%.0b9]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc212_112.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.2 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc212_112.2 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.2] {
// CHECK:STDOUT:     %.1: %pattern_type.047 = specific_constant constants.%x.param_patt.4a3, @SelfNested.WithSelf.F(constants.%SelfNested.facet.e71) [concrete = constants.%x.param_patt.4a3]
// CHECK:STDOUT:     %x.patt: %pattern_type.047 = wrapper_binding_pattern x, %.1 [concrete = constants.%x.patt.c39]
// CHECK:STDOUT:     %.2: %pattern_type.62c = specific_constant constants.%return.patt.86d, @SelfNested.WithSelf.F(constants.%SelfNested.facet.e71) [concrete = constants.%return.patt.86d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.0c2 = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.0c2 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %Array.148 = out_param call_param1
// CHECK:STDOUT:     %return: ref %Array.148 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc212_112.2), @SelfNestedBadReturnType.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.849]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadReturnType = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc212_112.1
// CHECK:STDOUT:   extend %SelfNested.ref
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoF {
// CHECK:STDOUT:   impl_decl @NoF.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoF [concrete = constants.%NoF]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25: <witness> = impl_self_witness @NoF.as.I.impl.%Self.ref, @I [concrete = constants.%.6c7]
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
// CHECK:STDOUT:   %.loc29: <witness> = impl_self_witness @FNotFunction.as.I.impl.%Self.ref, @I [concrete = constants.%.d4c]
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
// CHECK:STDOUT: class @FExtraParam {
// CHECK:STDOUT:   impl_decl @FExtraParam.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraParam [concrete = constants.%FExtraParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc42: <witness> = impl_self_witness @FExtraParam.as.I.impl.%Self.ref, @I [concrete = constants.%.44b]
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
// CHECK:STDOUT:   %.loc58: <witness> = impl_self_witness @FExtraImplicitParam.as.I.impl.%Self.ref, @I [concrete = constants.%.77c]
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
// CHECK:STDOUT:   %.loc75: <witness> = impl_self_witness @FExtraReturnType.as.I.impl.%Self.ref, @I [concrete = constants.%.ebd]
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
// CHECK:STDOUT:   %.loc90: <witness> = impl_self_witness @FMissingParam.as.J.impl.%Self.ref, @J [concrete = constants.%.599]
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
// CHECK:STDOUT:   %.loc106: <witness> = impl_self_witness @FMissingImplicitParam.as.J.impl.%Self.ref, @J [concrete = constants.%.f79]
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
// CHECK:STDOUT:   %.loc119: <witness> = impl_self_witness @FMissingReturnType.as.J.impl.%Self.ref, @J [concrete = constants.%.199]
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
// CHECK:STDOUT:   %.loc137: <witness> = impl_self_witness @FDifferentParamType.as.J.impl.%Self.ref, @J [concrete = constants.%.593]
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
// CHECK:STDOUT:   %.loc152: <witness> = impl_self_witness @FDifferentImplicitParamType.as.J.impl.%Self.ref, @J [concrete = constants.%.46c]
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
// CHECK:STDOUT:   %.loc165: <witness> = impl_self_witness @FDifferentReturnType.as.J.impl.%Self.ref, @J [concrete = constants.%.ce7]
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
// CHECK:STDOUT:   %.loc188: <witness> = impl_self_witness @SelfNestedBadParam.as.SelfNested.impl.%Self.ref, @SelfNested [concrete = constants.%.83b]
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
// CHECK:STDOUT:   %.loc201: <witness> = impl_self_witness @SelfNestedBadReturnType.as.SelfNested.impl.%Self.ref, @SelfNested [concrete = constants.%.020]
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
// CHECK:STDOUT: fn @FExtraParam.as.I.impl.F.loc53_18.1(%b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraParam.as.I.impl.F.loc53_18.2() [thunk @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc53_18.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.476)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraParam.as.I.impl.F.type.ec7b0b.1 = name_ref F, @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc53_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.ef9801.1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc69_15.1(%self.param: %FExtraImplicitParam);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc69_15.2() [thunk @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc69_15.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.7ed)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraImplicitParam.as.I.impl.F.type.54cac2.1 = name_ref F, @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc69_15.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.a42f35.1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraReturnType.as.I.impl.F() -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn(%self.param: bool, %b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc101_29.1(%self.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc101_29.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc101_29.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.2a3)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingParam.as.J.impl.F.type.aad6ec.1 = name_ref F, @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc101_29.1 [concrete = constants.%FMissingParam.as.J.impl.F.dd09dc.1]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc114_26.1(%b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc114_26.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc114_26.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.ba1)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingImplicitParam.as.J.impl.F.type.f666a8.1 = name_ref F, @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc114_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.620ed1.1]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.call: init bool = call %F.ref(%b.param)
// CHECK:STDOUT:   return %FMissingImplicitParam.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc132_30.1(%self.param: bool, %b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc132_30.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc132_30.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.b68)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingReturnType.as.J.impl.F.type.9cd56a.1 = name_ref F, @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc132_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.6975be.1]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.call: init %empty_tuple.type = call %FMissingReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc132: bool = converted %FMissingReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc147_38.1(%self.param: bool, %b.param: %FDifferentParamType) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc147_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc147_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.73e)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentParamType.as.J.impl.F.type.4a2e9b.1 = name_ref F, @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc147_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.28602a.1]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%FDifferentParamType)> [concrete = constants.%ImplicitAs.type.85e]
// CHECK:STDOUT:   %.4: %ImplicitAs.assoc_type.a99 = specific_constant imports.%Core.import_ref.383, @ImplicitAs.WithSelf(constants.%FDifferentParamType, constants.%Self.862) [concrete = constants.%assoc0.8f3]
// CHECK:STDOUT:   %Convert.ref: %ImplicitAs.assoc_type.a99 = name_ref Convert, %.4 [concrete = constants.%assoc0.8f3]
// CHECK:STDOUT:   %.5: %FDifferentParamType = converted %b.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.call: init bool = call %FDifferentParamType.as.J.impl.F.bound(%self.param, <error>)
// CHECK:STDOUT:   return %FDifferentParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc160_32.1(%self.param: %FDifferentImplicitParamType, %b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc160_32.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc160_32.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.00a)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.1 = name_ref F, @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc160_32.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.792c78.1]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%FDifferentImplicitParamType)> [concrete = constants.%ImplicitAs.type.444]
// CHECK:STDOUT:   %.4: %ImplicitAs.assoc_type.884 = specific_constant imports.%Core.import_ref.383, @ImplicitAs.WithSelf(constants.%FDifferentImplicitParamType, constants.%Self.862) [concrete = constants.%assoc0.cb0]
// CHECK:STDOUT:   %Convert.ref: %ImplicitAs.assoc_type.884 = name_ref Convert, %.4 [concrete = constants.%assoc0.cb0]
// CHECK:STDOUT:   %.5: %FDifferentImplicitParamType = converted %self.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.call: init bool = call %FDifferentImplicitParamType.as.J.impl.F.bound(<error>, %b.param)
// CHECK:STDOUT:   return %FDifferentImplicitParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc179_38.1(%self.param: bool, %b.param: bool) -> out %return.param: %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc179_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc179_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.6a7)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentReturnType.as.J.impl.F.type.ff55f4.1 = name_ref F, @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc179_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.9725a5.1]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc179_38.1: ref %FDifferentReturnType = temporary_storage
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.call: init %FDifferentReturnType to %.loc179_38.1 = call %FDifferentReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc179_38.2: bool = converted %FDifferentReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SelfNested.WithSelf.F(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:   %Self.as_type.loc184_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc184_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:   %ptr.loc184_16.1: type = ptr_type %Self.as_type.loc184_16.1 [symbolic = %ptr.loc184_16.1 (constants.%ptr.a16)]
// CHECK:STDOUT:   %struct_type.x.y.loc184_37.1: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc184_16.1 (%Self.as_type.8ab), .y: %i32} [symbolic = %struct_type.x.y.loc184_37.1 (constants.%struct_type.x.y.07c)]
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%ptr.loc184_16.1, %struct_type.x.y.loc184_37.1) [symbolic = %tuple (constants.%tuple.a60)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc184_16.1, %struct_type.x.y.loc184_37.1) [symbolic = %tuple.type (constants.%tuple.type.823)]
// CHECK:STDOUT:   %pattern_type.loc184_9: type = pattern_type %tuple.type [symbolic = %pattern_type.loc184_9 (constants.%pattern_type.1ac)]
// CHECK:STDOUT:   %x.param_patt.loc184_9.2: @SelfNested.WithSelf.F.%pattern_type.loc184_9 (%pattern_type.1ac) = value_param_pattern [symbolic = %x.param_patt.loc184_9.2 (constants.%x.param_patt.be0)]
// CHECK:STDOUT:   %x.patt.loc184_9.2: @SelfNested.WithSelf.F.%pattern_type.loc184_9 (%pattern_type.1ac) = wrapper_binding_pattern x, %x.param_patt.loc184_9.2 [symbolic = %x.patt.loc184_9.2 (constants.%x.patt.971)]
// CHECK:STDOUT:   %Array.loc184_57.1: type = class_type @Array, @Array(%Self.as_type.loc184_16.1, constants.%int_4) [symbolic = %Array.loc184_57.1 (constants.%Array.2fe)]
// CHECK:STDOUT:   %.loc184_57.1: Core.Form = init_form %Array.loc184_57.1 [symbolic = %.loc184_57.1 (constants.%.baf)]
// CHECK:STDOUT:   %pattern_type.loc184_57: type = pattern_type %Array.loc184_57.1 [symbolic = %pattern_type.loc184_57 (constants.%pattern_type.8ac)]
// CHECK:STDOUT:   %return.param_patt.loc184_57.2: @SelfNested.WithSelf.F.%pattern_type.loc184_57 (%pattern_type.8ac) = out_param_pattern [symbolic = %return.param_patt.loc184_57.2 (constants.%return.param_patt.404)]
// CHECK:STDOUT:   %return.patt.loc184_41.2: @SelfNested.WithSelf.F.%pattern_type.loc184_57 (%pattern_type.8ac) = return_slot_pattern %return.param_patt.loc184_57.2, %Array.loc184_57.1 [symbolic = %return.patt.loc184_41.2 (constants.%return.patt.472)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.823)) -> out %return.param: @SelfNested.WithSelf.F.%Array.loc184_57.1 (%Array.2fe);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc196_87.1(%x.param: %tuple.type.c98) -> out %return.param: %Array.58c;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc196_87.2(%x.param: %tuple.type.569) -> out %return.param: %Array.58c [thunk @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc196_87.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.d0d)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.1 = name_ref F, @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc196_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.5fd896.1]
// CHECK:STDOUT:   %.3: ref %Array.58c = splice_block %return.param {}
// CHECK:STDOUT:   %tuple.elem0: %ptr.52e = tuple_access %x.param, element0
// CHECK:STDOUT:   %tuple.elem1: %struct_type.x.y.18a = tuple_access %x.param, element1
// CHECK:STDOUT:   %.4: %SelfNestedBadParam = struct_access %tuple.elem1, element0
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32)> [concrete = constants.%ImplicitAs.type.09b]
// CHECK:STDOUT:   %.5: %ImplicitAs.assoc_type.d38 = specific_constant imports.%Core.import_ref.383, @ImplicitAs.WithSelf(constants.%i32, constants.%Self.862) [concrete = constants.%assoc0.a95]
// CHECK:STDOUT:   %Convert.ref: %ImplicitAs.assoc_type.d38 = name_ref Convert, %.5 [concrete = constants.%assoc0.a95]
// CHECK:STDOUT:   %.6: %i32 = converted %.4, <error> [concrete = <error>]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.call: init %Array.58c to %.3 = call %F.ref(<error>)
// CHECK:STDOUT:   return %SelfNestedBadParam.as.SelfNested.impl.F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc212_112.1(%x.param: %tuple.type.0c2) -> out %return.param: %Array.58c;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc212_112.2(%x.param: %tuple.type.0c2) -> out %return.param: %Array.148 [thunk @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc212_112.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.e71)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.1 = name_ref F, @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc212_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.1]
// CHECK:STDOUT:   %.loc212_112.1: ref %Array.58c = temporary_storage
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.call: init %Array.58c to %.loc212_112.1 = call %F.ref(%x.param)
// CHECK:STDOUT:   %.loc212_112.2: %Array.148 = converted %SelfNestedBadReturnType.as.SelfNested.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.de2
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.20b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.eb1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.eb1
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.37a
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.d9b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.304) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.304
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.a4f
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ee2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.476) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.476
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.f2e
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.91f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.476) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.7ed) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.7ed
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.0ed
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ec4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.7ed) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.6f2) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.6f2
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.fa4
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.5ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.6f2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.653
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.e09
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.e79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.653) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.2a3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.2a3
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.38e
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.433
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.2a3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.ba1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.ba1
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.5df
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.44b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.ba1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.b68) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.b68
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.628
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.70e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.b68) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.73e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.73e
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.fec
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.236
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.73e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.00a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.00a
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.fa8
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.a0b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.00a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.6a7) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.6a7
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.b41
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.982
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.6a7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%Self.e81) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.e81
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.3a1
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.796
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%Self.e81) {
// CHECK:STDOUT:   %Self => constants.%Self.e81
// CHECK:STDOUT:   %Self.as_type.loc184_16.1 => constants.%Self.as_type.8ab
// CHECK:STDOUT:   %ptr.loc184_16.1 => constants.%ptr.a16
// CHECK:STDOUT:   %struct_type.x.y.loc184_37.1 => constants.%struct_type.x.y.07c
// CHECK:STDOUT:   %tuple => constants.%tuple.a60
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.823
// CHECK:STDOUT:   %pattern_type.loc184_9 => constants.%pattern_type.1ac
// CHECK:STDOUT:   %x.param_patt.loc184_9.2 => constants.%x.param_patt.be0
// CHECK:STDOUT:   %x.patt.loc184_9.2 => constants.%x.patt.971
// CHECK:STDOUT:   %Array.loc184_57.1 => constants.%Array.2fe
// CHECK:STDOUT:   %.loc184_57.1 => constants.%.baf
// CHECK:STDOUT:   %pattern_type.loc184_57 => constants.%pattern_type.8ac
// CHECK:STDOUT:   %return.param_patt.loc184_57.2 => constants.%return.param_patt.404
// CHECK:STDOUT:   %return.patt.loc184_41.2 => constants.%return.patt.472
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.d0d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.d0d
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.49e
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.4af
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.d0d) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.d0d
// CHECK:STDOUT:   %Self.as_type.loc184_16.1 => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %ptr.loc184_16.1 => constants.%ptr.52e
// CHECK:STDOUT:   %struct_type.x.y.loc184_37.1 => constants.%struct_type.x.y.18a
// CHECK:STDOUT:   %tuple => constants.%tuple.ad6
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.569
// CHECK:STDOUT:   %pattern_type.loc184_9 => constants.%pattern_type.9fd
// CHECK:STDOUT:   %x.param_patt.loc184_9.2 => constants.%x.param_patt.274
// CHECK:STDOUT:   %x.patt.loc184_9.2 => constants.%x.patt.3f0
// CHECK:STDOUT:   %Array.loc184_57.1 => constants.%Array.58c
// CHECK:STDOUT:   %.loc184_57.1 => constants.%.867
// CHECK:STDOUT:   %pattern_type.loc184_57 => constants.%pattern_type.443
// CHECK:STDOUT:   %return.param_patt.loc184_57.2 => constants.%return.param_patt.a95
// CHECK:STDOUT:   %return.patt.loc184_41.2 => constants.%return.patt.cbd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.e71) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.e71
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.369
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.29b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.e71) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.e71
// CHECK:STDOUT:   %Self.as_type.loc184_16.1 => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %ptr.loc184_16.1 => constants.%ptr.f55
// CHECK:STDOUT:   %struct_type.x.y.loc184_37.1 => constants.%struct_type.x.y.c16
// CHECK:STDOUT:   %tuple => constants.%tuple.f23
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.0c2
// CHECK:STDOUT:   %pattern_type.loc184_9 => constants.%pattern_type.047
// CHECK:STDOUT:   %x.param_patt.loc184_9.2 => constants.%x.param_patt.4a3
// CHECK:STDOUT:   %x.patt.loc184_9.2 => constants.%x.patt.c39
// CHECK:STDOUT:   %Array.loc184_57.1 => constants.%Array.148
// CHECK:STDOUT:   %.loc184_57.1 => constants.%.0b9
// CHECK:STDOUT:   %pattern_type.loc184_57 => constants.%pattern_type.62c
// CHECK:STDOUT:   %return.param_patt.loc184_57.2 => constants.%return.param_patt.287
// CHECK:STDOUT:   %return.patt.loc184_41.2 => constants.%return.patt.86d
// CHECK:STDOUT: }
// CHECK:STDOUT:
