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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: error: 0 arguments passed to function expecting 1 argument [CallArgCountMismatch]
    // CHECK:STDERR:     fn F(self);
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: note: calling function declared here [InCallToEntity]
    // CHECK:STDERR:     fn F(self);
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-66]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-80]]:15: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+12]]:5: error: cannot implicitly convert expression of type `FDifferentReturnType` to `bool` [ConversionFailure]
    // CHECK:STDERR:     fn F(self: bool, b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+9]]:5: note: type `FDifferentReturnType` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR:     fn F(self: bool, b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-85]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
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
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.dcc: type = fn_type @I.WithSelf.F, @I.WithSelf(%Self.37e) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.ec6: %I.WithSelf.F.type.dcc = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.6c9: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [concrete]
// CHECK:STDOUT:   %I.impl_witness.85e: <witness> = impl_witness @NoF.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.752: %I.type = facet_value %NoF, (%I.impl_witness.85e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.3d3: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.752) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.38a: %I.WithSelf.F.type.3d3 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [concrete]
// CHECK:STDOUT:   %I.impl_witness.b51: <witness> = impl_witness @FNotFunction.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %F: type = class_type @F [concrete]
// CHECK:STDOUT:   %I.facet.010: %I.type = facet_value %FNotFunction, (%I.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.642: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.010) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.c29: %I.WithSelf.F.type.642 = struct_value () [concrete]
// CHECK:STDOUT:   %PossiblyF.type: type = fn_type @PossiblyF [concrete]
// CHECK:STDOUT:   %PossiblyF: %PossiblyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [concrete]
// CHECK:STDOUT:   %I.impl_witness.e2e: <witness> = impl_witness @FAlias.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet.f6f: %I.type = facet_value %FAlias, (%I.impl_witness.e2e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.751: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.f6f) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.6d6: %I.WithSelf.F.type.751 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.99e: <witness> = impl_witness @FExtraParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %b.param_patt.792: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.4e2: %pattern_type.831 = at_binding_pattern b, %b.param_patt.792 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.ec7b0b.1: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.1 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.ef9801.1: %FExtraParam.as.I.impl.F.type.ec7b0b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.657: %I.type = facet_value %FExtraParam, (%I.impl_witness.99e) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.255: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.657) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.fc1: %I.WithSelf.F.type.255 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.type.ec7b0b.2: type = fn_type @FExtraParam.as.I.impl.F.loc69_18.2 [concrete]
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.ef9801.2: %FExtraParam.as.I.impl.F.type.ec7b0b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.2da: <witness> = impl_witness @FExtraImplicitParam.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.54d: type = pattern_type %FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %self.param_patt.f41: %pattern_type.54d = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.3d5: %pattern_type.54d = at_binding_pattern self, %self.param_patt.f41 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.54cac2.1: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_15.1 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.a42f35.1: %FExtraImplicitParam.as.I.impl.F.type.54cac2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.334: %I.type = facet_value %FExtraImplicitParam, (%I.impl_witness.2da) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.6ff: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.334) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.39e: %I.WithSelf.F.type.6ff = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.type.54cac2.2: type = fn_type @FExtraImplicitParam.as.I.impl.F.loc85_15.2 [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.a42f35.2: %FExtraImplicitParam.as.I.impl.F.type.54cac2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [concrete]
// CHECK:STDOUT:   %I.impl_witness.b63: <witness> = impl_witness @FExtraReturnType.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %return.param_patt.5a1: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.c0c: %pattern_type.831 = return_slot_pattern %return.param_patt.5a1, bool [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.type: type = fn_type @FExtraReturnType.as.I.impl.F [concrete]
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F: %FExtraReturnType.as.I.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.dde: %I.type = facet_value %FExtraReturnType, (%I.impl_witness.b63) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.type.fea: type = fn_type @I.WithSelf.F, @I.WithSelf(%I.facet.dde) [concrete]
// CHECK:STDOUT:   %I.WithSelf.F.338: %I.WithSelf.F.type.fea = struct_value () [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.653: %J.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.210: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f7d: %pattern_type.831 = at_binding_pattern self, %self.param_patt.210 [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.a0a: type = fn_type @J.WithSelf.F, @J.WithSelf(%Self.653) [symbolic]
// CHECK:STDOUT:   %J.WithSelf.F.f69: %J.WithSelf.F.type.a0a = struct_value () [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.000: %J.assoc_type = assoc_entity element0, @J.WithSelf.%J.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.a34: <witness> = impl_witness @FMissingParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.aad6ec.1: type = fn_type @FMissingParam.as.J.impl.F.loc117_29.1 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.dd09dc.1: %FMissingParam.as.J.impl.F.type.aad6ec.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.d77: %J.type = facet_value %FMissingParam, (%J.impl_witness.a34) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.51b: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.d77) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.ed6: %J.WithSelf.F.type.51b = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.type.aad6ec.2: type = fn_type @FMissingParam.as.J.impl.F.loc117_29.2 [concrete]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.dd09dc.2: %FMissingParam.as.J.impl.F.type.aad6ec.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.7e6: <witness> = impl_witness @FMissingImplicitParam.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.f666a8.1: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.620ed1.1: %FMissingImplicitParam.as.J.impl.F.type.f666a8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.3f0: %J.type = facet_value %FMissingImplicitParam, (%J.impl_witness.7e6) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.e23: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.3f0) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.897: %J.WithSelf.F.type.e23 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.type.f666a8.2: type = fn_type @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.620ed1.2: %FMissingImplicitParam.as.J.impl.F.type.f666a8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.05c: <witness> = impl_witness @FMissingReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.9cd56a.1: type = fn_type @FMissingReturnType.as.J.impl.F.loc148_30.1 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.6975be.1: %FMissingReturnType.as.J.impl.F.type.9cd56a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.c42: %J.type = facet_value %FMissingReturnType, (%J.impl_witness.05c) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.f86: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.c42) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.d52: %J.WithSelf.F.type.f86 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.type.9cd56a.2: type = fn_type @FMissingReturnType.as.J.impl.F.loc148_30.2 [concrete]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.6975be.2: %FMissingReturnType.as.J.impl.F.type.9cd56a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.294: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.fcc: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.97e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.945: %ImplicitAs.WithSelf.Convert.type.97e = struct_value () [symbolic]
// CHECK:STDOUT:   %assoc0.0ac: %ImplicitAs.assoc_type.fcc = assoc_entity element0, imports.%Core.import_ref.f9d [symbolic]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.553: <witness> = impl_witness @FDifferentParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.5d3: type = pattern_type %FDifferentParamType [concrete]
// CHECK:STDOUT:   %b.param_patt.0a8: %pattern_type.5d3 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.05d: %pattern_type.5d3 = at_binding_pattern b, %b.param_patt.0a8 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.4a2e9b.1: type = fn_type @FDifferentParamType.as.J.impl.F.loc163_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.28602a.1: %FDifferentParamType.as.J.impl.F.type.4a2e9b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.47a: %J.type = facet_value %FDifferentParamType, (%J.impl_witness.553) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.9c3: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.47a) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.4cd: %J.WithSelf.F.type.9c3 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.type.4a2e9b.2: type = fn_type @FDifferentParamType.as.J.impl.F.loc163_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.28602a.2: %FDifferentParamType.as.J.impl.F.type.4a2e9b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4ba: type = facet_type <@ImplicitAs, @ImplicitAs(%FDifferentParamType)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.503: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%FDifferentParamType) [concrete]
// CHECK:STDOUT:   %assoc0.603: %ImplicitAs.assoc_type.503 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.cc3: <witness> = impl_witness @FDifferentImplicitParamType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.1bc: type = pattern_type %FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %self.param_patt.ea6: %pattern_type.1bc = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a07: %pattern_type.1bc = at_binding_pattern self, %self.param_patt.ea6 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.1: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc176_32.1 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.792c78.1: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.929: %J.type = facet_value %FDifferentImplicitParamType, (%J.impl_witness.cc3) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.97c: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.929) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.94b: %J.WithSelf.F.type.97c = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.2: type = fn_type @FDifferentImplicitParamType.as.J.impl.F.loc176_32.2 [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.792c78.2: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.de7: type = facet_type <@ImplicitAs, @ImplicitAs(%FDifferentImplicitParamType)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.36e: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%FDifferentImplicitParamType) [concrete]
// CHECK:STDOUT:   %assoc0.769: %ImplicitAs.assoc_type.36e = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.6ab: <witness> = impl_witness @FDifferentReturnType.as.J.impl.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %.1f5: Core.Form = init_form %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.b9f: type = pattern_type %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %return.param_patt.f37: %pattern_type.b9f = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4f: %pattern_type.b9f = return_slot_pattern %return.param_patt.f37, %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.ff55f4.1: type = fn_type @FDifferentReturnType.as.J.impl.F.loc194_38.1 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.9725a5.1: %FDifferentReturnType.as.J.impl.F.type.ff55f4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.b85: %J.type = facet_value %FDifferentReturnType, (%J.impl_witness.6ab) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.type.a9c: type = fn_type @J.WithSelf.F, @J.WithSelf(%J.facet.b85) [concrete]
// CHECK:STDOUT:   %J.WithSelf.F.6ec: %J.WithSelf.F.type.a9c = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.type.ff55f4.2: type = fn_type @FDifferentReturnType.as.J.impl.F.loc194_38.2 [concrete]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.9725a5.2: %FDifferentReturnType.as.J.impl.F.type.ff55f4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.e81: %SelfNested.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.8ab: type = facet_access_type %Self.e81 [symbolic]
// CHECK:STDOUT:   %ptr.a16: type = ptr_type %Self.as_type.8ab [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.4c5: type = struct_type {.x: %Self.as_type.8ab, .y: %i32} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.b6a: %tuple.type.24b = tuple_value (%ptr.a16, %struct_type.x.y.4c5) [symbolic]
// CHECK:STDOUT:   %tuple.type.c49: type = tuple_type (%ptr.a16, %struct_type.x.y.4c5) [symbolic]
// CHECK:STDOUT:   %pattern_type.60a: type = pattern_type %tuple.type.c49 [symbolic]
// CHECK:STDOUT:   %x.param_patt.e85: %pattern_type.60a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.ad4: %pattern_type.60a = at_binding_pattern x, %x.param_patt.e85 [symbolic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.4e4: type = array_type %int_4, %Self.as_type.8ab [symbolic]
// CHECK:STDOUT:   %.f45: Core.Form = init_form %array_type.4e4 [symbolic]
// CHECK:STDOUT:   %pattern_type.2ab: type = pattern_type %array_type.4e4 [symbolic]
// CHECK:STDOUT:   %return.param_patt.439: %pattern_type.2ab = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.869: %pattern_type.2ab = return_slot_pattern %return.param_patt.439, %array_type.4e4 [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.702: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%Self.e81) [symbolic]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.eec: %SelfNested.WithSelf.F.type.702 = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.219: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.d7c: <witness> = impl_witness @SelfNestedBadParam.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.52e: type = ptr_type %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %struct_type.x.y.fb8: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.991: %tuple.type.24b = tuple_value (%ptr.52e, %struct_type.x.y.fb8) [concrete]
// CHECK:STDOUT:   %tuple.type.e57: type = tuple_type (%ptr.52e, %struct_type.x.y.fb8) [concrete]
// CHECK:STDOUT:   %pattern_type.a68: type = pattern_type %tuple.type.e57 [concrete]
// CHECK:STDOUT:   %x.param_patt.140: %pattern_type.a68 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.19e: %pattern_type.a68 = at_binding_pattern x, %x.param_patt.140 [concrete]
// CHECK:STDOUT:   %array_type.291: type = array_type %int_4, %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %.cc2: Core.Form = init_form %array_type.291 [concrete]
// CHECK:STDOUT:   %pattern_type.3a0: type = pattern_type %array_type.291 [concrete]
// CHECK:STDOUT:   %return.param_patt.489: %pattern_type.3a0 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.98a: %pattern_type.3a0 = return_slot_pattern %return.param_patt.489, %array_type.291 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.1: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc211_87.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.5fd896.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.fc6: %SelfNested.type = facet_value %SelfNestedBadParam, (%SelfNested.impl_witness.d7c) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.75d: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.fc6) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.379: %SelfNested.WithSelf.F.type.75d = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.da2: type = struct_type {.x: %SelfNestedBadParam, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.09d: %tuple.type.24b = tuple_value (%ptr.52e, %struct_type.x.y.da2) [concrete]
// CHECK:STDOUT:   %tuple.type.9ab: type = tuple_type (%ptr.52e, %struct_type.x.y.da2) [concrete]
// CHECK:STDOUT:   %pattern_type.94f: type = pattern_type %tuple.type.9ab [concrete]
// CHECK:STDOUT:   %x.param_patt.7b9: %pattern_type.94f = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.4b4: %pattern_type.94f = at_binding_pattern x, %x.param_patt.7b9 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.2: type = fn_type @SelfNestedBadParam.as.SelfNested.impl.F.loc211_87.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.5fd896.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e23: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.368: %ImplicitAs.assoc_type.e23 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.abd: <witness> = impl_witness @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.f55: type = ptr_type %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %struct_type.x.y.f6a: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.96d: %tuple.type.24b = tuple_value (%ptr.f55, %struct_type.x.y.f6a) [concrete]
// CHECK:STDOUT:   %tuple.type.829: type = tuple_type (%ptr.f55, %struct_type.x.y.f6a) [concrete]
// CHECK:STDOUT:   %pattern_type.5c2: type = pattern_type %tuple.type.829 [concrete]
// CHECK:STDOUT:   %x.param_patt.7be: %pattern_type.5c2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.ef5: %pattern_type.5c2 = at_binding_pattern x, %x.param_patt.7be [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.1: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc227_112.1 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.68c: %SelfNested.type = facet_value %SelfNestedBadReturnType, (%SelfNested.impl_witness.abd) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type.53d: type = fn_type @SelfNested.WithSelf.F, @SelfNested.WithSelf(%SelfNested.facet.68c) [concrete]
// CHECK:STDOUT:   %SelfNested.WithSelf.F.f39: %SelfNested.WithSelf.F.type.53d = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.a08: type = array_type %int_4, %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %.442: Core.Form = init_form %array_type.a08 [concrete]
// CHECK:STDOUT:   %pattern_type.f0d: type = pattern_type %array_type.a08 [concrete]
// CHECK:STDOUT:   %return.param_patt.039: %pattern_type.f0d = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.3c4: %pattern_type.f0d = return_slot_pattern %return.param_patt.039, %array_type.a08 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.2: type = fn_type @SelfNestedBadReturnType.as.SelfNested.impl.F.loc227_112.2 [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.484: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.fcc) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.0ac)]
// CHECK:STDOUT:   %Core.import_ref.cb2: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.97e) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.945)]
// CHECK:STDOUT:   %Core.import_ref.f9d = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.dcc) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.ec6)] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %I.WithSelf.F.decl [concrete = constants.%assoc0.6c9]
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
// CHECK:STDOUT:   %Self: %J.type = symbolic_binding Self, 0 [symbolic = constants.%Self.653]
// CHECK:STDOUT:   %J.WithSelf.decl = interface_with_self_decl @J [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %J.WithSelf.F.decl: @J.WithSelf.%J.WithSelf.F.type (%J.WithSelf.F.type.a0a) = fn_decl @J.WithSelf.F [symbolic = @J.WithSelf.%J.WithSelf.F (constants.%J.WithSelf.F.f69)] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc103_44.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc103_44.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc103_44.2: Core.Form = init_form %.loc103_44.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc103_26: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc103_35: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %J.WithSelf.F.decl [concrete = constants.%assoc0.000]
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
// CHECK:STDOUT:   %SelfNested.WithSelf.F.decl: @SelfNested.WithSelf.%SelfNested.WithSelf.F.type (%SelfNested.WithSelf.F.type.702) = fn_decl @SelfNested.WithSelf.F [symbolic = @SelfNested.WithSelf.%SelfNested.WithSelf.F (constants.%SelfNested.WithSelf.F.eec)] {
// CHECK:STDOUT:     %x.param_patt.loc199_9.1: @SelfNested.WithSelf.F.%pattern_type.loc199_9 (%pattern_type.60a) = value_param_pattern [symbolic = %x.param_patt.loc199_9.2 (constants.%x.param_patt.e85)]
// CHECK:STDOUT:     %x.patt.loc199_9.1: @SelfNested.WithSelf.F.%pattern_type.loc199_9 (%pattern_type.60a) = at_binding_pattern x, %x.param_patt.loc199_9.1 [symbolic = %x.patt.loc199_9.2 (constants.%x.patt.ad4)]
// CHECK:STDOUT:     %return.param_patt.loc199_57.1: @SelfNested.WithSelf.F.%pattern_type.loc199_57 (%pattern_type.2ab) = out_param_pattern [symbolic = %return.param_patt.loc199_57.2 (constants.%return.param_patt.439)]
// CHECK:STDOUT:     %return.patt.loc199_41.1: @SelfNested.WithSelf.F.%pattern_type.loc199_57 (%pattern_type.2ab) = return_slot_pattern %return.param_patt.loc199_57.1, %array_type.loc199_57.2 [symbolic = %return.patt.loc199_41.2 (constants.%return.patt.869)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc199_50: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Self.as_type.loc199_50: type = facet_access_type %Self.ref.loc199_50 [symbolic = %Self.as_type.loc199_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:     %.loc199_50: type = converted %Self.ref.loc199_50, %Self.as_type.loc199_50 [symbolic = %Self.as_type.loc199_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:     %array_type.loc199_57.2: type = array_type %int_4, %.loc199_50 [symbolic = %array_type.loc199_57.1 (constants.%array_type.4e4)]
// CHECK:STDOUT:     %.loc199_57.2: Core.Form = init_form %array_type.loc199_57.2 [symbolic = %.loc199_57.1 (constants.%.f45)]
// CHECK:STDOUT:     %x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.c49) = value_param call_param0
// CHECK:STDOUT:     %.loc199_38.1: type = splice_block %.loc199_38.3 [symbolic = %tuple.type (constants.%tuple.type.c49)] {
// CHECK:STDOUT:       %Self.ref.loc199_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:       %Self.as_type.loc199_16.2: type = facet_access_type %Self.ref.loc199_12 [symbolic = %Self.as_type.loc199_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %.loc199_16: type = converted %Self.ref.loc199_12, %Self.as_type.loc199_16.2 [symbolic = %Self.as_type.loc199_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %ptr.loc199_16.2: type = ptr_type %.loc199_16 [symbolic = %ptr.loc199_16.1 (constants.%ptr.a16)]
// CHECK:STDOUT:       %Self.ref.loc199_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:       %Self.as_type.loc199_24: type = facet_access_type %Self.ref.loc199_24 [symbolic = %Self.as_type.loc199_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %.loc199_24: type = converted %Self.ref.loc199_24, %Self.as_type.loc199_24 [symbolic = %Self.as_type.loc199_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y.loc199_37.2: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc199_16.1 (%Self.as_type.8ab), .y: %i32} [symbolic = %struct_type.x.y.loc199_37.1 (constants.%struct_type.x.y.4c5)]
// CHECK:STDOUT:       %.loc199_38.2: %tuple.type.24b = tuple_literal (%ptr.loc199_16.2, %struct_type.x.y.loc199_37.2) [symbolic = %tuple (constants.%tuple.b6a)]
// CHECK:STDOUT:       %.loc199_38.3: type = converted %.loc199_38.2, constants.%tuple.type.c49 [symbolic = %tuple.type (constants.%tuple.type.c49)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.c49) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @SelfNested.WithSelf.F.%array_type.loc199_57.1 (%array_type.4e4) = out_param call_param1
// CHECK:STDOUT:     %return: ref @SelfNested.WithSelf.F.%array_type.loc199_57.1 (%array_type.4e4) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %SelfNested.WithSelf.F.decl [concrete = constants.%assoc0.219]
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.85e]
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.b51]
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
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.e2e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .PossiblyF = <poisoned>
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.1: %FExtraParam.as.I.impl.F.type.ec7b0b.1 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.ef9801.1] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc69: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraParam.as.I.impl.F.decl.loc69_18.2: %FExtraParam.as.I.impl.F.type.ec7b0b.2 = fn_decl @FExtraParam.as.I.impl.F.loc69_18.2 [concrete = constants.%FExtraParam.as.I.impl.F.ef9801.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraParam.as.I.impl.F.decl.loc69_18.2), @FExtraParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.99e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraParam.as.I.impl.F.decl.loc69_18.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraImplicitParam.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_15.1: %FExtraImplicitParam.as.I.impl.F.type.54cac2.1 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_15.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.a42f35.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.54d = value_param_pattern [concrete = constants.%self.param_patt.f41]
// CHECK:STDOUT:     %self.patt: %pattern_type.54d = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3d5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self: %FExtraImplicitParam = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FExtraImplicitParam.as.I.impl.F.decl.loc85_15.2: %FExtraImplicitParam.as.I.impl.F.type.54cac2.2 = fn_decl @FExtraImplicitParam.as.I.impl.F.loc85_15.2 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.a42f35.2] {} {}
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%FExtraImplicitParam.as.I.impl.F.decl.loc85_15.2), @FExtraImplicitParam.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.2da]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraImplicitParam.as.I.impl.F.decl.loc85_15.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FExtraReturnType.as.I.impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %FExtraReturnType.as.I.impl.F.decl: %FExtraReturnType.as.I.impl.F.type = fn_decl @FExtraReturnType.as.I.impl.F [concrete = constants.%FExtraReturnType.as.I.impl.F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc99_15.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc99_15.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc99_15.2: Core.Form = init_form %.loc99_15.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @FExtraReturnType.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.b63]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FExtraReturnType.as.I.impl.F.decl
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_29.1: %FMissingParam.as.J.impl.F.type.aad6ec.1 = fn_decl @FMissingParam.as.J.impl.F.loc117_29.1 [concrete = constants.%FMissingParam.as.J.impl.F.dd09dc.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc117_25.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc117_25.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc117_25.2: Core.Form = init_form %.loc117_25.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc117_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.d77) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.d77) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.decl.loc117_29.2: %FMissingParam.as.J.impl.F.type.aad6ec.2 = fn_decl @FMissingParam.as.J.impl.F.loc117_29.2 [concrete = constants.%FMissingParam.as.J.impl.F.dd09dc.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.d77) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %.1 [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.d77) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %.2 [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.d77) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FMissingParam.as.J.impl.F.decl.loc117_29.2), @FMissingParam.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.a34]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingParam.as.J.impl.F.decl.loc117_29.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingImplicitParam.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1: %FMissingImplicitParam.as.J.impl.F.type.f666a8.1 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.620ed1.1] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc130_22.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc130_22.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc130_22.2: Core.Form = init_form %.loc130_22.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc130_13: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.3f0) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.3f0) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.2: %FMissingImplicitParam.as.J.impl.F.type.f666a8.2 = fn_decl @FMissingImplicitParam.as.J.impl.F.loc130_26.2 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.620ed1.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.3f0) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %.1 [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.3f0) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %.2 [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.3f0) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.2), @FMissingImplicitParam.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.7e6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FMissingReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc148_30.1: %FMissingReturnType.as.J.impl.F.type.9cd56a.1 = fn_decl @FMissingReturnType.as.J.impl.F.loc148_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.6975be.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc148_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc148_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.c42) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.c42) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.decl.loc148_30.2: %FMissingReturnType.as.J.impl.F.type.9cd56a.2 = fn_decl @FMissingReturnType.as.J.impl.F.loc148_30.2 [concrete = constants.%FMissingReturnType.as.J.impl.F.6975be.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.c42) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %.1 [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.c42) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %.2 [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.c42) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FMissingReturnType.as.J.impl.F.decl.loc148_30.2), @FMissingReturnType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.05c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FMissingReturnType.as.J.impl.F.decl.loc148_30.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc163_38.1: %FDifferentParamType.as.J.impl.F.type.4a2e9b.1 = fn_decl @FDifferentParamType.as.J.impl.F.loc163_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.28602a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.5d3 = value_param_pattern [concrete = constants.%b.param_patt.0a8]
// CHECK:STDOUT:     %b.patt: %pattern_type.5d3 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.05d]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc163_34.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc163_34.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc163_34.2: Core.Form = init_form %.loc163_34.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc163_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: %FDifferentParamType = value_param call_param1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [concrete = constants.%FDifferentParamType]
// CHECK:STDOUT:     %b: %FDifferentParamType = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.47a) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.47a) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.decl.loc163_38.2: %FDifferentParamType.as.J.impl.F.type.4a2e9b.2 = fn_decl @FDifferentParamType.as.J.impl.F.loc163_38.2 [concrete = constants.%FDifferentParamType.as.J.impl.F.28602a.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.47a) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %.1 [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.47a) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %.2 [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.47a) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FDifferentParamType.as.J.impl.F.decl.loc163_38.2), @FDifferentParamType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.553]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentParamType.as.J.impl.F.decl.loc163_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentImplicitParamType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc176_32.1: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.1 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc176_32.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.792c78.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1bc = value_param_pattern [concrete = constants.%self.param_patt.ea6]
// CHECK:STDOUT:     %self.patt: %pattern_type.1bc = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a07]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern [concrete = constants.%return.param_patt.5a1]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern %return.param_patt, %.loc176_28.1 [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc176_28.1: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %.loc176_28.2: Core.Form = init_form %.loc176_28.1 [concrete = constants.%.f34]
// CHECK:STDOUT:     %self.param: %FDifferentImplicitParamType = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self: %FDifferentImplicitParamType = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc176_19: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.929) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.929) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.decl.loc176_32.2: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.2 = fn_decl @FDifferentImplicitParamType.as.J.impl.F.loc176_32.2 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.792c78.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.929) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %.1 [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.929) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %.2 [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.929) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FDifferentImplicitParamType.as.J.impl.F.decl.loc176_32.2), @FDifferentImplicitParamType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.cc3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentImplicitParamType.as.J.impl.F.decl.loc176_32.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @FDifferentReturnType.as.J.impl: %Self.ref as %J.ref {
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc194_38.1: %FDifferentReturnType.as.J.impl.F.type.ff55f4.1 = fn_decl @FDifferentReturnType.as.J.impl.F.loc194_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.9725a5.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b9f = out_param_pattern [concrete = constants.%return.param_patt.f37]
// CHECK:STDOUT:     %return.patt: %pattern_type.b9f = return_slot_pattern %return.param_patt, %Self.ref [concrete = constants.%return.patt.e4f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %.loc194_34: Core.Form = init_form %Self.ref [concrete = constants.%.1f5]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc194_16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc194_25: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param call_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc103_44.1: type = specific_constant @J.WithSelf.F.%.loc103_44.1, @J.WithSelf.F(constants.%J.facet.b85) [concrete = bool]
// CHECK:STDOUT:   %.loc103_44.2: Core.Form = specific_constant @J.WithSelf.F.%.loc103_44.2, @J.WithSelf.F(constants.%J.facet.b85) [concrete = constants.%.f34]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.decl.loc194_38.2: %FDifferentReturnType.as.J.impl.F.type.ff55f4.2 = fn_decl @FDifferentReturnType.as.J.impl.F.loc194_38.2 [concrete = constants.%FDifferentReturnType.as.J.impl.F.9725a5.2] {
// CHECK:STDOUT:     %.1: %pattern_type.831 = specific_constant constants.%self.param_patt.210, @J.WithSelf.F(constants.%J.facet.b85) [concrete = constants.%self.param_patt.210]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = at_binding_pattern self, %.1 [concrete = constants.%self.patt.f7d]
// CHECK:STDOUT:     %.2: %pattern_type.831 = specific_constant constants.%b.param_patt.792, @J.WithSelf.F(constants.%J.facet.b85) [concrete = constants.%b.param_patt.792]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = at_binding_pattern b, %.2 [concrete = constants.%b.patt.4e2]
// CHECK:STDOUT:     %.3: %pattern_type.831 = specific_constant constants.%return.patt.c0c, @J.WithSelf.F(constants.%J.facet.b85) [concrete = constants.%return.patt.c0c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %self: bool = wrapper_binding self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %b: bool = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (%FDifferentReturnType.as.J.impl.F.decl.loc194_38.2), @FDifferentReturnType.as.J.impl [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.6ab]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %FDifferentReturnType.as.J.impl.F.decl.loc194_38.1
// CHECK:STDOUT:   extend %J.ref
// CHECK:STDOUT:   witness = %J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadParam.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc211_87.1: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.1 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc211_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.5fd896.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.a68 = value_param_pattern [concrete = constants.%x.param_patt.140]
// CHECK:STDOUT:     %x.patt: %pattern_type.a68 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.19e]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.3a0 = out_param_pattern [concrete = constants.%return.param_patt.489]
// CHECK:STDOUT:     %return.patt: %pattern_type.3a0 = return_slot_pattern %return.param_patt, %array_type [concrete = constants.%return.patt.98a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc211_65: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref.loc211_65 [concrete = constants.%array_type.291]
// CHECK:STDOUT:     %.loc211_86: Core.Form = init_form %array_type [concrete = constants.%.cc2]
// CHECK:STDOUT:     %x.param: %tuple.type.e57 = value_param call_param0
// CHECK:STDOUT:     %.loc211_53.1: type = splice_block %.loc211_53.3 [concrete = constants.%tuple.type.e57] {
// CHECK:STDOUT:       %SelfNestedBadParam.ref.loc211_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadParam.ref.loc211_14 [concrete = constants.%ptr.52e]
// CHECK:STDOUT:       %i32.loc211_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc211_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete = constants.%struct_type.x.y.fb8]
// CHECK:STDOUT:       %.loc211_53.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.991]
// CHECK:STDOUT:       %.loc211_53.3: type = converted %.loc211_53.2, constants.%tuple.type.e57 [concrete = constants.%tuple.type.e57]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.e57 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.291 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.291 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc199_57.1: type = specific_constant @SelfNested.WithSelf.F.%array_type.loc199_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.fc6) [concrete = constants.%array_type.291]
// CHECK:STDOUT:   %.loc199_57.2: Core.Form = specific_constant @SelfNested.WithSelf.F.%.loc199_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.fc6) [concrete = constants.%.cc2]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc211_87.2: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.2 = fn_decl @SelfNestedBadParam.as.SelfNested.impl.F.loc211_87.2 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.5fd896.2] {
// CHECK:STDOUT:     %.1: %pattern_type.94f = specific_constant constants.%x.param_patt.7b9, @SelfNested.WithSelf.F(constants.%SelfNested.facet.fc6) [concrete = constants.%x.param_patt.7b9]
// CHECK:STDOUT:     %x.patt: %pattern_type.94f = at_binding_pattern x, %.1 [concrete = constants.%x.patt.4b4]
// CHECK:STDOUT:     %.2: %pattern_type.3a0 = specific_constant constants.%return.patt.98a, @SelfNested.WithSelf.F(constants.%SelfNested.facet.fc6) [concrete = constants.%return.patt.98a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.9ab = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.9ab = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.291 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.291 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc211_87.2), @SelfNestedBadParam.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.d7c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadParam.as.SelfNested.impl.F.decl.loc211_87.1
// CHECK:STDOUT:   extend %SelfNested.ref
// CHECK:STDOUT:   witness = %SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @SelfNestedBadReturnType.as.SelfNested.impl: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc227_112.1: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.1 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc227_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.5c2 = value_param_pattern [concrete = constants.%x.param_patt.7be]
// CHECK:STDOUT:     %x.patt: %pattern_type.5c2 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.ef5]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.3a0 = out_param_pattern [concrete = constants.%return.param_patt.489]
// CHECK:STDOUT:     %return.patt: %pattern_type.3a0 = return_slot_pattern %return.param_patt, %array_type [concrete = constants.%return.patt.98a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref [concrete = constants.%array_type.291]
// CHECK:STDOUT:     %.loc227_111: Core.Form = init_form %array_type [concrete = constants.%.cc2]
// CHECK:STDOUT:     %x.param: %tuple.type.829 = value_param call_param0
// CHECK:STDOUT:     %.loc227_78.1: type = splice_block %.loc227_78.3 [concrete = constants.%tuple.type.829] {
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc227_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadReturnType.ref.loc227_14 [concrete = constants.%ptr.f55]
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc227_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete = constants.%struct_type.x.y.f6a]
// CHECK:STDOUT:       %.loc227_78.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y) [concrete = constants.%tuple.96d]
// CHECK:STDOUT:       %.loc227_78.3: type = converted %.loc227_78.2, constants.%tuple.type.829 [concrete = constants.%tuple.type.829]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.829 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.291 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.291 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc199_57.1: type = specific_constant @SelfNested.WithSelf.F.%array_type.loc199_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.68c) [concrete = constants.%array_type.a08]
// CHECK:STDOUT:   %.loc199_57.2: Core.Form = specific_constant @SelfNested.WithSelf.F.%.loc199_57.2, @SelfNested.WithSelf.F(constants.%SelfNested.facet.68c) [concrete = constants.%.442]
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc227_112.2: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.2 = fn_decl @SelfNestedBadReturnType.as.SelfNested.impl.F.loc227_112.2 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.2] {
// CHECK:STDOUT:     %.1: %pattern_type.5c2 = specific_constant constants.%x.param_patt.7be, @SelfNested.WithSelf.F(constants.%SelfNested.facet.68c) [concrete = constants.%x.param_patt.7be]
// CHECK:STDOUT:     %x.patt: %pattern_type.5c2 = at_binding_pattern x, %.1 [concrete = constants.%x.patt.ef5]
// CHECK:STDOUT:     %.2: %pattern_type.f0d = specific_constant constants.%return.patt.3c4, @SelfNested.WithSelf.F(constants.%SelfNested.facet.68c) [concrete = constants.%return.patt.3c4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %tuple.type.829 = value_param call_param0
// CHECK:STDOUT:     %x: %tuple.type.829 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.a08 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.a08 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc227_112.2), @SelfNestedBadReturnType.as.SelfNested.impl [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.abd]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadReturnType = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc227_112.1
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
// CHECK:STDOUT: fn @FExtraParam.as.I.impl.F.loc69_18.2() [thunk @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.657)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraParam.as.I.impl.F.type.ec7b0b.1 = name_ref F, @FExtraParam.as.I.impl.%FExtraParam.as.I.impl.F.decl.loc69_18.1 [concrete = constants.%FExtraParam.as.I.impl.F.ef9801.1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_15.1(%self.param: %FExtraImplicitParam);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraImplicitParam.as.I.impl.F.loc85_15.2() [thunk @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_15.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet.334)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FExtraImplicitParam.as.I.impl.F.type.54cac2.1 = name_ref F, @FExtraImplicitParam.as.I.impl.%FExtraImplicitParam.as.I.impl.F.decl.loc85_15.1 [concrete = constants.%FExtraImplicitParam.as.I.impl.F.a42f35.1]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FExtraReturnType.as.I.impl.F() -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J.WithSelf.F(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn(%self.param: bool, %b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc117_29.1(%self.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingParam.as.J.impl.F.loc117_29.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_29.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.d77)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingParam.as.J.impl.F.type.aad6ec.1 = name_ref F, @FMissingParam.as.J.impl.%FMissingParam.as.J.impl.F.decl.loc117_29.1 [concrete = constants.%FMissingParam.as.J.impl.F.dd09dc.1]
// CHECK:STDOUT:   %FMissingParam.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.1(%b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingImplicitParam.as.J.impl.F.loc130_26.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.3f0)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingImplicitParam.as.J.impl.F.type.f666a8.1 = name_ref F, @FMissingImplicitParam.as.J.impl.%FMissingImplicitParam.as.J.impl.F.decl.loc130_26.1 [concrete = constants.%FMissingImplicitParam.as.J.impl.F.620ed1.1]
// CHECK:STDOUT:   %FMissingImplicitParam.as.J.impl.F.call: init bool = call %F.ref(%b.param)
// CHECK:STDOUT:   return %FMissingImplicitParam.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc148_30.1(%self.param: bool, %b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FMissingReturnType.as.J.impl.F.loc148_30.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc148_30.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.c42)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FMissingReturnType.as.J.impl.F.type.9cd56a.1 = name_ref F, @FMissingReturnType.as.J.impl.%FMissingReturnType.as.J.impl.F.decl.loc148_30.1 [concrete = constants.%FMissingReturnType.as.J.impl.F.6975be.1]
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %FMissingReturnType.as.J.impl.F.call: init %empty_tuple.type = call %FMissingReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc148: bool = converted %FMissingReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc163_38.1(%self.param: bool, %b.param: %FDifferentParamType) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentParamType.as.J.impl.F.loc163_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc163_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.47a)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentParamType.as.J.impl.F.type.4a2e9b.1 = name_ref F, @FDifferentParamType.as.J.impl.%FDifferentParamType.as.J.impl.F.decl.loc163_38.1 [concrete = constants.%FDifferentParamType.as.J.impl.F.28602a.1]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%FDifferentParamType)> [concrete = constants.%ImplicitAs.type.4ba]
// CHECK:STDOUT:   %.4: %ImplicitAs.assoc_type.503 = specific_constant imports.%Core.import_ref.484, @ImplicitAs.WithSelf(constants.%FDifferentParamType, constants.%Self.294) [concrete = constants.%assoc0.603]
// CHECK:STDOUT:   %Convert.ref: %ImplicitAs.assoc_type.503 = name_ref Convert, %.4 [concrete = constants.%assoc0.603]
// CHECK:STDOUT:   %.5: %FDifferentParamType = converted %b.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentParamType.as.J.impl.F.call: init bool = call %FDifferentParamType.as.J.impl.F.bound(%self.param, <error>)
// CHECK:STDOUT:   return %FDifferentParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc176_32.1(%self.param: %FDifferentImplicitParamType, %b.param: bool) -> out %return.param: bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentImplicitParamType.as.J.impl.F.loc176_32.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc176_32.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.929)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentImplicitParamType.as.J.impl.F.type.6ef8a6.1 = name_ref F, @FDifferentImplicitParamType.as.J.impl.%FDifferentImplicitParamType.as.J.impl.F.decl.loc176_32.1 [concrete = constants.%FDifferentImplicitParamType.as.J.impl.F.792c78.1]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%FDifferentImplicitParamType)> [concrete = constants.%ImplicitAs.type.de7]
// CHECK:STDOUT:   %.4: %ImplicitAs.assoc_type.36e = specific_constant imports.%Core.import_ref.484, @ImplicitAs.WithSelf(constants.%FDifferentImplicitParamType, constants.%Self.294) [concrete = constants.%assoc0.769]
// CHECK:STDOUT:   %Convert.ref: %ImplicitAs.assoc_type.36e = name_ref Convert, %.4 [concrete = constants.%assoc0.769]
// CHECK:STDOUT:   %.5: %FDifferentImplicitParamType = converted %self.param, <error> [concrete = <error>]
// CHECK:STDOUT:   %FDifferentImplicitParamType.as.J.impl.F.call: init bool = call %FDifferentImplicitParamType.as.J.impl.F.bound(<error>, %b.param)
// CHECK:STDOUT:   return %FDifferentImplicitParamType.as.J.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc194_38.1(%self.param: bool, %b.param: bool) -> out %return.param: %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @FDifferentReturnType.as.J.impl.F.loc194_38.2(%self.param: bool, %b.param: bool) -> out %return.param: bool [thunk @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc194_38.1 for @J.WithSelf.%J.WithSelf.F.decl, @J.WithSelf.F(constants.%J.facet.b85)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %FDifferentReturnType.as.J.impl.F.type.ff55f4.1 = name_ref F, @FDifferentReturnType.as.J.impl.%FDifferentReturnType.as.J.impl.F.decl.loc194_38.1 [concrete = constants.%FDifferentReturnType.as.J.impl.F.9725a5.1]
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   %.loc194_38.1: ref %FDifferentReturnType = temporary_storage
// CHECK:STDOUT:   %FDifferentReturnType.as.J.impl.F.call: init %FDifferentReturnType to %.loc194_38.1 = call %FDifferentReturnType.as.J.impl.F.bound(%self.param, %b.param)
// CHECK:STDOUT:   %.loc194_38.2: bool = converted %FDifferentReturnType.as.J.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SelfNested.WithSelf.F(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.e81)]
// CHECK:STDOUT:   %Self.as_type.loc199_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc199_16.1 (constants.%Self.as_type.8ab)]
// CHECK:STDOUT:   %ptr.loc199_16.1: type = ptr_type %Self.as_type.loc199_16.1 [symbolic = %ptr.loc199_16.1 (constants.%ptr.a16)]
// CHECK:STDOUT:   %struct_type.x.y.loc199_37.1: type = struct_type {.x: @SelfNested.WithSelf.F.%Self.as_type.loc199_16.1 (%Self.as_type.8ab), .y: %i32} [symbolic = %struct_type.x.y.loc199_37.1 (constants.%struct_type.x.y.4c5)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%ptr.loc199_16.1, %struct_type.x.y.loc199_37.1) [symbolic = %tuple (constants.%tuple.b6a)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc199_16.1, %struct_type.x.y.loc199_37.1) [symbolic = %tuple.type (constants.%tuple.type.c49)]
// CHECK:STDOUT:   %pattern_type.loc199_9: type = pattern_type %tuple.type [symbolic = %pattern_type.loc199_9 (constants.%pattern_type.60a)]
// CHECK:STDOUT:   %x.param_patt.loc199_9.2: @SelfNested.WithSelf.F.%pattern_type.loc199_9 (%pattern_type.60a) = value_param_pattern [symbolic = %x.param_patt.loc199_9.2 (constants.%x.param_patt.e85)]
// CHECK:STDOUT:   %x.patt.loc199_9.2: @SelfNested.WithSelf.F.%pattern_type.loc199_9 (%pattern_type.60a) = at_binding_pattern x, %x.param_patt.loc199_9.2 [symbolic = %x.patt.loc199_9.2 (constants.%x.patt.ad4)]
// CHECK:STDOUT:   %array_type.loc199_57.1: type = array_type constants.%int_4, %Self.as_type.loc199_16.1 [symbolic = %array_type.loc199_57.1 (constants.%array_type.4e4)]
// CHECK:STDOUT:   %.loc199_57.1: Core.Form = init_form %array_type.loc199_57.1 [symbolic = %.loc199_57.1 (constants.%.f45)]
// CHECK:STDOUT:   %pattern_type.loc199_57: type = pattern_type %array_type.loc199_57.1 [symbolic = %pattern_type.loc199_57 (constants.%pattern_type.2ab)]
// CHECK:STDOUT:   %return.param_patt.loc199_57.2: @SelfNested.WithSelf.F.%pattern_type.loc199_57 (%pattern_type.2ab) = out_param_pattern [symbolic = %return.param_patt.loc199_57.2 (constants.%return.param_patt.439)]
// CHECK:STDOUT:   %return.patt.loc199_41.2: @SelfNested.WithSelf.F.%pattern_type.loc199_57 (%pattern_type.2ab) = return_slot_pattern %return.param_patt.loc199_57.2, %array_type.loc199_57.1 [symbolic = %return.patt.loc199_41.2 (constants.%return.patt.869)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @SelfNested.WithSelf.F.%tuple.type (%tuple.type.c49)) -> out %return.param: @SelfNested.WithSelf.F.%array_type.loc199_57.1 (%array_type.4e4);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc211_87.1(%x.param: %tuple.type.e57) -> out %return.param: %array_type.291;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadParam.as.SelfNested.impl.F.loc211_87.2(%x.param: %tuple.type.9ab) -> out %return.param: %array_type.291 [thunk @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc211_87.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.fc6)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadParam.as.SelfNested.impl.F.type.4325e7.1 = name_ref F, @SelfNestedBadParam.as.SelfNested.impl.%SelfNestedBadParam.as.SelfNested.impl.F.decl.loc211_87.1 [concrete = constants.%SelfNestedBadParam.as.SelfNested.impl.F.5fd896.1]
// CHECK:STDOUT:   %.3: ref %array_type.291 = splice_block %return.param {}
// CHECK:STDOUT:   %tuple.elem0: %ptr.52e = tuple_access %x.param, element0
// CHECK:STDOUT:   %tuple.elem1: %struct_type.x.y.da2 = tuple_access %x.param, element1
// CHECK:STDOUT:   %.4: %SelfNestedBadParam = struct_access %tuple.elem1, element0
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32)> [concrete = constants.%ImplicitAs.type.914]
// CHECK:STDOUT:   %.5: %ImplicitAs.assoc_type.e23 = specific_constant imports.%Core.import_ref.484, @ImplicitAs.WithSelf(constants.%i32, constants.%Self.294) [concrete = constants.%assoc0.368]
// CHECK:STDOUT:   %Convert.ref: %ImplicitAs.assoc_type.e23 = name_ref Convert, %.5 [concrete = constants.%assoc0.368]
// CHECK:STDOUT:   %.6: %i32 = converted %.4, <error> [concrete = <error>]
// CHECK:STDOUT:   %SelfNestedBadParam.as.SelfNested.impl.F.call: init %array_type.291 to %.3 = call %F.ref(<error>)
// CHECK:STDOUT:   return %SelfNestedBadParam.as.SelfNested.impl.F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc227_112.1(%x.param: %tuple.type.829) -> out %return.param: %array_type.291;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfNestedBadReturnType.as.SelfNested.impl.F.loc227_112.2(%x.param: %tuple.type.829) -> out %return.param: %array_type.a08 [thunk @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc227_112.1 for @SelfNested.WithSelf.%SelfNested.WithSelf.F.decl, @SelfNested.WithSelf.F(constants.%SelfNested.facet.68c)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %SelfNestedBadReturnType.as.SelfNested.impl.F.type.1cbabe.1 = name_ref F, @SelfNestedBadReturnType.as.SelfNested.impl.%SelfNestedBadReturnType.as.SelfNested.impl.F.decl.loc227_112.1 [concrete = constants.%SelfNestedBadReturnType.as.SelfNested.impl.F.5a7d54.1]
// CHECK:STDOUT:   %.loc227_112.1: ref %array_type.291 = temporary_storage
// CHECK:STDOUT:   %SelfNestedBadReturnType.as.SelfNested.impl.F.call: init %array_type.291 to %.loc227_112.1 = call %F.ref(%x.param)
// CHECK:STDOUT:   %.loc227_112.2: %array_type.a08 = converted %SelfNestedBadReturnType.as.SelfNested.impl.F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.37e
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.dcc
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.ec6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.752) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.752
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.3d3
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.38a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.010) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.010
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.642
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.c29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.f6f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.f6f
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.751
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.6d6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.657) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.657
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.255
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.fc1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.657) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.334) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.334
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.6ff
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.39e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.334) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet.dde) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%I.facet.dde
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.fea
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.338
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%I.facet.dde) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%Self.653) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.653
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.a0a
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.f69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%Self.653) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.d77) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.d77
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.51b
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.ed6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.d77) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.3f0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.3f0
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.e23
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.897
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.3f0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.c42) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.c42
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.f86
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.d52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.c42) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.47a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.47a
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.9c3
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.4cd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.47a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.929) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.929
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.97c
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.94b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.929) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf(constants.%J.facet.b85) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%J.facet.b85
// CHECK:STDOUT:   %J.WithSelf.F.type => constants.%J.WithSelf.F.type.a9c
// CHECK:STDOUT:   %J.WithSelf.F => constants.%J.WithSelf.F.6ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J.WithSelf.F(constants.%J.facet.b85) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%Self.e81) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.e81
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.702
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.eec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%Self.e81) {
// CHECK:STDOUT:   %Self => constants.%Self.e81
// CHECK:STDOUT:   %Self.as_type.loc199_16.1 => constants.%Self.as_type.8ab
// CHECK:STDOUT:   %ptr.loc199_16.1 => constants.%ptr.a16
// CHECK:STDOUT:   %struct_type.x.y.loc199_37.1 => constants.%struct_type.x.y.4c5
// CHECK:STDOUT:   %tuple => constants.%tuple.b6a
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.c49
// CHECK:STDOUT:   %pattern_type.loc199_9 => constants.%pattern_type.60a
// CHECK:STDOUT:   %x.param_patt.loc199_9.2 => constants.%x.param_patt.e85
// CHECK:STDOUT:   %x.patt.loc199_9.2 => constants.%x.patt.ad4
// CHECK:STDOUT:   %array_type.loc199_57.1 => constants.%array_type.4e4
// CHECK:STDOUT:   %.loc199_57.1 => constants.%.f45
// CHECK:STDOUT:   %pattern_type.loc199_57 => constants.%pattern_type.2ab
// CHECK:STDOUT:   %return.param_patt.loc199_57.2 => constants.%return.param_patt.439
// CHECK:STDOUT:   %return.patt.loc199_41.2 => constants.%return.patt.869
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.fc6) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.fc6
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.75d
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.379
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.fc6) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.fc6
// CHECK:STDOUT:   %Self.as_type.loc199_16.1 => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %ptr.loc199_16.1 => constants.%ptr.52e
// CHECK:STDOUT:   %struct_type.x.y.loc199_37.1 => constants.%struct_type.x.y.da2
// CHECK:STDOUT:   %tuple => constants.%tuple.09d
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.9ab
// CHECK:STDOUT:   %pattern_type.loc199_9 => constants.%pattern_type.94f
// CHECK:STDOUT:   %x.param_patt.loc199_9.2 => constants.%x.param_patt.7b9
// CHECK:STDOUT:   %x.patt.loc199_9.2 => constants.%x.patt.4b4
// CHECK:STDOUT:   %array_type.loc199_57.1 => constants.%array_type.291
// CHECK:STDOUT:   %.loc199_57.1 => constants.%.cc2
// CHECK:STDOUT:   %pattern_type.loc199_57 => constants.%pattern_type.3a0
// CHECK:STDOUT:   %return.param_patt.loc199_57.2 => constants.%return.param_patt.489
// CHECK:STDOUT:   %return.patt.loc199_41.2 => constants.%return.patt.98a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf(constants.%SelfNested.facet.68c) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.68c
// CHECK:STDOUT:   %SelfNested.WithSelf.F.type => constants.%SelfNested.WithSelf.F.type.53d
// CHECK:STDOUT:   %SelfNested.WithSelf.F => constants.%SelfNested.WithSelf.F.f39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SelfNested.WithSelf.F(constants.%SelfNested.facet.68c) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.68c
// CHECK:STDOUT:   %Self.as_type.loc199_16.1 => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %ptr.loc199_16.1 => constants.%ptr.f55
// CHECK:STDOUT:   %struct_type.x.y.loc199_37.1 => constants.%struct_type.x.y.f6a
// CHECK:STDOUT:   %tuple => constants.%tuple.96d
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.829
// CHECK:STDOUT:   %pattern_type.loc199_9 => constants.%pattern_type.5c2
// CHECK:STDOUT:   %x.param_patt.loc199_9.2 => constants.%x.param_patt.7be
// CHECK:STDOUT:   %x.patt.loc199_9.2 => constants.%x.patt.ef5
// CHECK:STDOUT:   %array_type.loc199_57.1 => constants.%array_type.a08
// CHECK:STDOUT:   %.loc199_57.1 => constants.%.442
// CHECK:STDOUT:   %pattern_type.loc199_57 => constants.%pattern_type.f0d
// CHECK:STDOUT:   %return.param_patt.loc199_57.2 => constants.%return.param_patt.039
// CHECK:STDOUT:   %return.patt.loc199_41.2 => constants.%return.patt.3c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
