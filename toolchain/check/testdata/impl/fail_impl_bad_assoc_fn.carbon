// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+13]]:5: error: cannot implicitly convert expression of type `()` to `bool` [ConversionFailure]
    // CHECK:STDERR:     fn F[self: bool](b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:5: note: type `()` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:     fn F[self: bool](b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-39]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-43]]:32: error: forming value of incomplete type `FDifferentParamType` [IncompleteTypeInValueConversion]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^~~~~~~
    fn F[self: bool](b: bool);
  }
}

// CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+3]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
// CHECK:STDERR: class FDifferentParamType {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
class FDifferentParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:22: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F[self: bool](b: Self) -> bool;
    // CHECK:STDERR:                      ^~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-58]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-62]]:20: error: forming value of incomplete type `FDifferentImplicitParamType` [IncompleteTypeInValueConversion]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                    ^~~~~~~~~~
    fn F[self: bool](b: Self) -> bool;
  }
}

// CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+3]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
// CHECK:STDERR: class FDifferentImplicitParamType {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+13]]:5: error: function returns incomplete type `FDifferentReturnType` [IncompleteTypeInFunctionReturnType]
    // CHECK:STDERR:     fn F[self: bool](b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-5]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
    // CHECK:STDERR: class FDifferentReturnType {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:31: note: return type declared here [IncompleteReturnTypeHere]
    // CHECK:STDERR:     fn F[self: bool](b: bool) -> Self;
    // CHECK:STDERR:                               ^~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-96]]:15: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](b: bool) -> Self;
  }
}

interface SelfNested {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+3]]:41: error: function returns incomplete type `array(SelfNestedBadParam, 4)` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~
  fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
}

// CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+3]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
// CHECK:STDERR: class SelfNestedBadParam {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
class SelfNestedBadParam {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+43]]:5: note: while building thunk calling this function [ThunkCallee]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-12]]:8: error: parameter has incomplete type `(SelfNestedBadParam*, {.x: SelfNestedBadParam, .y: i32})` in function definition [IncompleteTypeInFunctionParam]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-9]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
    // CHECK:STDERR: class SelfNestedBadParam {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+33]]:5: note: while building thunk calling this function [ThunkCallee]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+29]]:5: error: function returns incomplete type `array(SelfNestedBadParam, 4)` [IncompleteTypeInFunctionReturnType]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-19]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
    // CHECK:STDERR: class SelfNestedBadParam {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+23]]:56: note: return type declared here [IncompleteReturnTypeHere]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:                                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-31]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-35]]:8: error: cannot implicitly convert expression of type `SelfNestedBadParam` to `i32` [ConversionFailure]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-38]]:8: note: type `SelfNestedBadParam` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+10]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-44]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-48]]:41: error: function returns incomplete type `array(SelfNestedBadReturnType, 4)` [IncompleteTypeInFunctionReturnType]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~
    fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
  }
}

// CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+3]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
// CHECK:STDERR: class SelfNestedBadReturnType {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SelfNestedBadReturnType {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+27]]:5: note: while building thunk calling this function [ThunkCallee]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-64]]:8: error: parameter has incomplete type `(SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})` in function definition [IncompleteTypeInFunctionParam]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-9]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
    // CHECK:STDERR: class SelfNestedBadReturnType {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+17]]:5: note: while building thunk calling this function [ThunkCallee]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-74]]:8: error: forming value of incomplete type `(SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})` [IncompleteTypeInValueConversion]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-19]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
    // CHECK:STDERR: class SelfNestedBadReturnType {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-83]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [concrete]
// CHECK:STDOUT:   %I.impl_witness.901: <witness> = impl_witness @NoF.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [concrete]
// CHECK:STDOUT:   %I.impl_witness.44e: <witness> = impl_witness @FNotFunction.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.70c: type = class_type @F.25 [concrete]
// CHECK:STDOUT:   %PossiblyF.type: type = fn_type @PossiblyF [concrete]
// CHECK:STDOUT:   %PossiblyF: %PossiblyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [concrete]
// CHECK:STDOUT:   %I.impl_witness.c80: <witness> = impl_witness @FAlias.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.1f3: <witness> = impl_witness @FExtraParam.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %F.type.44ef8c.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.424e9e.1: %F.type.44ef8c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.fa4: %I.type = facet_value %FExtraParam, (%I.impl_witness.1f3) [concrete]
// CHECK:STDOUT:   %F.type.44ef8c.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.424e9e.2: %F.type.44ef8c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %I.impl_witness.a7b: <witness> = impl_witness @FExtraImplicitParam.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.8ae: type = pattern_type %FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %F.type.e1abdd.1: type = fn_type @F.4 [concrete]
// CHECK:STDOUT:   %F.6ff574.1: %F.type.e1abdd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.ee5: %I.type = facet_value %FExtraImplicitParam, (%I.impl_witness.a7b) [concrete]
// CHECK:STDOUT:   %F.type.e1abdd.2: type = fn_type @F.5 [concrete]
// CHECK:STDOUT:   %F.6ff574.2: %F.type.e1abdd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [concrete]
// CHECK:STDOUT:   %I.impl_witness.59a: <witness> = impl_witness @FExtraReturnType.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type.387: type = fn_type @F.6 [concrete]
// CHECK:STDOUT:   %F.df5: %F.type.387 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.d5c: %I.type = facet_value %FExtraReturnType, (%I.impl_witness.59a) [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.7 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.922: %J.assoc_type = assoc_entity element0, @J.%F.decl [concrete]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.4e9: <witness> = impl_witness @FMissingParam.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type.69596c.1: type = fn_type @F.8 [concrete]
// CHECK:STDOUT:   %F.738f31.1: %F.type.69596c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.94c: %J.type = facet_value %FMissingParam, (%J.impl_witness.4e9) [concrete]
// CHECK:STDOUT:   %F.type.69596c.2: type = fn_type @F.9 [concrete]
// CHECK:STDOUT:   %F.738f31.2: %F.type.69596c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [concrete]
// CHECK:STDOUT:   %J.impl_witness.841: <witness> = impl_witness @FMissingImplicitParam.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type.d97cef.1: type = fn_type @F.10 [concrete]
// CHECK:STDOUT:   %F.01de92.1: %F.type.d97cef.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.e7e: %J.type = facet_value %FMissingImplicitParam, (%J.impl_witness.841) [concrete]
// CHECK:STDOUT:   %F.type.d97cef.2: type = fn_type @F.11 [concrete]
// CHECK:STDOUT:   %F.01de92.2: %F.type.d97cef.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.e75: <witness> = impl_witness @FMissingReturnType.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type.123d7a.1: type = fn_type @F.12 [concrete]
// CHECK:STDOUT:   %F.c7d02d.1: %F.type.123d7a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.974: %J.type = facet_value %FMissingReturnType, (%J.impl_witness.e75) [concrete]
// CHECK:STDOUT:   %F.type.123d7a.2: type = fn_type @F.13 [concrete]
// CHECK:STDOUT:   %F.c7d02d.2: %F.type.123d7a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.cbe: <witness> = impl_witness @FDifferentParamType.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.b90: type = pattern_type %FDifferentParamType [concrete]
// CHECK:STDOUT:   %F.type.6b537d.1: type = fn_type @F.14 [concrete]
// CHECK:STDOUT:   %F.04313a.1: %F.type.6b537d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.37e: %J.type = facet_value %FDifferentParamType, (%J.impl_witness.cbe) [concrete]
// CHECK:STDOUT:   %F.type.6b537d.2: type = fn_type @F.15 [concrete]
// CHECK:STDOUT:   %F.04313a.2: %F.type.6b537d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %J.impl_witness.af2: <witness> = impl_witness @FDifferentImplicitParamType.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9cc: type = pattern_type %FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %F.type.d6232a.1: type = fn_type @F.16 [concrete]
// CHECK:STDOUT:   %F.886f70.1: %F.type.d6232a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.bf1: %J.type = facet_value %FDifferentImplicitParamType, (%J.impl_witness.af2) [concrete]
// CHECK:STDOUT:   %F.type.d6232a.2: type = fn_type @F.17 [concrete]
// CHECK:STDOUT:   %F.886f70.2: %F.type.d6232a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [concrete]
// CHECK:STDOUT:   %J.impl_witness.bc1: <witness> = impl_witness @FDifferentReturnType.%J.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.8ee: type = pattern_type %FDifferentReturnType [concrete]
// CHECK:STDOUT:   %F.type.d3b58f.1: type = fn_type @F.18 [concrete]
// CHECK:STDOUT:   %F.be86c9.1: %F.type.d3b58f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.67f: %J.type = facet_value %FDifferentReturnType, (%J.impl_witness.bc1) [concrete]
// CHECK:STDOUT:   %F.type.d3b58f.2: type = fn_type @F.19 [concrete]
// CHECK:STDOUT:   %F.be86c9.2: %F.type.d3b58f.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %F.type.6ed: type = fn_type @F.20 [concrete]
// CHECK:STDOUT:   %F.998: %F.type.6ed = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.beb: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.%F.decl [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.e31: <witness> = impl_witness @SelfNestedBadParam.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.4cd: type = ptr_type %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %struct_type.x.y.871: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.a7d: type = tuple_type (%ptr.4cd, %struct_type.x.y.871) [concrete]
// CHECK:STDOUT:   %pattern_type.714: type = pattern_type %tuple.type.a7d [concrete]
// CHECK:STDOUT:   %array_type.a41: type = array_type %int_4, %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %pattern_type.1f1: type = pattern_type %array_type.a41 [concrete]
// CHECK:STDOUT:   %F.type.f90165.1: type = fn_type @F.21 [concrete]
// CHECK:STDOUT:   %F.fa8d04.1: %F.type.f90165.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.608: %SelfNested.type = facet_value %SelfNestedBadParam, (%SelfNested.impl_witness.e31) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.a89: type = struct_type {.x: %SelfNestedBadParam, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.9c9: type = tuple_type (%ptr.4cd, %struct_type.x.y.a89) [concrete]
// CHECK:STDOUT:   %pattern_type.a5c: type = pattern_type %tuple.type.9c9 [concrete]
// CHECK:STDOUT:   %F.type.f90165.2: type = fn_type @F.22 [concrete]
// CHECK:STDOUT:   %F.fa8d04.2: %F.type.f90165.2 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness.d5e: <witness> = impl_witness @SelfNestedBadReturnType.%SelfNested.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.612: type = ptr_type %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %struct_type.x.y.ac5: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.eb9: type = tuple_type (%ptr.612, %struct_type.x.y.ac5) [concrete]
// CHECK:STDOUT:   %pattern_type.23f: type = pattern_type %tuple.type.eb9 [concrete]
// CHECK:STDOUT:   %F.type.0e7d1d.1: type = fn_type @F.23 [concrete]
// CHECK:STDOUT:   %F.0bc78a.1: %F.type.0e7d1d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.766: %SelfNested.type = facet_value %SelfNestedBadReturnType, (%SelfNested.impl_witness.d5e) [concrete]
// CHECK:STDOUT:   %array_type.126: type = array_type %int_4, %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %pattern_type.f56: type = pattern_type %array_type.126 [concrete]
// CHECK:STDOUT:   %F.type.0e7d1d.2: type = fn_type @F.24 [concrete]
// CHECK:STDOUT:   %F.0bc78a.2: %F.type.0e7d1d.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.7 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc99_44: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc99_44.1: type = value_of_initializer %bool.make_type.loc99_44 [concrete = bool]
// CHECK:STDOUT:     %.loc99_44.2: type = converted %bool.make_type.loc99_44, %.loc99_44.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc99_26.1: type = splice_block %.loc99_26.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc99_26: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc99_26.2: type = value_of_initializer %bool.make_type.loc99_26 [concrete = bool]
// CHECK:STDOUT:       %.loc99_26.3: type = converted %bool.make_type.loc99_26, %.loc99_26.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc99_35.1: type = splice_block %.loc99_35.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc99_35: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc99_35.2: type = value_of_initializer %bool.make_type.loc99_35 [concrete = bool]
// CHECK:STDOUT:       %.loc99_35.3: type = converted %bool.make_type.loc99_35, %.loc99_35.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.922]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2ff]
// CHECK:STDOUT:   %F.decl: %F.type.6ed = fn_decl @F.20 [concrete = constants.%F.998] {
// CHECK:STDOUT:     %x.patt: @F.20.%pattern_type.loc207_8 (%pattern_type.ced) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.20.%pattern_type.loc207_8 (%pattern_type.ced) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.20.%pattern_type.loc207_41 (%pattern_type.c7e) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.20.%pattern_type.loc207_41 (%pattern_type.c7e) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc207_50: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Self.as_type.loc207_50: type = facet_access_type %Self.ref.loc207_50 [symbolic = %Self.as_type.loc207_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:     %.loc207_50: type = converted %Self.ref.loc207_50, %Self.as_type.loc207_50 [symbolic = %Self.as_type.loc207_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:     %array_type.loc207_57.2: type = array_type %int_4, %.loc207_50 [symbolic = %array_type.loc207_57.1 (constants.%array_type.873)]
// CHECK:STDOUT:     %x.param: @F.20.%tuple.type (%tuple.type.229) = value_param call_param0
// CHECK:STDOUT:     %.loc207_38.1: type = splice_block %.loc207_38.3 [symbolic = %tuple.type (constants.%tuple.type.229)] {
// CHECK:STDOUT:       %Self.ref.loc207_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc207_16.2: type = facet_access_type %Self.ref.loc207_12 [symbolic = %Self.as_type.loc207_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc207_16: type = converted %Self.ref.loc207_12, %Self.as_type.loc207_16.2 [symbolic = %Self.as_type.loc207_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %ptr.loc207_16.2: type = ptr_type %.loc207_16 [symbolic = %ptr.loc207_16.1 (constants.%ptr.e87)]
// CHECK:STDOUT:       %Self.ref.loc207_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc207_24: type = facet_access_type %Self.ref.loc207_24 [symbolic = %Self.as_type.loc207_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %.loc207_24: type = converted %Self.ref.loc207_24, %Self.as_type.loc207_24 [symbolic = %Self.as_type.loc207_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y.loc207_37.2: type = struct_type {.x: @F.20.%Self.as_type.loc207_16.1 (%Self.as_type.e1e), .y: %i32} [symbolic = %struct_type.x.y.loc207_37.1 (constants.%struct_type.x.y.81e)]
// CHECK:STDOUT:       %.loc207_38.2: %tuple.type.24b = tuple_literal (%ptr.loc207_16.2, %struct_type.x.y.loc207_37.2)
// CHECK:STDOUT:       %.loc207_38.3: type = converted %.loc207_38.2, constants.%tuple.type.229 [symbolic = %tuple.type (constants.%tuple.type.229)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.20.%tuple.type (%tuple.type.229) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.20.%array_type.loc207_57.1 (%array_type.873) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.20.%array_type.loc207_57.1 (%array_type.873) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %SelfNested.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0.beb]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.0c9: %Self.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   witness = @NoF.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b55: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: type = class_decl @F.25 [concrete = constants.%F.70c] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FNotFunction.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.199: %Self.ref as %I.ref {
// CHECK:STDOUT:   %PossiblyF.ref: %PossiblyF.type = name_ref PossiblyF, file.%PossiblyF.decl [concrete = constants.%PossiblyF]
// CHECK:STDOUT:   %F: %PossiblyF.type = bind_alias F, file.%PossiblyF.decl [concrete = constants.%PossiblyF]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .PossiblyF = <poisoned>
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = @FAlias.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.ddd: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl.loc65_18.1: %F.type.44ef8c.1 = fn_decl @F.2 [concrete = constants.%F.424e9e.1] {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc65_13.1: type = splice_block %.loc65_13.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc65_13.2: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc65_13.3: type = converted %bool.make_type, %.loc65_13.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc65_18.2: %F.type.44ef8c.2 = fn_decl @F.3 [concrete = constants.%F.424e9e.2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl.loc65_18.1
// CHECK:STDOUT:   witness = @FExtraParam.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.698: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl.loc81_23.1: %F.type.e1abdd.1 = fn_decl @F.4 [concrete = constants.%F.6ff574.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.8ae = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8ae = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self: %FExtraImplicitParam = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc81_23.2: %F.type.e1abdd.2 = fn_decl @F.5 [concrete = constants.%F.6ff574.2] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl.loc81_23.1
// CHECK:STDOUT:   witness = @FExtraImplicitParam.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.77b: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.387 = fn_decl @F.6 [concrete = constants.%F.df5] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc95_15.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc95_15.2: type = converted %bool.make_type, %.loc95_15.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FExtraReturnType.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5cf: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl.loc113_31.1: %F.type.69596c.1 = fn_decl @F.8 [concrete = constants.%F.738f31.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc113_27: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc113_27.1: type = value_of_initializer %bool.make_type.loc113_27 [concrete = bool]
// CHECK:STDOUT:     %.loc113_27.2: type = converted %bool.make_type.loc113_27, %.loc113_27.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc113_16.1: type = splice_block %.loc113_16.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc113_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc113_16.2: type = value_of_initializer %bool.make_type.loc113_16 [concrete = bool]
// CHECK:STDOUT:       %.loc113_16.3: type = converted %bool.make_type.loc113_16, %.loc113_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc113_31.2: %F.type.69596c.2 = fn_decl @F.9 [concrete = constants.%F.738f31.2] {
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
// CHECK:STDOUT:   .F = %F.decl.loc113_31.1
// CHECK:STDOUT:   witness = @FMissingParam.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.bac: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl.loc126_26.1: %F.type.d97cef.1 = fn_decl @F.10 [concrete = constants.%F.01de92.1] {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc126_22: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc126_22.1: type = value_of_initializer %bool.make_type.loc126_22 [concrete = bool]
// CHECK:STDOUT:     %.loc126_22.2: type = converted %bool.make_type.loc126_22, %.loc126_22.1 [concrete = bool]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc126_13.1: type = splice_block %.loc126_13.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc126_13: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc126_13.2: type = value_of_initializer %bool.make_type.loc126_13 [concrete = bool]
// CHECK:STDOUT:       %.loc126_13.3: type = converted %bool.make_type.loc126_13, %.loc126_13.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc126_26.2: %F.type.d97cef.2 = fn_decl @F.11 [concrete = constants.%F.01de92.2] {
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
// CHECK:STDOUT:   .F = %F.decl.loc126_26.1
// CHECK:STDOUT:   witness = @FMissingImplicitParam.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1a7: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl.loc145_30.1: %F.type.123d7a.1 = fn_decl @F.12 [concrete = constants.%F.c7d02d.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc145_16.1: type = splice_block %.loc145_16.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc145_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc145_16.2: type = value_of_initializer %bool.make_type.loc145_16 [concrete = bool]
// CHECK:STDOUT:       %.loc145_16.3: type = converted %bool.make_type.loc145_16, %.loc145_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc145_25.1: type = splice_block %.loc145_25.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc145_25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc145_25.2: type = value_of_initializer %bool.make_type.loc145_25 [concrete = bool]
// CHECK:STDOUT:       %.loc145_25.3: type = converted %bool.make_type.loc145_25, %.loc145_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc145_30.2: %F.type.123d7a.2 = fn_decl @F.13 [concrete = constants.%F.c7d02d.2] {
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
// CHECK:STDOUT:   .F = %F.decl.loc145_30.1
// CHECK:STDOUT:   witness = @FMissingReturnType.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.f2b: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl.loc164_38.1: %F.type.6b537d.1 = fn_decl @F.14 [concrete = constants.%F.04313a.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.b90 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.b90 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc164_34: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc164_34.1: type = value_of_initializer %bool.make_type.loc164_34 [concrete = bool]
// CHECK:STDOUT:     %.loc164_34.2: type = converted %bool.make_type.loc164_34, %.loc164_34.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc164_16.1: type = splice_block %.loc164_16.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc164_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc164_16.2: type = value_of_initializer %bool.make_type.loc164_16 [concrete = bool]
// CHECK:STDOUT:       %.loc164_16.3: type = converted %bool.make_type.loc164_16, %.loc164_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: %FDifferentParamType = value_param call_param1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [concrete = constants.%FDifferentParamType]
// CHECK:STDOUT:     %b: %FDifferentParamType = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc164_38.2: %F.type.6b537d.2 = fn_decl @F.15 [concrete = constants.%F.04313a.2] {
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
// CHECK:STDOUT:   .F = %F.decl.loc164_38.1
// CHECK:STDOUT:   witness = @FDifferentParamType.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.db4: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl.loc180_38.1: %F.type.d6232a.1 = fn_decl @F.16 [concrete = constants.%F.886f70.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.9cc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9cc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc180_34: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc180_34.1: type = value_of_initializer %bool.make_type.loc180_34 [concrete = bool]
// CHECK:STDOUT:     %.loc180_34.2: type = converted %bool.make_type.loc180_34, %.loc180_34.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %FDifferentImplicitParamType = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self: %FDifferentImplicitParamType = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc180_25.1: type = splice_block %.loc180_25.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc180_25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc180_25.2: type = value_of_initializer %bool.make_type.loc180_25 [concrete = bool]
// CHECK:STDOUT:       %.loc180_25.3: type = converted %bool.make_type.loc180_25, %.loc180_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc180_38.2: %F.type.d6232a.2 = fn_decl @F.17 [concrete = constants.%F.886f70.2] {
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
// CHECK:STDOUT:   .F = %F.decl.loc180_38.1
// CHECK:STDOUT:   witness = @FDifferentImplicitParamType.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.fcc: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl.loc199_38.1: %F.type.d3b58f.1 = fn_decl @F.18 [concrete = constants.%F.be86c9.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.8ee = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8ee = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc199_16.1: type = splice_block %.loc199_16.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc199_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc199_16.2: type = value_of_initializer %bool.make_type.loc199_16 [concrete = bool]
// CHECK:STDOUT:       %.loc199_16.3: type = converted %bool.make_type.loc199_16, %.loc199_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc199_25.1: type = splice_block %.loc199_25.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc199_25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc199_25.2: type = value_of_initializer %bool.make_type.loc199_25 [concrete = bool]
// CHECK:STDOUT:       %.loc199_25.3: type = converted %bool.make_type.loc199_25, %.loc199_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param call_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc199_38.2: %F.type.d3b58f.2 = fn_decl @F.19 [concrete = constants.%F.be86c9.2] {
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
// CHECK:STDOUT:   .F = %F.decl.loc199_38.1
// CHECK:STDOUT:   witness = @FDifferentReturnType.%J.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6a5: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl.loc258_87.1: %F.type.f90165.1 = fn_decl @F.21 [concrete = constants.%F.fa8d04.1] {
// CHECK:STDOUT:     %x.patt: %pattern_type.714 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.714 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1f1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1f1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc258_65: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref.loc258_65 [concrete = constants.%array_type.a41]
// CHECK:STDOUT:     %x.param: %tuple.type.a7d = value_param call_param0
// CHECK:STDOUT:     %.loc258_53.1: type = splice_block %.loc258_53.3 [concrete = constants.%tuple.type.a7d] {
// CHECK:STDOUT:       %SelfNestedBadParam.ref.loc258_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadParam.ref.loc258_14 [concrete = constants.%ptr.4cd]
// CHECK:STDOUT:       %int_32.loc258_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc258_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc258_49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc258_49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete = constants.%struct_type.x.y.871]
// CHECK:STDOUT:       %.loc258_53.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc258_53.3: type = converted %.loc258_53.2, constants.%tuple.type.a7d [concrete = constants.%tuple.type.a7d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a7d = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.a41 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.a41 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc258_87.2: %F.type.f90165.2 = fn_decl @F.22 [concrete = constants.%F.fa8d04.2] {
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
// CHECK:STDOUT:   .F = %F.decl.loc258_87.1
// CHECK:STDOUT:   witness = @SelfNestedBadParam.%SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.bfc: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl.loc294_112.1: %F.type.0e7d1d.1 = fn_decl @F.23 [concrete = constants.%F.0bc78a.1] {
// CHECK:STDOUT:     %x.patt: %pattern_type.23f = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.23f = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.1f1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1f1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref [concrete = constants.%array_type.a41]
// CHECK:STDOUT:     %x.param: %tuple.type.eb9 = value_param call_param0
// CHECK:STDOUT:     %.loc294_78.1: type = splice_block %.loc294_78.3 [concrete = constants.%tuple.type.eb9] {
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc294_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadReturnType.ref.loc294_14 [concrete = constants.%ptr.612]
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc294_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete = constants.%struct_type.x.y.ac5]
// CHECK:STDOUT:       %.loc294_78.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc294_78.3: type = converted %.loc294_78.2, constants.%tuple.type.eb9 [concrete = constants.%tuple.type.eb9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.eb9 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.a41 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.a41 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc294_112.2: %F.type.0e7d1d.2 = fn_decl @F.24 [concrete = constants.%F.0bc78a.2] {
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
// CHECK:STDOUT:   .F = %F.decl.loc294_112.1
// CHECK:STDOUT:   witness = @SelfNestedBadReturnType.%SelfNested.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoF {
// CHECK:STDOUT:   impl_decl @impl.0c9 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoF [concrete = constants.%NoF]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @impl.0c9 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.901]
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
// CHECK:STDOUT:   impl_decl @impl.b55 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FNotFunction [concrete = constants.%FNotFunction]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @impl.b55 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.44e]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FNotFunction
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F.25;
// CHECK:STDOUT:
// CHECK:STDOUT: class @FAlias {
// CHECK:STDOUT:   impl_decl @impl.199 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FAlias [concrete = constants.%FAlias]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @impl.199 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.c80]
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
// CHECK:STDOUT:   impl_decl @impl.ddd [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraParam [concrete = constants.%FExtraParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.ddd.%F.decl.loc65_18.2), @impl.ddd [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.1f3]
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
// CHECK:STDOUT:   impl_decl @impl.698 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.698.%F.decl.loc81_23.2), @impl.698 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.a7b]
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
// CHECK:STDOUT:   impl_decl @impl.77b [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraReturnType [concrete = constants.%FExtraReturnType]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @impl.77b [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.59a]
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
// CHECK:STDOUT:   impl_decl @impl.5cf [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingParam [concrete = constants.%FMissingParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@impl.5cf.%F.decl.loc113_31.2), @impl.5cf [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.4e9]
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
// CHECK:STDOUT:   impl_decl @impl.bac [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingImplicitParam [concrete = constants.%FMissingImplicitParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@impl.bac.%F.decl.loc126_26.2), @impl.bac [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.841]
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
// CHECK:STDOUT:   impl_decl @impl.1a7 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingReturnType [concrete = constants.%FMissingReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@impl.1a7.%F.decl.loc145_30.2), @impl.1a7 [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.e75]
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
// CHECK:STDOUT:   impl_decl @impl.f2b [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [concrete = constants.%FDifferentParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@impl.f2b.%F.decl.loc164_38.2), @impl.f2b [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.cbe]
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
// CHECK:STDOUT:   impl_decl @impl.db4 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@impl.db4.%F.decl.loc180_38.2), @impl.db4 [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.af2]
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
// CHECK:STDOUT:   impl_decl @impl.fcc [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.impl_witness_table = impl_witness_table (@impl.fcc.%F.decl.loc199_38.2), @impl.fcc [concrete]
// CHECK:STDOUT:   %J.impl_witness: <witness> = impl_witness %J.impl_witness_table [concrete = constants.%J.impl_witness.bc1]
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
// CHECK:STDOUT:   impl_decl @impl.6a5 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadParam [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (@impl.6a5.%F.decl.loc258_87.2), @impl.6a5 [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.e31]
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
// CHECK:STDOUT:   impl_decl @impl.bfc [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadReturnType [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfNested.impl_witness_table = impl_witness_table (@impl.bfc.%F.decl.loc294_112.2), @impl.bfc [concrete]
// CHECK:STDOUT:   %SelfNested.impl_witness: <witness> = impl_witness %SelfNested.impl_witness_table [concrete = constants.%SelfNested.impl_witness.d5e]
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
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PossiblyF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.4(%self.param: %FExtraImplicitParam);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call @impl.698.%F.decl.loc81_23.1(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.7(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn(%self.param: bool, %b.param: bool) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.8(%self.param: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.9(%self.param: bool, %b.param: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %self.ref, @impl.5cf.%F.decl.loc113_31.1
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.10(%b.param: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.11(%self.param: bool, %b.param: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %F.call: init bool = call @impl.bac.%F.decl.loc126_26.1(%b.ref)
// CHECK:STDOUT:   %.loc126_26.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc126_26.2: bool = converted %F.call, %.loc126_26.1
// CHECK:STDOUT:   return %.loc126_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.12(%self.param: bool, %b.param: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.13(%self.param: bool, %b.param: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %self.ref, @impl.1a7.%F.decl.loc145_30.1
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(%self.ref, %b.ref)
// CHECK:STDOUT:   %.loc145: bool = converted %F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.14(%self.param: bool, %b.param: %FDifferentParamType) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.15(%self.param: bool, %b.param: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %self.ref, @impl.f2b.%F.decl.loc164_38.1
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %F.call: init bool = call %F.bound(%self.ref, <error>)
// CHECK:STDOUT:   %.loc164_38.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc164_38.2: bool = converted %F.call, %.loc164_38.1
// CHECK:STDOUT:   return %.loc164_38.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.16(%self.param: %FDifferentImplicitParamType, %b.param: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.17(%self.param: bool, %b.param: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %self.ref, @impl.db4.%F.decl.loc180_38.1
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %F.call: init bool = call %F.bound(<error>, %b.ref)
// CHECK:STDOUT:   %.loc180_38.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc180_38.2: bool = converted %F.call, %.loc180_38.1
// CHECK:STDOUT:   return %.loc180_38.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.18(%self.param: bool, %b.param: bool) -> %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.19(%self.param: bool, %b.param: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: bool = name_ref self, %self.param
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %self.ref, @impl.fcc.%F.decl.loc199_38.1
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b.param
// CHECK:STDOUT:   %F.call: init <error> = call %F.bound(%self.ref, %b.ref)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.20(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:   %Self.as_type.loc207_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc207_16.1 (constants.%Self.as_type.e1e)]
// CHECK:STDOUT:   %ptr.loc207_16.1: type = ptr_type %Self.as_type.loc207_16.1 [symbolic = %ptr.loc207_16.1 (constants.%ptr.e87)]
// CHECK:STDOUT:   %struct_type.x.y.loc207_37.1: type = struct_type {.x: @F.20.%Self.as_type.loc207_16.1 (%Self.as_type.e1e), .y: %i32} [symbolic = %struct_type.x.y.loc207_37.1 (constants.%struct_type.x.y.81e)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%ptr.loc207_16.1, %struct_type.x.y.loc207_37.1) [symbolic = %tuple.type (constants.%tuple.type.229)]
// CHECK:STDOUT:   %pattern_type.loc207_8: type = pattern_type %tuple.type [symbolic = %pattern_type.loc207_8 (constants.%pattern_type.ced)]
// CHECK:STDOUT:   %array_type.loc207_57.1: type = array_type constants.%int_4, %Self.as_type.loc207_16.1 [symbolic = %array_type.loc207_57.1 (constants.%array_type.873)]
// CHECK:STDOUT:   %pattern_type.loc207_41: type = pattern_type %array_type.loc207_57.1 [symbolic = %pattern_type.loc207_41 (constants.%pattern_type.c7e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.20.%tuple.type (%tuple.type.229)) -> @F.20.%array_type.loc207_57.1 (%array_type.873);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.21(%x.param: %tuple.type.a7d) -> %return.param: %array_type.a41;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.22(%x.param: %tuple.type.9c9) -> %return.param: %array_type.a41 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %tuple.type.9c9 = name_ref x, %x.param
// CHECK:STDOUT:   %tuple.elem0: %ptr.4cd = tuple_access %x.ref, element0
// CHECK:STDOUT:   %tuple.elem1: %struct_type.x.y.a89 = tuple_access %x.ref, element1
// CHECK:STDOUT:   %.loc207_9.1: %SelfNestedBadParam = struct_access %tuple.elem1, element0
// CHECK:STDOUT:   %.loc207_9.2: %i32 = converted %.loc207_9.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init <error> = call @impl.6a5.%F.decl.loc258_87.1(<error>)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.23(%x.param: %tuple.type.eb9) -> %return.param: %array_type.a41;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.24(%x.param: %tuple.type.eb9) -> %array_type.126 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %tuple.type.eb9 = name_ref x, %x.param
// CHECK:STDOUT:   %.loc294: ref %array_type.a41 = temporary_storage
// CHECK:STDOUT:   %F.call: init %array_type.a41 = call @impl.bfc.%F.decl.loc294_112.1(<error>) to %.loc294
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.fa4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.ee5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.d5c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.7(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.7(constants.%J.facet.94c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.7(constants.%J.facet.e7e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.7(constants.%J.facet.974) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.7(constants.%J.facet.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.7(constants.%J.facet.bf1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.7(constants.%J.facet.67f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.20(constants.%Self.2ff) {
// CHECK:STDOUT:   %Self => constants.%Self.2ff
// CHECK:STDOUT:   %Self.as_type.loc207_16.1 => constants.%Self.as_type.e1e
// CHECK:STDOUT:   %ptr.loc207_16.1 => constants.%ptr.e87
// CHECK:STDOUT:   %struct_type.x.y.loc207_37.1 => constants.%struct_type.x.y.81e
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.229
// CHECK:STDOUT:   %pattern_type.loc207_8 => constants.%pattern_type.ced
// CHECK:STDOUT:   %array_type.loc207_57.1 => constants.%array_type.873
// CHECK:STDOUT:   %pattern_type.loc207_41 => constants.%pattern_type.c7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.20(constants.%SelfNested.facet.608) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.608
// CHECK:STDOUT:   %Self.as_type.loc207_16.1 => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %ptr.loc207_16.1 => constants.%ptr.4cd
// CHECK:STDOUT:   %struct_type.x.y.loc207_37.1 => constants.%struct_type.x.y.a89
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.9c9
// CHECK:STDOUT:   %pattern_type.loc207_8 => constants.%pattern_type.a5c
// CHECK:STDOUT:   %array_type.loc207_57.1 => constants.%array_type.a41
// CHECK:STDOUT:   %pattern_type.loc207_41 => constants.%pattern_type.1f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.20(constants.%SelfNested.facet.766) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.766
// CHECK:STDOUT:   %Self.as_type.loc207_16.1 => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %ptr.loc207_16.1 => constants.%ptr.612
// CHECK:STDOUT:   %struct_type.x.y.loc207_37.1 => constants.%struct_type.x.y.ac5
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.eb9
// CHECK:STDOUT:   %pattern_type.loc207_8 => constants.%pattern_type.23f
// CHECK:STDOUT:   %array_type.loc207_57.1 => constants.%array_type.126
// CHECK:STDOUT:   %pattern_type.loc207_41 => constants.%pattern_type.f56
// CHECK:STDOUT: }
// CHECK:STDOUT:
