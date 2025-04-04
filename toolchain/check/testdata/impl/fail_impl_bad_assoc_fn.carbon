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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
    // CHECK:STDERR:     fn F(b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-47]]:15: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F(b: bool);
  }
}

class FExtraImplicitParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: redeclaration differs because of implicit parameter list [RedeclParamListDiffers]
    // CHECK:STDERR:     fn F[self: Self]();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-60]]:15: note: previously declared without implicit parameter list [RedeclParamListPrevious]
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-74]]:15: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F() -> bool;
  }
}

interface J { fn F[self: bool](b: bool) -> bool; }

class FMissingParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: redeclaration differs because of parameter count of 0 [RedeclParamCountDiffers]
    // CHECK:STDERR:     fn F[self: bool]() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-7]]:15: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool]() -> bool;
  }
}

class FMissingImplicitParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: redeclaration differs because of missing implicit parameter list [RedeclParamListDiffers]
    // CHECK:STDERR:     fn F(b: bool) -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-20]]:15: note: previously declared with implicit parameter list [RedeclParamListPrevious]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(b: bool) -> bool;
  }
}

class FMissingReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: function redeclaration differs because no return type is provided [FunctionRedeclReturnTypeDiffersNoReturn]
    // CHECK:STDERR:     fn F[self: bool](b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-33]]:15: note: previously declared with return type `bool` [FunctionRedeclReturnTypePrevious]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](b: bool);
  }
}

class FDifferentParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:22: error: type `FDifferentParamType` of parameter 1 in redeclaration differs from previous parameter type `bool` [RedeclParamDiffersType]
    // CHECK:STDERR:     fn F[self: bool](b: Self) -> bool;
    // CHECK:STDERR:                      ^~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-46]]:32: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](b: Self) -> bool;
  }
}

class FDifferentImplicitParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: error: type `FDifferentImplicitParamType` of implicit parameter 1 in redeclaration differs from previous parameter type `bool` [RedeclParamDiffersType]
    // CHECK:STDERR:     fn F[self: Self](b: bool) -> bool;
    // CHECK:STDERR:          ^~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-59]]:20: note: previous declaration's corresponding implicit parameter here [RedeclParamPrevious]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                    ^~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: Self](b: bool) -> bool;
  }
}

class FDifferentReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: function redeclaration differs because return type is `FDifferentReturnType` [FunctionRedeclReturnTypeDiffers]
    // CHECK:STDERR:     fn F[self: bool](b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-72]]:15: note: previously declared with return type `bool` [FunctionRedeclReturnTypePrevious]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](b: bool) -> Self;
  }
}

// TODO: This should probably be permitted.
class FDifferentParamName {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:22: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
    // CHECK:STDERR:     fn F[self: bool](not_b: bool) -> bool;
    // CHECK:STDERR:                      ^~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-86]]:32: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](not_b: bool) -> bool;
  }
}

interface SelfNested {
  fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
}

class SelfNestedBadParam {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: error: type `(SelfNestedBadParam*, {.x: i32, .y: i32})` of parameter 1 in redeclaration differs from previous parameter type `(SelfNestedBadParam*, {.x: SelfNestedBadParam, .y: i32})` [RedeclParamDiffersType]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-8]]:8: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> array(Self, 4);
    // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> array(SelfNestedBadParam, 4);
  }
}

class SelfNestedBadReturnType {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: function redeclaration differs because return type is `array(SelfNestedBadParam, 4)` [FunctionRedeclReturnTypeDiffers]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> array(SelfNestedBadParam, 4);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-21]]:3: note: previously declared with return type `array(SelfNestedBadReturnType, 4)` [FunctionRedeclReturnTypePrevious]
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
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.1: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.2: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.70c: type = class_type @F.16 [concrete]
// CHECK:STDOUT:   %PossiblyF.type: type = fn_type @PossiblyF [concrete]
// CHECK:STDOUT:   %PossiblyF: %PossiblyF.type = struct_value () [concrete]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.3: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.4: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.44e: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.424: %F.type.44e = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.bbf: %I.type = facet_value %FExtraParam, (%impl_witness.85bcb7.4) [concrete]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.5: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.type.e1a: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.6ff: %F.type.e1a = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.f18: %I.type = facet_value %FExtraImplicitParam, (%impl_witness.85bcb7.5) [concrete]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.6: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.type.387: type = fn_type @F.4 [concrete]
// CHECK:STDOUT:   %F.df5: %F.type.387 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.e05: %I.type = facet_value %FExtraReturnType, (%impl_witness.85bcb7.6) [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.5 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type @J [concrete]
// CHECK:STDOUT:   %assoc0.922: %J.assoc_type = assoc_entity element0, @J.%F.decl [concrete]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.7: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.type.695: type = fn_type @F.6 [concrete]
// CHECK:STDOUT:   %F.738: %F.type.695 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.746: %J.type = facet_value %FMissingParam, (%impl_witness.85bcb7.7) [concrete]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.8: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.type.d97: type = fn_type @F.7 [concrete]
// CHECK:STDOUT:   %F.01d: %F.type.d97 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.300: %J.type = facet_value %FMissingImplicitParam, (%impl_witness.85bcb7.8) [concrete]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.9: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.type.123: type = fn_type @F.8 [concrete]
// CHECK:STDOUT:   %F.c7d: %F.type.123 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.c9e: %J.type = facet_value %FMissingReturnType, (%impl_witness.85bcb7.9) [concrete]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.10: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.type.6b5: type = fn_type @F.9 [concrete]
// CHECK:STDOUT:   %F.043: %F.type.6b5 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.ee9: %J.type = facet_value %FDifferentParamType, (%impl_witness.85bcb7.10) [concrete]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.11: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.type.d62: type = fn_type @F.10 [concrete]
// CHECK:STDOUT:   %F.886: %F.type.d62 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.5d6: %J.type = facet_value %FDifferentImplicitParamType, (%impl_witness.85bcb7.11) [concrete]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.12: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.type.d3b: type = fn_type @F.11 [concrete]
// CHECK:STDOUT:   %F.be8: %F.type.d3b = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.1f5: %J.type = facet_value %FDifferentReturnType, (%impl_witness.85bcb7.12) [concrete]
// CHECK:STDOUT:   %FDifferentParamName: type = class_type @FDifferentParamName [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.13: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %F.type.d19: type = fn_type @F.12 [concrete]
// CHECK:STDOUT:   %F.669: %F.type.d19 = struct_value () [concrete]
// CHECK:STDOUT:   %J.facet.19c: %J.type = facet_value %FDifferentParamName, (%impl_witness.85bcb7.13) [concrete]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [concrete]
// CHECK:STDOUT:   %Self.2ff: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.2ff [symbolic]
// CHECK:STDOUT:   %ptr.e87: type = ptr_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.81e: type = struct_type {.x: %Self.as_type, .y: %i32} [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.229: type = tuple_type (%ptr.e87, %struct_type.x.y.81e) [symbolic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.873: type = array_type %int_4, %Self.as_type [symbolic]
// CHECK:STDOUT:   %F.type.6ed: type = fn_type @F.13 [concrete]
// CHECK:STDOUT:   %F.998: %F.type.6ed = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.assoc_type: type = assoc_entity_type @SelfNested [concrete]
// CHECK:STDOUT:   %assoc0.beb: %SelfNested.assoc_type = assoc_entity element0, @SelfNested.%F.decl [concrete]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.14: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %ptr.4cd: type = ptr_type %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %struct_type.x.y.871: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.a7d: type = tuple_type (%ptr.4cd, %struct_type.x.y.871) [concrete]
// CHECK:STDOUT:   %array_type.a41: type = array_type %int_4, %SelfNestedBadParam [concrete]
// CHECK:STDOUT:   %F.type.f90: type = fn_type @F.14 [concrete]
// CHECK:STDOUT:   %F.fa8: %F.type.f90 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.46a: %SelfNested.type = facet_value %SelfNestedBadParam, (%impl_witness.85bcb7.14) [concrete]
// CHECK:STDOUT:   %struct_type.x.y.a89: type = struct_type {.x: %SelfNestedBadParam, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.9c9: type = tuple_type (%ptr.4cd, %struct_type.x.y.a89) [concrete]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %impl_witness.85bcb7.15: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %ptr.612: type = ptr_type %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT:   %struct_type.x.y.ac5: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete]
// CHECK:STDOUT:   %tuple.type.eb9: type = tuple_type (%ptr.612, %struct_type.x.y.ac5) [concrete]
// CHECK:STDOUT:   %F.type.0e7: type = fn_type @F.15 [concrete]
// CHECK:STDOUT:   %F.0bc: %F.type.0e7 = struct_value () [concrete]
// CHECK:STDOUT:   %SelfNested.facet.e62: %SelfNested.type = facet_value %SelfNestedBadReturnType, (%impl_witness.85bcb7.15) [concrete]
// CHECK:STDOUT:   %array_type.126: type = array_type %int_4, %SelfNestedBadReturnType [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     .FDifferentParamName = %FDifferentParamName.decl
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
// CHECK:STDOUT:   %FDifferentParamName.decl: type = class_decl @FDifferentParamName [concrete = constants.%FDifferentParamName] {} {}
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
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.5 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc93_44: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc93_44.1: type = value_of_initializer %bool.make_type.loc93_44 [concrete = bool]
// CHECK:STDOUT:     %.loc93_44.2: type = converted %bool.make_type.loc93_44, %.loc93_44.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc93_26.1: type = splice_block %.loc93_26.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc93_26: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc93_26.2: type = value_of_initializer %bool.make_type.loc93_26 [concrete = bool]
// CHECK:STDOUT:       %.loc93_26.3: type = converted %bool.make_type.loc93_26, %.loc93_26.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc93_35.1: type = splice_block %.loc93_35.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc93_35: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc93_35.2: type = value_of_initializer %bool.make_type.loc93_35 [concrete = bool]
// CHECK:STDOUT:       %.loc93_35.3: type = converted %bool.make_type.loc93_35, %.loc93_35.2 [concrete = bool]
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
// CHECK:STDOUT:   %F.decl: %F.type.6ed = fn_decl @F.13 [concrete = constants.%F.998] {
// CHECK:STDOUT:     %x.patt: @F.13.%tuple.type (%tuple.type.229) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.13.%tuple.type (%tuple.type.229) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.13.%array_type.loc188_57.1 (%array_type.873) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.13.%array_type.loc188_57.1 (%array_type.873) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc188_50: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %Self.as_type.loc188_50: type = facet_access_type %Self.ref.loc188_50 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc188_50: type = converted %Self.ref.loc188_50, %Self.as_type.loc188_50 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %array_type.loc188_57.2: type = array_type %int_4, %.loc188_50 [symbolic = %array_type.loc188_57.1 (constants.%array_type.873)]
// CHECK:STDOUT:     %x.param: @F.13.%tuple.type (%tuple.type.229) = value_param call_param0
// CHECK:STDOUT:     %.loc188_38.1: type = splice_block %.loc188_38.3 [symbolic = %tuple.type (constants.%tuple.type.229)] {
// CHECK:STDOUT:       %Self.ref.loc188_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc188_16.2: type = facet_access_type %Self.ref.loc188_12 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc188_16: type = converted %Self.ref.loc188_12, %Self.as_type.loc188_16.2 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %ptr.loc188_16.2: type = ptr_type %.loc188_16 [symbolic = %ptr.loc188_16.1 (constants.%ptr.e87)]
// CHECK:STDOUT:       %Self.ref.loc188_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:       %Self.as_type.loc188_24: type = facet_access_type %Self.ref.loc188_24 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc188_24: type = converted %Self.ref.loc188_24, %Self.as_type.loc188_24 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y.loc188_37.2: type = struct_type {.x: %Self.as_type, .y: %i32} [symbolic = %struct_type.x.y.loc188_37.1 (constants.%struct_type.x.y.81e)]
// CHECK:STDOUT:       %.loc188_38.2: %tuple.type.24b = tuple_literal (%ptr.loc188_16.2, %struct_type.x.y.loc188_37.2)
// CHECK:STDOUT:       %.loc188_38.3: type = converted %.loc188_38.2, constants.%tuple.type.229 [symbolic = %tuple.type (constants.%tuple.type.229)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.13.%tuple.type (%tuple.type.229) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.13.%array_type.loc188_57.1 (%array_type.873) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.13.%array_type.loc188_57.1 (%array_type.873) = return_slot %return.param
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
// CHECK:STDOUT:   witness = @NoF.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b55: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: type = class_decl @F.16 [concrete = constants.%F.70c] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FNotFunction.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.199: %Self.ref as %I.ref {
// CHECK:STDOUT:   %PossiblyF.ref: %PossiblyF.type = name_ref PossiblyF, file.%PossiblyF.decl [concrete = constants.%PossiblyF]
// CHECK:STDOUT:   %F: %PossiblyF.type = bind_alias F, file.%PossiblyF.decl [concrete = constants.%PossiblyF]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .PossiblyF = <poisoned>
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = @FAlias.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.ddd: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.44e = fn_decl @F.2 [concrete = constants.%F.424] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc62_13.1: type = splice_block %.loc62_13.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc62_13.2: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc62_13.3: type = converted %bool.make_type, %.loc62_13.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FExtraParam.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.698: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.e1a = fn_decl @F.3 [concrete = constants.%F.6ff] {
// CHECK:STDOUT:     %self.patt: %FExtraImplicitParam = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %FExtraImplicitParam = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [concrete = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self: %FExtraImplicitParam = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FExtraImplicitParam.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.77b: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.387 = fn_decl @F.4 [concrete = constants.%F.df5] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc89_15.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc89_15.2: type = converted %bool.make_type, %.loc89_15.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FExtraReturnType.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5cf: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.695 = fn_decl @F.6 [concrete = constants.%F.738] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc104_27: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc104_27.1: type = value_of_initializer %bool.make_type.loc104_27 [concrete = bool]
// CHECK:STDOUT:     %.loc104_27.2: type = converted %bool.make_type.loc104_27, %.loc104_27.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc104_16.1: type = splice_block %.loc104_16.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc104_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc104_16.2: type = value_of_initializer %bool.make_type.loc104_16 [concrete = bool]
// CHECK:STDOUT:       %.loc104_16.3: type = converted %bool.make_type.loc104_16, %.loc104_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FMissingParam.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.bac: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.d97 = fn_decl @F.7 [concrete = constants.%F.01d] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc117_22: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc117_22.1: type = value_of_initializer %bool.make_type.loc117_22 [concrete = bool]
// CHECK:STDOUT:     %.loc117_22.2: type = converted %bool.make_type.loc117_22, %.loc117_22.1 [concrete = bool]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc117_13.1: type = splice_block %.loc117_13.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc117_13: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc117_13.2: type = value_of_initializer %bool.make_type.loc117_13 [concrete = bool]
// CHECK:STDOUT:       %.loc117_13.3: type = converted %bool.make_type.loc117_13, %.loc117_13.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FMissingImplicitParam.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1a7: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.123 = fn_decl @F.8 [concrete = constants.%F.c7d] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc130_16.1: type = splice_block %.loc130_16.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc130_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc130_16.2: type = value_of_initializer %bool.make_type.loc130_16 [concrete = bool]
// CHECK:STDOUT:       %.loc130_16.3: type = converted %bool.make_type.loc130_16, %.loc130_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc130_25.1: type = splice_block %.loc130_25.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc130_25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc130_25.2: type = value_of_initializer %bool.make_type.loc130_25 [concrete = bool]
// CHECK:STDOUT:       %.loc130_25.3: type = converted %bool.make_type.loc130_25, %.loc130_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FMissingReturnType.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.f2b: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.6b5 = fn_decl @F.9 [concrete = constants.%F.043] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %b.patt: %FDifferentParamType = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %FDifferentParamType = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc143_34: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc143_34.1: type = value_of_initializer %bool.make_type.loc143_34 [concrete = bool]
// CHECK:STDOUT:     %.loc143_34.2: type = converted %bool.make_type.loc143_34, %.loc143_34.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc143_16.1: type = splice_block %.loc143_16.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc143_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc143_16.2: type = value_of_initializer %bool.make_type.loc143_16 [concrete = bool]
// CHECK:STDOUT:       %.loc143_16.3: type = converted %bool.make_type.loc143_16, %.loc143_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: %FDifferentParamType = value_param call_param1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [concrete = constants.%FDifferentParamType]
// CHECK:STDOUT:     %b: %FDifferentParamType = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FDifferentParamType.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.db4: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.d62 = fn_decl @F.10 [concrete = constants.%F.886] {
// CHECK:STDOUT:     %self.patt: %FDifferentImplicitParamType = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %FDifferentImplicitParamType = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc156_34: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc156_34.1: type = value_of_initializer %bool.make_type.loc156_34 [concrete = bool]
// CHECK:STDOUT:     %.loc156_34.2: type = converted %bool.make_type.loc156_34, %.loc156_34.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: %FDifferentImplicitParamType = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [concrete = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self: %FDifferentImplicitParamType = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc156_25.1: type = splice_block %.loc156_25.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc156_25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc156_25.2: type = value_of_initializer %bool.make_type.loc156_25 [concrete = bool]
// CHECK:STDOUT:       %.loc156_25.3: type = converted %bool.make_type.loc156_25, %.loc156_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FDifferentImplicitParamType.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.fcc: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.d3b = fn_decl @F.11 [concrete = constants.%F.be8] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: %FDifferentReturnType = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %FDifferentReturnType = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [concrete = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc169_16.1: type = splice_block %.loc169_16.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc169_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc169_16.2: type = value_of_initializer %bool.make_type.loc169_16 [concrete = bool]
// CHECK:STDOUT:       %.loc169_16.3: type = converted %bool.make_type.loc169_16, %.loc169_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc169_25.1: type = splice_block %.loc169_25.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc169_25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc169_25.2: type = value_of_initializer %bool.make_type.loc169_25 [concrete = bool]
// CHECK:STDOUT:       %.loc169_25.3: type = converted %bool.make_type.loc169_25, %.loc169_25.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param call_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FDifferentReturnType.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.c5d: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.d19 = fn_decl @F.12 [concrete = constants.%F.669] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %not_b.patt: bool = binding_pattern not_b
// CHECK:STDOUT:     %not_b.param_patt: bool = value_param_pattern %not_b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc183_38: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc183_38.1: type = value_of_initializer %bool.make_type.loc183_38 [concrete = bool]
// CHECK:STDOUT:     %.loc183_38.2: type = converted %bool.make_type.loc183_38, %.loc183_38.1 [concrete = bool]
// CHECK:STDOUT:     %self.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc183_16.1: type = splice_block %.loc183_16.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc183_16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc183_16.2: type = value_of_initializer %bool.make_type.loc183_16 [concrete = bool]
// CHECK:STDOUT:       %.loc183_16.3: type = converted %bool.make_type.loc183_16, %.loc183_16.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %not_b.param: bool = value_param call_param1
// CHECK:STDOUT:     %.loc183_29.1: type = splice_block %.loc183_29.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc183_29: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc183_29.2: type = value_of_initializer %bool.make_type.loc183_29 [concrete = bool]
// CHECK:STDOUT:       %.loc183_29.3: type = converted %bool.make_type.loc183_29, %.loc183_29.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %not_b: bool = bind_name not_b, %not_b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @FDifferentParamName.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6a5: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.f90 = fn_decl @F.14 [concrete = constants.%F.fa8] {
// CHECK:STDOUT:     %x.patt: %tuple.type.a7d = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.a7d = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %array_type.a41 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type.a41 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc200_65: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref.loc200_65 [concrete = constants.%array_type.a41]
// CHECK:STDOUT:     %x.param: %tuple.type.a7d = value_param call_param0
// CHECK:STDOUT:     %.loc200_53.1: type = splice_block %.loc200_53.3 [concrete = constants.%tuple.type.a7d] {
// CHECK:STDOUT:       %SelfNestedBadParam.ref.loc200_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadParam.ref.loc200_14 [concrete = constants.%ptr.4cd]
// CHECK:STDOUT:       %int_32.loc200_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc200_40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_32.loc200_49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc200_49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete = constants.%struct_type.x.y.871]
// CHECK:STDOUT:       %.loc200_53.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc200_53.3: type = converted %.loc200_53.2, constants.%tuple.type.a7d [concrete = constants.%tuple.type.a7d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.a7d = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.a41 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.a41 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @SelfNestedBadParam.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.bfc: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.0e7 = fn_decl @F.15 [concrete = constants.%F.0bc] {
// CHECK:STDOUT:     %x.patt: %tuple.type.eb9 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.eb9 = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: %array_type.a41 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type.a41 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam.ref [concrete = constants.%array_type.a41]
// CHECK:STDOUT:     %x.param: %tuple.type.eb9 = value_param call_param0
// CHECK:STDOUT:     %.loc213_78.1: type = splice_block %.loc213_78.3 [concrete = constants.%tuple.type.eb9] {
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc213_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %ptr: type = ptr_type %SelfNestedBadReturnType.ref.loc213_14 [concrete = constants.%ptr.612]
// CHECK:STDOUT:       %SelfNestedBadReturnType.ref.loc213_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [concrete = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.x.y: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [concrete = constants.%struct_type.x.y.ac5]
// CHECK:STDOUT:       %.loc213_78.2: %tuple.type.24b = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:       %.loc213_78.3: type = converted %.loc213_78.2, constants.%tuple.type.eb9 [concrete = constants.%tuple.type.eb9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %tuple.type.eb9 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.a41 = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.a41 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SelfNestedBadReturnType = <poisoned>
// CHECK:STDOUT:   .SelfNestedBadParam = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @SelfNestedBadReturnType.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoF {
// CHECK:STDOUT:   impl_decl @impl.0c9 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoF [concrete = constants.%NoF]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.1]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FNotFunction
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F.16;
// CHECK:STDOUT:
// CHECK:STDOUT: class @FAlias {
// CHECK:STDOUT:   impl_decl @impl.199 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FAlias [concrete = constants.%FAlias]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.3]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.4]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.5]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.6]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.7]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.8]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.9]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.10]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.11]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.12]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentReturnType
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamName {
// CHECK:STDOUT:   impl_decl @impl.c5d [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamName [concrete = constants.%FDifferentParamName]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.13]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamName
// CHECK:STDOUT:   .J = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadParam {
// CHECK:STDOUT:   impl_decl @impl.6a5 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadParam [concrete = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [concrete = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.14]
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness.85bcb7.15]
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
// CHECK:STDOUT: fn @F.2(%b.param_patt: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %FExtraImplicitParam]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.4() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.5(@J.%Self: %J.type) {
// CHECK:STDOUT:   fn[%self.param_patt: bool](%b.param_patt: bool) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6[%self.param_patt: bool]() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.7(%b.param_patt: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.8[%self.param_patt: bool](%b.param_patt: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.9[%self.param_patt: bool](%b.param_patt: %FDifferentParamType) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.10[%self.param_patt: %FDifferentImplicitParamType](%b.param_patt: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.11[%self.param_patt: bool](%b.param_patt: bool) -> %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.12[%self.param_patt: bool](%not_b.param_patt: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.13(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2ff)]
// CHECK:STDOUT:   %Self.as_type.loc188_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %ptr.loc188_16.1: type = ptr_type %Self.as_type.loc188_16.1 [symbolic = %ptr.loc188_16.1 (constants.%ptr.e87)]
// CHECK:STDOUT:   %struct_type.x.y.loc188_37.1: type = struct_type {.x: @F.13.%Self.as_type.loc188_16.1 (%Self.as_type), .y: %i32} [symbolic = %struct_type.x.y.loc188_37.1 (constants.%struct_type.x.y.81e)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.13.%ptr.loc188_16.1 (%ptr.e87), @F.13.%struct_type.x.y.loc188_37.1 (%struct_type.x.y.81e)) [symbolic = %tuple.type (constants.%tuple.type.229)]
// CHECK:STDOUT:   %array_type.loc188_57.1: type = array_type constants.%int_4, %Self.as_type.loc188_16.1 [symbolic = %array_type.loc188_57.1 (constants.%array_type.873)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.13.%tuple.type (%tuple.type.229)) -> @F.13.%array_type.loc188_57.1 (%array_type.873);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.14(%x.param_patt: %tuple.type.a7d) -> %array_type.a41;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.15(%x.param_patt: %tuple.type.eb9) -> %array_type.a41;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.bbf) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.f18) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.e05) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.746) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.300) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.c9e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.ee9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.5d6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.1f5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.19c) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%Self.2ff) {
// CHECK:STDOUT:   %Self => constants.%Self.2ff
// CHECK:STDOUT:   %Self.as_type.loc188_16.1 => constants.%Self.as_type
// CHECK:STDOUT:   %ptr.loc188_16.1 => constants.%ptr.e87
// CHECK:STDOUT:   %struct_type.x.y.loc188_37.1 => constants.%struct_type.x.y.81e
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.229
// CHECK:STDOUT:   %array_type.loc188_57.1 => constants.%array_type.873
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%SelfNested.facet.46a) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.46a
// CHECK:STDOUT:   %Self.as_type.loc188_16.1 => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %ptr.loc188_16.1 => constants.%ptr.4cd
// CHECK:STDOUT:   %struct_type.x.y.loc188_37.1 => constants.%struct_type.x.y.a89
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.9c9
// CHECK:STDOUT:   %array_type.loc188_57.1 => constants.%array_type.a41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%SelfNested.facet.e62) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.e62
// CHECK:STDOUT:   %Self.as_type.loc188_16.1 => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %ptr.loc188_16.1 => constants.%ptr.612
// CHECK:STDOUT:   %struct_type.x.y.loc188_37.1 => constants.%struct_type.x.y.ac5
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.eb9
// CHECK:STDOUT:   %array_type.loc188_57.1 => constants.%array_type.126
// CHECK:STDOUT: }
// CHECK:STDOUT:
