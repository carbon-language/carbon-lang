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
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-6]]:15: note: associated function F declared here [ImplAssociatedFunctionHere]
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-18]]:15: note: associated function F declared here [ImplAssociatedFunctionHere]
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-34]]:15: note: associated function F declared here [ImplAssociatedFunctionHere]
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
  fn F(x: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
}

class SelfNestedBadParam {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: error: type `(SelfNestedBadParam*, {.x: i32, .y: i32})` of parameter 1 in redeclaration differs from previous parameter type `(SelfNestedBadParam*, {.x: SelfNestedBadParam, .y: i32})` [RedeclParamDiffersType]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> [SelfNestedBadParam; 4];
    // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-8]]:8: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
    // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> [SelfNestedBadParam; 4];
  }
}

class SelfNestedBadReturnType {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: error: function redeclaration differs because return type is `[SelfNestedBadParam; 4]` [FunctionRedeclReturnTypeDiffers]
    // CHECK:STDERR:     fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> [SelfNestedBadParam; 4];
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-21]]:3: note: previously declared with return type `[SelfNestedBadReturnType; 4]` [FunctionRedeclReturnTypePrevious]
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> [SelfNestedBadParam; 4];
  }
}

// CHECK:STDOUT: --- fail_impl_bad_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.1: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %assoc0.1: %F.assoc_type.1 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [template]
// CHECK:STDOUT:   %F.2: type = class_type @F.16 [template]
// CHECK:STDOUT:   %PossiblyF.type: type = fn_type @PossiblyF [template]
// CHECK:STDOUT:   %PossiblyF: %PossiblyF.type = struct_value () [template]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [template]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.3: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %I.facet.1: %I.type = facet_value %FExtraParam, %FExtraParam [template]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.4: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %I.facet.2: %I.type = facet_value %FExtraImplicitParam, %FExtraImplicitParam [template]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [template]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.4 [template]
// CHECK:STDOUT:   %F.5: %F.type.4 = struct_value () [template]
// CHECK:STDOUT:   %I.facet.3: %I.type = facet_value %FExtraReturnType, %FExtraReturnType [template]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %Self.2: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.5 [template]
// CHECK:STDOUT:   %F.6: %F.type.5 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.2: type = assoc_entity_type %J.type, %F.type.5 [template]
// CHECK:STDOUT:   %assoc0.2: %F.assoc_type.2 = assoc_entity element0, @J.%F.decl [template]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [template]
// CHECK:STDOUT:   %F.type.6: type = fn_type @F.6 [template]
// CHECK:STDOUT:   %F.7: %F.type.6 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.1: %J.type = facet_value %FMissingParam, %FMissingParam [template]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [template]
// CHECK:STDOUT:   %F.type.7: type = fn_type @F.7 [template]
// CHECK:STDOUT:   %F.8: %F.type.7 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.2: %J.type = facet_value %FMissingImplicitParam, %FMissingImplicitParam [template]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [template]
// CHECK:STDOUT:   %F.type.8: type = fn_type @F.8 [template]
// CHECK:STDOUT:   %F.9: %F.type.8 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.3: %J.type = facet_value %FMissingReturnType, %FMissingReturnType [template]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [template]
// CHECK:STDOUT:   %F.type.9: type = fn_type @F.9 [template]
// CHECK:STDOUT:   %F.10: %F.type.9 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.4: %J.type = facet_value %FDifferentParamType, %FDifferentParamType [template]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [template]
// CHECK:STDOUT:   %F.type.10: type = fn_type @F.10 [template]
// CHECK:STDOUT:   %F.11: %F.type.10 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.5: %J.type = facet_value %FDifferentImplicitParamType, %FDifferentImplicitParamType [template]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [template]
// CHECK:STDOUT:   %F.type.11: type = fn_type @F.11 [template]
// CHECK:STDOUT:   %F.12: %F.type.11 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.6: %J.type = facet_value %FDifferentReturnType, %FDifferentReturnType [template]
// CHECK:STDOUT:   %FDifferentParamName: type = class_type @FDifferentParamName [template]
// CHECK:STDOUT:   %F.type.12: type = fn_type @F.12 [template]
// CHECK:STDOUT:   %F.13: %F.type.12 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.7: %J.type = facet_value %FDifferentParamName, %FDifferentParamName [template]
// CHECK:STDOUT:   %SelfNested.type: type = facet_type <@SelfNested> [template]
// CHECK:STDOUT:   %Self.3: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.3 [symbolic]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %struct_type.x.y.1: type = struct_type {.x: %Self.as_type, .y: %i32} [symbolic]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%ptr.1, %struct_type.x.y.1) [symbolic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %array_type.1: type = array_type %int_4, %Self.as_type [symbolic]
// CHECK:STDOUT:   %F.type.13: type = fn_type @F.13 [template]
// CHECK:STDOUT:   %F.14: %F.type.13 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type.3: type = assoc_entity_type %SelfNested.type, %F.type.13 [template]
// CHECK:STDOUT:   %assoc0.3: %F.assoc_type.3 = assoc_entity element0, @SelfNested.%F.decl [template]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %SelfNestedBadParam [template]
// CHECK:STDOUT:   %struct_type.x.y.2: type = struct_type {.x: %i32, .y: %i32} [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%ptr.2, %struct_type.x.y.2) [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_4, %SelfNestedBadParam [template]
// CHECK:STDOUT:   %F.type.14: type = fn_type @F.14 [template]
// CHECK:STDOUT:   %F.15: %F.type.14 = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.facet.1: %SelfNested.type = facet_value %SelfNestedBadParam, %SelfNestedBadParam [template]
// CHECK:STDOUT:   %struct_type.x.y.3: type = struct_type {.x: %SelfNestedBadParam, .y: %i32} [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (%ptr.2, %struct_type.x.y.3) [template]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %SelfNestedBadReturnType [template]
// CHECK:STDOUT:   %struct_type.x.y.4: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (%ptr.3, %struct_type.x.y.4) [template]
// CHECK:STDOUT:   %F.type.15: type = fn_type @F.15 [template]
// CHECK:STDOUT:   %F.16: %F.type.15 = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.facet.2: %SelfNested.type = facet_value %SelfNestedBadReturnType, %SelfNestedBadReturnType [template]
// CHECK:STDOUT:   %array_type.3: type = array_type %int_4, %SelfNestedBadReturnType [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
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
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %NoF.decl: type = class_decl @NoF [template = constants.%NoF] {} {}
// CHECK:STDOUT:   %FNotFunction.decl: type = class_decl @FNotFunction [template = constants.%FNotFunction] {} {}
// CHECK:STDOUT:   %PossiblyF.decl: %PossiblyF.type = fn_decl @PossiblyF [template = constants.%PossiblyF] {} {}
// CHECK:STDOUT:   %FAlias.decl: type = class_decl @FAlias [template = constants.%FAlias] {} {}
// CHECK:STDOUT:   %FExtraParam.decl: type = class_decl @FExtraParam [template = constants.%FExtraParam] {} {}
// CHECK:STDOUT:   %FExtraImplicitParam.decl: type = class_decl @FExtraImplicitParam [template = constants.%FExtraImplicitParam] {} {}
// CHECK:STDOUT:   %FExtraReturnType.decl: type = class_decl @FExtraReturnType [template = constants.%FExtraReturnType] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%J.type] {} {}
// CHECK:STDOUT:   %FMissingParam.decl: type = class_decl @FMissingParam [template = constants.%FMissingParam] {} {}
// CHECK:STDOUT:   %FMissingImplicitParam.decl: type = class_decl @FMissingImplicitParam [template = constants.%FMissingImplicitParam] {} {}
// CHECK:STDOUT:   %FMissingReturnType.decl: type = class_decl @FMissingReturnType [template = constants.%FMissingReturnType] {} {}
// CHECK:STDOUT:   %FDifferentParamType.decl: type = class_decl @FDifferentParamType [template = constants.%FDifferentParamType] {} {}
// CHECK:STDOUT:   %FDifferentImplicitParamType.decl: type = class_decl @FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType] {} {}
// CHECK:STDOUT:   %FDifferentReturnType.decl: type = class_decl @FDifferentReturnType [template = constants.%FDifferentReturnType] {} {}
// CHECK:STDOUT:   %FDifferentParamName.decl: type = class_decl @FDifferentParamName [template = constants.%FDifferentParamName] {} {}
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [template = constants.%SelfNested.type] {} {}
// CHECK:STDOUT:   %SelfNestedBadParam.decl: type = class_decl @SelfNestedBadParam [template = constants.%SelfNestedBadParam] {} {}
// CHECK:STDOUT:   %SelfNestedBadReturnType.decl: type = class_decl @SelfNestedBadReturnType [template = constants.%SelfNestedBadReturnType] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type.1 = assoc_entity element0, %F.decl [template = constants.%assoc0.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: %F.type.5 = fn_decl @F.5 [template = constants.%F.6] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc93_26: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc93_26.1: type = value_of_initializer %bool.make_type.loc93_26 [template = bool]
// CHECK:STDOUT:     %.loc93_26.2: type = converted %bool.make_type.loc93_26, %.loc93_26.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc93_35: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc93_35.1: type = value_of_initializer %bool.make_type.loc93_35 [template = bool]
// CHECK:STDOUT:     %.loc93_35.2: type = converted %bool.make_type.loc93_35, %.loc93_35.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc93_44: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc93_44.1: type = value_of_initializer %bool.make_type.loc93_44 [template = bool]
// CHECK:STDOUT:     %.loc93_44.2: type = converted %bool.make_type.loc93_44, %.loc93_44.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type.2 = assoc_entity element0, %F.decl [template = constants.%assoc0.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:   %F.decl: %F.type.13 = fn_decl @F.13 [template = constants.%F.14] {
// CHECK:STDOUT:     %x.patt: @F.13.%tuple.type (%tuple.type.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.13.%tuple.type (%tuple.type.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.13.%array_type.loc188_52.1 (%array_type.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.13.%array_type.loc188_52.1 (%array_type.1) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc188_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %Self.as_type.loc188_16.2: type = facet_access_type %Self.ref.loc188_12 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc188_16: type = converted %Self.ref.loc188_12, %Self.as_type.loc188_16.2 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %ptr.loc188_16.2: type = ptr_type %Self.as_type [symbolic = %ptr.loc188_16.1 (constants.%ptr.1)]
// CHECK:STDOUT:     %Self.ref.loc188_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %Self.as_type.loc188_24: type = facet_access_type %Self.ref.loc188_24 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc188_24: type = converted %Self.ref.loc188_24, %Self.as_type.loc188_24 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc188_34.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc188_34.2: type = converted %int.make_type_signed, %.loc188_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.x.y.loc188_37.2: type = struct_type {.x: %Self.as_type, .y: %i32} [symbolic = %struct_type.x.y.loc188_37.1 (constants.%struct_type.x.y.1)]
// CHECK:STDOUT:     %.loc188_38.1: %tuple.type.1 = tuple_literal (%ptr.loc188_16.2, %struct_type.x.y.loc188_37.2)
// CHECK:STDOUT:     %.loc188_38.2: type = converted %.loc188_38.1, constants.%tuple.type.2 [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:     %Self.ref.loc188_45: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:     %Self.as_type.loc188_45: type = facet_access_type %Self.ref.loc188_45 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc188_45: type = converted %Self.ref.loc188_45, %Self.as_type.loc188_45 [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %array_type.loc188_52.2: type = array_type %int_4, %Self.as_type [symbolic = %array_type.loc188_52.1 (constants.%array_type.1)]
// CHECK:STDOUT:     %x.param: @F.13.%tuple.type (%tuple.type.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @F.13.%tuple.type (%tuple.type.2) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.13.%array_type.loc188_52.1 (%array_type.1) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.13.%array_type.loc188_52.1 (%array_type.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %F.assoc_type.3 = assoc_entity element0, %F.decl [template = constants.%assoc0.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %I.ref {
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: type = class_decl @F.16 [template = constants.%F.2] {} {}
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %Self.ref as %I.ref {
// CHECK:STDOUT:   %PossiblyF.ref: %PossiblyF.type = name_ref PossiblyF, file.%PossiblyF.decl [template = constants.%PossiblyF]
// CHECK:STDOUT:   %F: %PossiblyF.type = bind_alias F, file.%PossiblyF.decl [template = constants.%PossiblyF]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.3] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc62_13.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc62_13.2: type = converted %bool.make_type, %.loc62_13.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.4] {
// CHECK:STDOUT:     %self.patt: %FExtraImplicitParam = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %FExtraImplicitParam = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [template = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = value_param runtime_param0
// CHECK:STDOUT:     %self: %FExtraImplicitParam = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.4 = fn_decl @F.4 [template = constants.%F.5] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc89_15.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc89_15.2: type = converted %bool.make_type, %.loc89_15.1 [template = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.6 = fn_decl @F.6 [template = constants.%F.7] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc104_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc104_16.1: type = value_of_initializer %bool.make_type.loc104_16 [template = bool]
// CHECK:STDOUT:     %.loc104_16.2: type = converted %bool.make_type.loc104_16, %.loc104_16.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc104_27: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc104_27.1: type = value_of_initializer %bool.make_type.loc104_27 [template = bool]
// CHECK:STDOUT:     %.loc104_27.2: type = converted %bool.make_type.loc104_27, %.loc104_27.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.7 = fn_decl @F.7 [template = constants.%F.8] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc117_13: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc117_13.1: type = value_of_initializer %bool.make_type.loc117_13 [template = bool]
// CHECK:STDOUT:     %.loc117_13.2: type = converted %bool.make_type.loc117_13, %.loc117_13.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc117_22: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc117_22.1: type = value_of_initializer %bool.make_type.loc117_22 [template = bool]
// CHECK:STDOUT:     %.loc117_22.2: type = converted %bool.make_type.loc117_22, %.loc117_22.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.8 = fn_decl @F.8 [template = constants.%F.9] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc130_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc130_16.1: type = value_of_initializer %bool.make_type.loc130_16 [template = bool]
// CHECK:STDOUT:     %.loc130_16.2: type = converted %bool.make_type.loc130_16, %.loc130_16.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc130_25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc130_25.1: type = value_of_initializer %bool.make_type.loc130_25 [template = bool]
// CHECK:STDOUT:     %.loc130_25.2: type = converted %bool.make_type.loc130_25, %.loc130_25.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.9 = fn_decl @F.9 [template = constants.%F.10] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %FDifferentParamType = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %FDifferentParamType = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc143_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc143_16.1: type = value_of_initializer %bool.make_type.loc143_16 [template = bool]
// CHECK:STDOUT:     %.loc143_16.2: type = converted %bool.make_type.loc143_16, %.loc143_16.1 [template = bool]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [template = constants.%FDifferentParamType]
// CHECK:STDOUT:     %bool.make_type.loc143_34: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc143_34.1: type = value_of_initializer %bool.make_type.loc143_34 [template = bool]
// CHECK:STDOUT:     %.loc143_34.2: type = converted %bool.make_type.loc143_34, %.loc143_34.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: %FDifferentParamType = value_param runtime_param1
// CHECK:STDOUT:     %b: %FDifferentParamType = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.11: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.10 = fn_decl @F.10 [template = constants.%F.11] {
// CHECK:STDOUT:     %self.patt: %FDifferentImplicitParamType = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %FDifferentImplicitParamType = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %bool.make_type.loc156_25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc156_25.1: type = value_of_initializer %bool.make_type.loc156_25 [template = bool]
// CHECK:STDOUT:     %.loc156_25.2: type = converted %bool.make_type.loc156_25, %.loc156_25.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc156_34: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc156_34.1: type = value_of_initializer %bool.make_type.loc156_34 [template = bool]
// CHECK:STDOUT:     %.loc156_34.2: type = converted %bool.make_type.loc156_34, %.loc156_34.1 [template = bool]
// CHECK:STDOUT:     %self.param: %FDifferentImplicitParamType = value_param runtime_param0
// CHECK:STDOUT:     %self: %FDifferentImplicitParamType = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.12: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.11 = fn_decl @F.11 [template = constants.%F.12] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %FDifferentReturnType = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %FDifferentReturnType = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc169_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc169_16.1: type = value_of_initializer %bool.make_type.loc169_16 [template = bool]
// CHECK:STDOUT:     %.loc169_16.2: type = converted %bool.make_type.loc169_16, %.loc169_16.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc169_25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc169_25.1: type = value_of_initializer %bool.make_type.loc169_25 [template = bool]
// CHECK:STDOUT:     %.loc169_25.2: type = converted %bool.make_type.loc169_25, %.loc169_25.1 [template = bool]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [template = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %FDifferentReturnType = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.13: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.12 = fn_decl @F.12 [template = constants.%F.13] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: bool = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %not_b.patt: bool = binding_pattern not_b
// CHECK:STDOUT:     %not_b.param_patt: bool = value_param_pattern %not_b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc183_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc183_16.1: type = value_of_initializer %bool.make_type.loc183_16 [template = bool]
// CHECK:STDOUT:     %.loc183_16.2: type = converted %bool.make_type.loc183_16, %.loc183_16.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc183_29: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc183_29.1: type = value_of_initializer %bool.make_type.loc183_29 [template = bool]
// CHECK:STDOUT:     %.loc183_29.2: type = converted %bool.make_type.loc183_29, %.loc183_29.1 [template = bool]
// CHECK:STDOUT:     %bool.make_type.loc183_38: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc183_38.1: type = value_of_initializer %bool.make_type.loc183_38 [template = bool]
// CHECK:STDOUT:     %.loc183_38.2: type = converted %bool.make_type.loc183_38, %.loc183_38.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %not_b.param: bool = value_param runtime_param1
// CHECK:STDOUT:     %not_b: bool = bind_name not_b, %not_b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.14: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.14 = fn_decl @F.14 [template = constants.%F.15] {
// CHECK:STDOUT:     %x.patt: %tuple.type.3 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.3 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %array_type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc200_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %ptr: type = ptr_type %SelfNestedBadParam [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32.loc200_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc200_40: init type = call constants.%Int(%int_32.loc200_40) [template = constants.%i32]
// CHECK:STDOUT:     %.loc200_40.1: type = value_of_initializer %int.make_type_signed.loc200_40 [template = constants.%i32]
// CHECK:STDOUT:     %.loc200_40.2: type = converted %int.make_type_signed.loc200_40, %.loc200_40.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc200_49: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc200_49: init type = call constants.%Int(%int_32.loc200_49) [template = constants.%i32]
// CHECK:STDOUT:     %.loc200_49.1: type = value_of_initializer %int.make_type_signed.loc200_49 [template = constants.%i32]
// CHECK:STDOUT:     %.loc200_49.2: type = converted %int.make_type_signed.loc200_49, %.loc200_49.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [template = constants.%struct_type.x.y.2]
// CHECK:STDOUT:     %.loc200_53.1: %tuple.type.1 = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:     %.loc200_53.2: type = converted %.loc200_53.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc200_60: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam [template = constants.%array_type.2]
// CHECK:STDOUT:     %x.param: %tuple.type.3 = value_param runtime_param0
// CHECK:STDOUT:     %x: %tuple.type.3 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %array_type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.15: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.15 = fn_decl @F.15 [template = constants.%F.16] {
// CHECK:STDOUT:     %x.patt: %tuple.type.5 = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %tuple.type.5 = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %array_type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadReturnType.ref.loc212_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %ptr: type = ptr_type %SelfNestedBadReturnType [template = constants.%ptr.3]
// CHECK:STDOUT:     %SelfNestedBadReturnType.ref.loc212_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc212_74.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc212_74.2: type = converted %int.make_type_signed, %.loc212_74.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.x.y: type = struct_type {.x: %SelfNestedBadReturnType, .y: %i32} [template = constants.%struct_type.x.y.4]
// CHECK:STDOUT:     %.loc212_78.1: %tuple.type.1 = tuple_literal (%ptr, %struct_type.x.y)
// CHECK:STDOUT:     %.loc212_78.2: type = converted %.loc212_78.1, constants.%tuple.type.5 [template = constants.%tuple.type.5]
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:     %array_type: type = array_type %int_4, %SelfNestedBadParam [template = constants.%array_type.2]
// CHECK:STDOUT:     %x.param: %tuple.type.5 = value_param runtime_param0
// CHECK:STDOUT:     %x: %tuple.type.5 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %array_type.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %array_type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %interface: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %interface
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoF {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoF [template = constants.%NoF]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoF
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FNotFunction {
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FNotFunction [template = constants.%FNotFunction]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FNotFunction
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F.16;
// CHECK:STDOUT:
// CHECK:STDOUT: class @FAlias {
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FAlias [template = constants.%FAlias]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FAlias
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraParam {
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraParam [template = constants.%FExtraParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraParam
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraImplicitParam {
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [template = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraImplicitParam
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraReturnType {
// CHECK:STDOUT:   impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraReturnType [template = constants.%FExtraReturnType]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraReturnType
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingParam {
// CHECK:STDOUT:   impl_decl @impl.7 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingParam [template = constants.%FMissingParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingParam
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingImplicitParam {
// CHECK:STDOUT:   impl_decl @impl.8 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingImplicitParam [template = constants.%FMissingImplicitParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingImplicitParam
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingReturnType {
// CHECK:STDOUT:   impl_decl @impl.9 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingReturnType [template = constants.%FMissingReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingReturnType
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamType {
// CHECK:STDOUT:   impl_decl @impl.10 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [template = constants.%FDifferentParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamType
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentImplicitParamType {
// CHECK:STDOUT:   impl_decl @impl.11 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentImplicitParamType
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentReturnType {
// CHECK:STDOUT:   impl_decl @impl.12 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [template = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentReturnType
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamName {
// CHECK:STDOUT:   impl_decl @impl.13 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamName [template = constants.%FDifferentParamName]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamName
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadParam {
// CHECK:STDOUT:   impl_decl @impl.14 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadParam [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadParam
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadReturnType {
// CHECK:STDOUT:   impl_decl @impl.15 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadReturnType [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:
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
// CHECK:STDOUT:
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
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Self.as_type.loc188_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc188_16.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %ptr.loc188_16.1: type = ptr_type @F.13.%Self.as_type.loc188_16.1 (%Self.as_type) [symbolic = %ptr.loc188_16.1 (constants.%ptr.1)]
// CHECK:STDOUT:   %struct_type.x.y.loc188_37.1: type = struct_type {.x: @F.13.%Self.as_type.loc188_16.1 (%Self.as_type), .y: %i32} [symbolic = %struct_type.x.y.loc188_37.1 (constants.%struct_type.x.y.1)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.13.%ptr.loc188_16.1 (%ptr.1), @F.13.%struct_type.x.y.loc188_37.1 (%struct_type.x.y.1)) [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:   %array_type.loc188_52.1: type = array_type constants.%int_4, @F.13.%Self.as_type.loc188_16.1 (%Self.as_type) [symbolic = %array_type.loc188_52.1 (constants.%array_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param_patt: @F.13.%tuple.type (%tuple.type.2)) -> @F.13.%array_type.loc188_52.1 (%array_type.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.14(%x.param_patt: %tuple.type.3) -> %array_type.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.15(%x.param_patt: %tuple.type.5) -> %array_type.2;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.4) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%J.facet.7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%Self.3) {
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Self.as_type.loc188_16.1 => constants.%Self.as_type
// CHECK:STDOUT:   %ptr.loc188_16.1 => constants.%ptr.1
// CHECK:STDOUT:   %struct_type.x.y.loc188_37.1 => constants.%struct_type.x.y.1
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2
// CHECK:STDOUT:   %array_type.loc188_52.1 => constants.%array_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%SelfNested.facet.1) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.1
// CHECK:STDOUT:   %Self.as_type.loc188_16.1 => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %ptr.loc188_16.1 => constants.%ptr.2
// CHECK:STDOUT:   %struct_type.x.y.loc188_37.1 => constants.%struct_type.x.y.3
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.4
// CHECK:STDOUT:   %array_type.loc188_52.1 => constants.%array_type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%SelfNested.facet.2) {
// CHECK:STDOUT:   %Self => constants.%SelfNested.facet.2
// CHECK:STDOUT:   %Self.as_type.loc188_16.1 => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %ptr.loc188_16.1 => constants.%ptr.3
// CHECK:STDOUT:   %struct_type.x.y.loc188_37.1 => constants.%struct_type.x.y.4
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.5
// CHECK:STDOUT:   %array_type.loc188_52.1 => constants.%array_type.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
