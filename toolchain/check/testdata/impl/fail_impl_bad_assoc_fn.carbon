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
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:3: error: missing implementation of F in impl of interface I
  // CHECK:STDERR:   impl as I {}
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-6]]:15: note: associated function F declared here
  // CHECK:STDERR: interface I { fn F(); }
  // CHECK:STDERR:               ^~~~~~~
  // CHECK:STDERR:
  impl as I {}
}

class FNotFunction {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: associated function F implemented by non-function
    // CHECK:STDERR:     class F;
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-18]]:15: note: associated function F declared here
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:11: error: associated function F implemented by non-function
    // CHECK:STDERR:     alias F = PossiblyF;
    // CHECK:STDERR:           ^
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-34]]:15: note: associated function F declared here
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    alias F = PossiblyF;
  }
}

class FExtraParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: redeclaration differs because of parameter count of 1
    // CHECK:STDERR:     fn F(b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-47]]:15: note: previously declared with parameter count of 0
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F(b: bool);
  }
}

class FExtraImplicitParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: redeclaration differs because of implicit parameter list
    // CHECK:STDERR:     fn F[self: Self]();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-60]]:15: note: previously declared without implicit parameter list
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F[self: Self]();
  }
}

// TODO: Should this be permitted?
class FExtraReturnType {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: function redeclaration differs because return type is `bool`
    // CHECK:STDERR:     fn F() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-74]]:15: note: previously declared with no return type
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F() -> bool;
  }
}

interface J { fn F[self: bool](b: bool) -> bool; }

class FMissingParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: redeclaration differs because of parameter count of 0
    // CHECK:STDERR:     fn F[self: bool]() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-7]]:15: note: previously declared with parameter count of 1
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool]() -> bool;
  }
}

class FMissingImplicitParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: redeclaration differs because of missing implicit parameter list
    // CHECK:STDERR:     fn F(b: bool) -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-20]]:15: note: previously declared with implicit parameter list
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(b: bool) -> bool;
  }
}

class FMissingReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: function redeclaration differs because no return type is provided
    // CHECK:STDERR:     fn F[self: bool](b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-33]]:15: note: previously declared with return type `bool`
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](b: bool);
  }
}

class FDifferentParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:22: error: redeclaration differs at parameter 1
    // CHECK:STDERR:     fn F[self: bool](b: Self) -> bool;
    // CHECK:STDERR:                      ^
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-46]]:32: note: previous declaration's corresponding parameter here
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^
    // CHECK:STDERR:
    fn F[self: bool](b: Self) -> bool;
  }
}

class FDifferentImplicitParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: error: redeclaration differs at implicit parameter 1
    // CHECK:STDERR:     fn F[self: Self](b: bool) -> bool;
    // CHECK:STDERR:          ^~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-59]]:20: note: previous declaration's corresponding implicit parameter here
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                    ^~~~
    // CHECK:STDERR:
    fn F[self: Self](b: bool) -> bool;
  }
}

class FDifferentReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: error: function redeclaration differs because return type is `FDifferentReturnType`
    // CHECK:STDERR:     fn F[self: bool](b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-72]]:15: note: previously declared with return type `bool`
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](b: bool) -> Self;
  }
}

// TODO: This should probably be permitted.
class FDifferentParamName {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:22: error: redeclaration differs at parameter 1
    // CHECK:STDERR:     fn F[self: bool](not_b: bool) -> bool;
    // CHECK:STDERR:                      ^~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-86]]:32: note: previous declaration's corresponding parameter here
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^
    // CHECK:STDERR:
    fn F[self: bool](not_b: bool) -> bool;
  }
}

interface SelfNested {
  fn F(x: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
}

class SelfNestedBadParam {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: error: redeclaration differs at parameter 1
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> [SelfNestedBadParam; 4];
    // CHECK:STDERR:          ^
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-8]]:8: note: previous declaration's corresponding parameter here
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
    // CHECK:STDERR:        ^
    // CHECK:STDERR:
    fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> [SelfNestedBadParam; 4];
  }
}

class SelfNestedBadReturnType {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: error: function redeclaration differs because return type is `[SelfNestedBadParam; 4]`
    // CHECK:STDERR:     fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> [SelfNestedBadParam; 4];
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-21]]:3: note: previously declared with return type `[SelfNestedBadReturnType; 4]`
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> [SelfNestedBadParam; 4];
  }
}

// CHECK:STDOUT: --- fail_impl_bad_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = interface_type @I [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
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
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.4: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [template]
// CHECK:STDOUT:   %F.type.4: type = fn_type @F.4 [template]
// CHECK:STDOUT:   %F.5: %F.type.4 = struct_value () [template]
// CHECK:STDOUT:   %J.type: type = interface_type @J [template]
// CHECK:STDOUT:   %Self.2: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.5: type = fn_type @F.5 [template]
// CHECK:STDOUT:   %F.6: %F.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %J.type, %F.type.5 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @J.%F.decl [template]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [template]
// CHECK:STDOUT:   %F.type.6: type = fn_type @F.6 [template]
// CHECK:STDOUT:   %F.7: %F.type.6 = struct_value () [template]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [template]
// CHECK:STDOUT:   %F.type.7: type = fn_type @F.7 [template]
// CHECK:STDOUT:   %F.8: %F.type.7 = struct_value () [template]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [template]
// CHECK:STDOUT:   %F.type.8: type = fn_type @F.8 [template]
// CHECK:STDOUT:   %F.9: %F.type.8 = struct_value () [template]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [template]
// CHECK:STDOUT:   %F.type.9: type = fn_type @F.9 [template]
// CHECK:STDOUT:   %F.10: %F.type.9 = struct_value () [template]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [template]
// CHECK:STDOUT:   %F.type.10: type = fn_type @F.10 [template]
// CHECK:STDOUT:   %F.11: %F.type.10 = struct_value () [template]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [template]
// CHECK:STDOUT:   %F.type.11: type = fn_type @F.11 [template]
// CHECK:STDOUT:   %F.12: %F.type.11 = struct_value () [template]
// CHECK:STDOUT:   %FDifferentParamName: type = class_type @FDifferentParamName [template]
// CHECK:STDOUT:   %F.type.12: type = fn_type @F.12 [template]
// CHECK:STDOUT:   %F.13: %F.type.12 = struct_value () [template]
// CHECK:STDOUT:   %SelfNested.type: type = interface_type @SelfNested [template]
// CHECK:STDOUT:   %Self.3: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.8: type = ptr_type %Self.3 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = struct_type {.x: %Self.3, .y: i32} [symbolic]
// CHECK:STDOUT:   %.10: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.11: type = tuple_type (%.8, %.9) [symbolic]
// CHECK:STDOUT:   %.12: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.13: type = array_type %.12, %Self.3 [symbolic]
// CHECK:STDOUT:   %F.type.13: type = fn_type @F.13 [template]
// CHECK:STDOUT:   %F.14: %F.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %SelfNested.type, %F.type.13 [template]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, @SelfNested.%F.decl [template]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [template]
// CHECK:STDOUT:   %.16: type = ptr_type %SelfNestedBadParam [template]
// CHECK:STDOUT:   %.17: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.18: type = tuple_type (%.16, %.17) [template]
// CHECK:STDOUT:   %.19: type = array_type %.12, %SelfNestedBadParam [template]
// CHECK:STDOUT:   %F.type.14: type = fn_type @F.14 [template]
// CHECK:STDOUT:   %F.15: %F.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.20: type = struct_type {.x: %SelfNestedBadParam, .y: i32} [template]
// CHECK:STDOUT:   %.21: type = tuple_type (%.16, %.20) [template]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [template]
// CHECK:STDOUT:   %.22: type = ptr_type %SelfNestedBadReturnType [template]
// CHECK:STDOUT:   %.23: type = struct_type {.x: %SelfNestedBadReturnType, .y: i32} [template]
// CHECK:STDOUT:   %.24: type = tuple_type (%.22, %.23) [template]
// CHECK:STDOUT:   %F.type.15: type = fn_type @F.15 [template]
// CHECK:STDOUT:   %F.16: %F.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: type = array_type %.12, %SelfNestedBadReturnType [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %.loc11: %.2 = assoc_entity element0, %F.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc11
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: %F.type.5 = fn_decl @F.5 [template = constants.%F.6] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc93_26: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc93_26.1: type = value_of_initializer %bool.make_type.loc93_26 [template = bool]
// CHECK:STDOUT:     %.loc93_26.2: type = converted %bool.make_type.loc93_26, %.loc93_26.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = param self, runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %bool.make_type.loc93_35: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc93_35.1: type = value_of_initializer %bool.make_type.loc93_35 [template = bool]
// CHECK:STDOUT:     %.loc93_35.2: type = converted %bool.make_type.loc93_35, %.loc93_35.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = param b, runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %bool.make_type.loc93_44: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc93_44.1: type = value_of_initializer %bool.make_type.loc93_44 [template = bool]
// CHECK:STDOUT:     %.loc93_44.2: type = converted %bool.make_type.loc93_44, %.loc93_44.1 [template = bool]
// CHECK:STDOUT:     %return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc93: %.6 = assoc_entity element0, %F.decl [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc93
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:   %F.decl: %F.type.13 = fn_decl @F.13 [template = constants.%F.14] {
// CHECK:STDOUT:     %x.patt: @F.13.%.3 (%.11) = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc188_12: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc188_16.1: type = facet_type_access %Self.ref.loc188_12 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc188_16.2: type = converted %Self.ref.loc188_12, %.loc188_16.1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc188_16.3: type = ptr_type %Self.3 [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:     %Self.ref.loc188_24: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc188_24.1: type = facet_type_access %Self.ref.loc188_24 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc188_24.2: type = converted %Self.ref.loc188_24, %.loc188_24.1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc188_34.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc188_34.2: type = converted %int.make_type_32, %.loc188_34.1 [template = i32]
// CHECK:STDOUT:     %.loc188_37: type = struct_type {.x: %Self.3, .y: i32} [symbolic = %.2 (constants.%.9)]
// CHECK:STDOUT:     %.loc188_38.1: %.10 = tuple_literal (%.loc188_16.3, %.loc188_37)
// CHECK:STDOUT:     %.loc188_38.2: type = converted %.loc188_38.1, constants.%.11 [symbolic = %.3 (constants.%.11)]
// CHECK:STDOUT:     %x.param: @F.13.%.3 (%.11) = param x, runtime_param0
// CHECK:STDOUT:     %x: @F.13.%.3 (%.11) = bind_name x, %x.param
// CHECK:STDOUT:     %Self.ref.loc188_45: %SelfNested.type = name_ref Self, @SelfNested.%Self [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc188_51: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:     %.loc188_45.1: type = facet_type_access %Self.ref.loc188_45 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc188_45.2: type = converted %Self.ref.loc188_45, %.loc188_45.1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:     %.loc188_52: type = array_type %.loc188_51, %Self.3 [symbolic = %.4 (constants.%.13)]
// CHECK:STDOUT:     %return: ref @F.13.%.4 (%.13) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc188: %.14 = assoc_entity element0, %F.decl [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc188
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %Self.ref as %I.ref {
// CHECK:STDOUT:   %.loc21: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.loc21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: type = class_decl @F.16 [template = constants.%F.2] {} {}
// CHECK:STDOUT:   %.loc25: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %Self.ref as %I.ref {
// CHECK:STDOUT:   %PossiblyF.ref: %PossiblyF.type = name_ref PossiblyF, file.%PossiblyF.decl [template = constants.%PossiblyF]
// CHECK:STDOUT:   %F: %PossiblyF.type = bind_alias F, file.%PossiblyF.decl [template = constants.%PossiblyF]
// CHECK:STDOUT:   %.loc41: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.loc41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.3] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc62_13.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc62_13.2: type = converted %bool.make_type, %.loc62_13.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = param b, runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc54: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.4] {
// CHECK:STDOUT:     %self.patt: %FExtraImplicitParam = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [template = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self.param: %FExtraImplicitParam = param self, runtime_param0
// CHECK:STDOUT:     %self: %FExtraImplicitParam = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc67: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.4 = fn_decl @F.4 [template = constants.%F.5] {} {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc89_15.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc89_15.2: type = converted %bool.make_type, %.loc89_15.1 [template = bool]
// CHECK:STDOUT:     %return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc81: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.6 = fn_decl @F.6 [template = constants.%F.7] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc104_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc104_16.1: type = value_of_initializer %bool.make_type.loc104_16 [template = bool]
// CHECK:STDOUT:     %.loc104_16.2: type = converted %bool.make_type.loc104_16, %.loc104_16.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = param self, runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %bool.make_type.loc104_27: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc104_27.1: type = value_of_initializer %bool.make_type.loc104_27 [template = bool]
// CHECK:STDOUT:     %.loc104_27.2: type = converted %bool.make_type.loc104_27, %.loc104_27.1 [template = bool]
// CHECK:STDOUT:     %return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc96: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.7 = fn_decl @F.7 [template = constants.%F.8] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc117_13: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc117_13.1: type = value_of_initializer %bool.make_type.loc117_13 [template = bool]
// CHECK:STDOUT:     %.loc117_13.2: type = converted %bool.make_type.loc117_13, %.loc117_13.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = param b, runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %bool.make_type.loc117_22: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc117_22.1: type = value_of_initializer %bool.make_type.loc117_22 [template = bool]
// CHECK:STDOUT:     %.loc117_22.2: type = converted %bool.make_type.loc117_22, %.loc117_22.1 [template = bool]
// CHECK:STDOUT:     %return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc109: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.8 = fn_decl @F.8 [template = constants.%F.9] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc130_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc130_16.1: type = value_of_initializer %bool.make_type.loc130_16 [template = bool]
// CHECK:STDOUT:     %.loc130_16.2: type = converted %bool.make_type.loc130_16, %.loc130_16.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = param self, runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %bool.make_type.loc130_25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc130_25.1: type = value_of_initializer %bool.make_type.loc130_25 [template = bool]
// CHECK:STDOUT:     %.loc130_25.2: type = converted %bool.make_type.loc130_25, %.loc130_25.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = param b, runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc122: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.9 = fn_decl @F.9 [template = constants.%F.10] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %b.patt: %FDifferentParamType = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc143_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc143_16.1: type = value_of_initializer %bool.make_type.loc143_16 [template = bool]
// CHECK:STDOUT:     %.loc143_16.2: type = converted %bool.make_type.loc143_16, %.loc143_16.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = param self, runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [template = constants.%FDifferentParamType]
// CHECK:STDOUT:     %b.param: %FDifferentParamType = param b, runtime_param1
// CHECK:STDOUT:     %b: %FDifferentParamType = bind_name b, %b.param
// CHECK:STDOUT:     %bool.make_type.loc143_34: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc143_34.1: type = value_of_initializer %bool.make_type.loc143_34 [template = bool]
// CHECK:STDOUT:     %.loc143_34.2: type = converted %bool.make_type.loc143_34, %.loc143_34.1 [template = bool]
// CHECK:STDOUT:     %return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc135: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.11: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.10 = fn_decl @F.10 [template = constants.%F.11] {
// CHECK:STDOUT:     %self.patt: %FDifferentImplicitParamType = binding_pattern self
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self.param: %FDifferentImplicitParamType = param self, runtime_param0
// CHECK:STDOUT:     %self: %FDifferentImplicitParamType = bind_name self, %self.param
// CHECK:STDOUT:     %bool.make_type.loc156_25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc156_25.1: type = value_of_initializer %bool.make_type.loc156_25 [template = bool]
// CHECK:STDOUT:     %.loc156_25.2: type = converted %bool.make_type.loc156_25, %.loc156_25.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = param b, runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %bool.make_type.loc156_34: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc156_34.1: type = value_of_initializer %bool.make_type.loc156_34 [template = bool]
// CHECK:STDOUT:     %.loc156_34.2: type = converted %bool.make_type.loc156_34, %.loc156_34.1 [template = bool]
// CHECK:STDOUT:     %return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc148: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.12: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.11 = fn_decl @F.11 [template = constants.%F.12] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc169_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc169_16.1: type = value_of_initializer %bool.make_type.loc169_16 [template = bool]
// CHECK:STDOUT:     %.loc169_16.2: type = converted %bool.make_type.loc169_16, %.loc169_16.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = param self, runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %bool.make_type.loc169_25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc169_25.1: type = value_of_initializer %bool.make_type.loc169_25 [template = bool]
// CHECK:STDOUT:     %.loc169_25.2: type = converted %bool.make_type.loc169_25, %.loc169_25.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = param b, runtime_param1
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [template = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %return: ref %FDifferentReturnType = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc161: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.13: %Self.ref as %J.ref {
// CHECK:STDOUT:   %F.decl: %F.type.12 = fn_decl @F.12 [template = constants.%F.13] {
// CHECK:STDOUT:     %self.patt: bool = binding_pattern self
// CHECK:STDOUT:     %not_b.patt: bool = binding_pattern not_b
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc183_16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc183_16.1: type = value_of_initializer %bool.make_type.loc183_16 [template = bool]
// CHECK:STDOUT:     %.loc183_16.2: type = converted %bool.make_type.loc183_16, %.loc183_16.1 [template = bool]
// CHECK:STDOUT:     %self.param: bool = param self, runtime_param0
// CHECK:STDOUT:     %self: bool = bind_name self, %self.param
// CHECK:STDOUT:     %bool.make_type.loc183_29: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc183_29.1: type = value_of_initializer %bool.make_type.loc183_29 [template = bool]
// CHECK:STDOUT:     %.loc183_29.2: type = converted %bool.make_type.loc183_29, %.loc183_29.1 [template = bool]
// CHECK:STDOUT:     %not_b.param: bool = param not_b, runtime_param1
// CHECK:STDOUT:     %not_b: bool = bind_name not_b, %not_b.param
// CHECK:STDOUT:     %bool.make_type.loc183_38: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc183_38.1: type = value_of_initializer %bool.make_type.loc183_38 [template = bool]
// CHECK:STDOUT:     %.loc183_38.2: type = converted %bool.make_type.loc183_38, %.loc183_38.1 [template = bool]
// CHECK:STDOUT:     %return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc175: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.14: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.14 = fn_decl @F.14 [template = constants.%F.15] {
// CHECK:STDOUT:     %x.patt: %.18 = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc200_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %.loc200_32: type = ptr_type %SelfNestedBadParam [template = constants.%.16]
// CHECK:STDOUT:     %int.make_type_32.loc200_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc200_40.1: type = value_of_initializer %int.make_type_32.loc200_40 [template = i32]
// CHECK:STDOUT:     %.loc200_40.2: type = converted %int.make_type_32.loc200_40, %.loc200_40.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc200_49: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc200_49.1: type = value_of_initializer %int.make_type_32.loc200_49 [template = i32]
// CHECK:STDOUT:     %.loc200_49.2: type = converted %int.make_type_32.loc200_49, %.loc200_49.1 [template = i32]
// CHECK:STDOUT:     %.loc200_52: type = struct_type {.x: i32, .y: i32} [template = constants.%.17]
// CHECK:STDOUT:     %.loc200_53.1: %.10 = tuple_literal (%.loc200_32, %.loc200_52)
// CHECK:STDOUT:     %.loc200_53.2: type = converted %.loc200_53.1, constants.%.18 [template = constants.%.18]
// CHECK:STDOUT:     %x.param: %.18 = param x, runtime_param0
// CHECK:STDOUT:     %x: %.18 = bind_name x, %x.param
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc200_60: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %.loc200_80: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:     %.loc200_81: type = array_type %.loc200_80, %SelfNestedBadParam [template = constants.%.19]
// CHECK:STDOUT:     %return: ref %.19 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc192: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.15: %Self.ref as %SelfNested.ref {
// CHECK:STDOUT:   %F.decl: %F.type.15 = fn_decl @F.15 [template = constants.%F.16] {
// CHECK:STDOUT:     %x.patt: %.24 = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SelfNestedBadReturnType.ref.loc212_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %.loc212_37: type = ptr_type %SelfNestedBadReturnType [template = constants.%.22]
// CHECK:STDOUT:     %SelfNestedBadReturnType.ref.loc212_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc212_74.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc212_74.2: type = converted %int.make_type_32, %.loc212_74.1 [template = i32]
// CHECK:STDOUT:     %.loc212_77: type = struct_type {.x: %SelfNestedBadReturnType, .y: i32} [template = constants.%.23]
// CHECK:STDOUT:     %.loc212_78.1: %.10 = tuple_literal (%.loc212_37, %.loc212_77)
// CHECK:STDOUT:     %.loc212_78.2: type = converted %.loc212_78.1, constants.%.24 [template = constants.%.24]
// CHECK:STDOUT:     %x.param: %.24 = param x, runtime_param0
// CHECK:STDOUT:     %x: %.24 = bind_name x, %x.param
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %.loc212_105: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:     %.loc212_106: type = array_type %.loc212_105, %SelfNestedBadParam [template = constants.%.19]
// CHECK:STDOUT:     %return: ref %.19 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc205: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.loc205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoF {
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NoF [template = constants.%NoF]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoF
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FNotFunction {
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FNotFunction [template = constants.%FNotFunction]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc35: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FNotFunction
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F.16;
// CHECK:STDOUT:
// CHECK:STDOUT: class @FAlias {
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FAlias [template = constants.%FAlias]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc51: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FAlias
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraParam {
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraParam [template = constants.%FExtraParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc64: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraImplicitParam {
// CHECK:STDOUT:   impl_decl @impl.5 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [template = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc77: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraImplicitParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraReturnType {
// CHECK:STDOUT:   impl_decl @impl.6 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraReturnType [template = constants.%FExtraReturnType]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc91: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingParam {
// CHECK:STDOUT:   impl_decl @impl.7 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingParam [template = constants.%FMissingParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc106: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingImplicitParam {
// CHECK:STDOUT:   impl_decl @impl.8 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingImplicitParam [template = constants.%FMissingImplicitParam]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc119: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingImplicitParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingReturnType {
// CHECK:STDOUT:   impl_decl @impl.9 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FMissingReturnType [template = constants.%FMissingReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc132: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamType {
// CHECK:STDOUT:   impl_decl @impl.10 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [template = constants.%FDifferentParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc145: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentImplicitParamType {
// CHECK:STDOUT:   impl_decl @impl.11 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc158: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentImplicitParamType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentReturnType {
// CHECK:STDOUT:   impl_decl @impl.12 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [template = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc171: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamName {
// CHECK:STDOUT:   impl_decl @impl.13 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamName [template = constants.%FDifferentParamName]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc185: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamName
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadParam {
// CHECK:STDOUT:   impl_decl @impl.14 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadParam [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc202: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadReturnType {
// CHECK:STDOUT:   impl_decl @impl.15 [template] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SelfNestedBadReturnType [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%SelfNested.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc214: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PossiblyF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%b: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self: %FExtraImplicitParam]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.4() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.5(@J.%Self: %J.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: bool](%b: bool) -> bool;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6[%self: bool]() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.7(%b: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.8[%self: bool](%b: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.9[%self: bool](%b: %FDifferentParamType) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.10[%self: %FDifferentImplicitParamType](%b: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.11[%self: bool](%b: bool) -> %FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.12[%self: bool](%not_b: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.13(@SelfNested.%Self: %SelfNested.type) {
// CHECK:STDOUT:   %Self: %SelfNested.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %.1: type = ptr_type @F.13.%Self (%Self.3) [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:   %.2: type = struct_type {.x: @F.13.%Self (%Self.3), .y: i32} [symbolic = %.2 (constants.%.9)]
// CHECK:STDOUT:   %.3: type = tuple_type (@F.13.%.1 (%.8), @F.13.%.2 (%.9)) [symbolic = %.3 (constants.%.11)]
// CHECK:STDOUT:   %.4: type = array_type constants.%.12, @F.13.%Self (%Self.3) [symbolic = %.4 (constants.%.13)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x: @F.13.%.3 (%.11)) -> @F.13.%.4 (%.13);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.14(%x: %.18) -> %.19;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.15(%x: %.24) -> %.19;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%FExtraParam) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%FExtraImplicitParam) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%FExtraReturnType) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%Self.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%FMissingParam) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%FMissingImplicitParam) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%FMissingReturnType) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%FDifferentParamType) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%FDifferentImplicitParamType) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%FDifferentReturnType) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.5(constants.%FDifferentParamName) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%Self.3) {
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT:   %.3 => constants.%.11
// CHECK:STDOUT:   %.4 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%SelfNestedBadParam) {
// CHECK:STDOUT:   %Self => constants.%SelfNestedBadParam
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %.2 => constants.%.20
// CHECK:STDOUT:   %.3 => constants.%.21
// CHECK:STDOUT:   %.4 => constants.%.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.13(constants.%SelfNestedBadReturnType) {
// CHECK:STDOUT:   %Self => constants.%SelfNestedBadReturnType
// CHECK:STDOUT:   %.1 => constants.%.22
// CHECK:STDOUT:   %.2 => constants.%.23
// CHECK:STDOUT:   %.3 => constants.%.24
// CHECK:STDOUT:   %.4 => constants.%.25
// CHECK:STDOUT: }
// CHECK:STDOUT:
