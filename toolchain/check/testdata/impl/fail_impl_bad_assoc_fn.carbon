// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I { fn F(); }

class NoF {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:3: ERROR: Missing implementation of F in impl of interface I.
  // CHECK:STDERR:   impl as I {}
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-6]]:15: Associated function F declared here.
  // CHECK:STDERR: interface I { fn F(); }
  // CHECK:STDERR:               ^~~~~~~
  // CHECK:STDERR:
  impl as I {}
}

class FNotFunction {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: ERROR: Associated function F implemented by non-function.
    // CHECK:STDERR:     class F;
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-18]]:15: Associated function F declared here.
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:11: ERROR: Associated function F implemented by non-function.
    // CHECK:STDERR:     alias F = PossiblyF;
    // CHECK:STDERR:           ^
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-34]]:15: Associated function F declared here.
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    alias F = PossiblyF;
  }
}

class FExtraParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: ERROR: Redeclaration differs because of parameter count of 1.
    // CHECK:STDERR:     fn F(b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-47]]:15: Previously declared with parameter count of 0.
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F(b: bool);
  }
}

class FExtraImplicitParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: ERROR: Redeclaration differs because of implicit parameter count of 1.
    // CHECK:STDERR:     fn F[self: Self]();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-60]]:15: Previously declared with implicit parameter count of 0.
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F[self: Self]();
  }
}

// TODO: Should this be permitted?
class FExtraReturnType {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: ERROR: Function redeclaration differs because return type is `bool`.
    // CHECK:STDERR:     fn F() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-74]]:15: Previously declared with no return type.
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    // CHECK:STDERR:
    fn F() -> bool;
  }
}

interface J { fn F[self: bool](b: bool) -> bool; }

class FMissingParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: ERROR: Redeclaration differs because of parameter count of 0.
    // CHECK:STDERR:     fn F[self: bool]() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-7]]:15: Previously declared with parameter count of 1.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool]() -> bool;
  }
}

class FMissingImplicitParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: ERROR: Redeclaration differs because of implicit parameter count of 0.
    // CHECK:STDERR:     fn F(b: bool) -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-20]]:15: Previously declared with implicit parameter count of 1.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F(b: bool) -> bool;
  }
}

class FMissingReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: ERROR: Function redeclaration differs because no return type is provided.
    // CHECK:STDERR:     fn F[self: bool](b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-33]]:15: Previously declared with return type `bool`.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](b: bool);
  }
}

class FDifferentParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:22: ERROR: Redeclaration differs at parameter 1.
    // CHECK:STDERR:     fn F[self: bool](b: Self) -> bool;
    // CHECK:STDERR:                      ^
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-46]]:32: Previous declaration's corresponding parameter here.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^
    // CHECK:STDERR:
    fn F[self: bool](b: Self) -> bool;
  }
}

class FDifferentImplicitParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: ERROR: Redeclaration differs at implicit parameter 1.
    // CHECK:STDERR:     fn F[self: Self](b: bool) -> bool;
    // CHECK:STDERR:          ^~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-59]]:20: Previous declaration's corresponding implicit parameter here.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                    ^~~~
    // CHECK:STDERR:
    fn F[self: Self](b: bool) -> bool;
  }
}

class FDifferentReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:5: ERROR: Function redeclaration differs because return type is `FDifferentReturnType`.
    // CHECK:STDERR:     fn F[self: bool](b: bool) -> Self;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-72]]:15: Previously declared with return type `bool`.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    fn F[self: bool](b: bool) -> Self;
  }
}

// TODO: This should probably be permitted.
class FDifferentParamName {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:22: ERROR: Redeclaration differs at parameter 1.
    // CHECK:STDERR:     fn F[self: bool](not_b: bool) -> bool;
    // CHECK:STDERR:                      ^~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-86]]:32: Previous declaration's corresponding parameter here.
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
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+7]]:10: ERROR: Redeclaration differs at parameter 1.
    // CHECK:STDERR:     fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> [SelfNestedBadParam; 4];
    // CHECK:STDERR:          ^
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-8]]:8: Previous declaration's corresponding parameter here.
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
    // CHECK:STDERR:        ^
    // CHECK:STDERR:
    fn F(x: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> [SelfNestedBadParam; 4];
  }
}

class SelfNestedBadReturnType {
  impl as SelfNested {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: ERROR: Function redeclaration differs because return type is `[SelfNestedBadParam; 4]`.
    // CHECK:STDERR:     fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> [SelfNestedBadParam; 4];
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-21]]:3: Previously declared with return type `[SelfNestedBadReturnType; 4]`.
    // CHECK:STDERR:   fn F(x: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fn F(x: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> [SelfNestedBadParam; 4];
  }
}

// CHECK:STDOUT: --- fail_impl_bad_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self.1: I = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @I, F [template]
// CHECK:STDOUT:   %.4: <associated F in I> = assoc_entity element0, @I.%F.decl [template]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [template]
// CHECK:STDOUT:   %F.2: type = class_type @F.16 [template]
// CHECK:STDOUT:   %PossiblyF: type = fn_type @PossiblyF [template]
// CHECK:STDOUT:   %struct.2: PossiblyF = struct_value () [template]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [template]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [template]
// CHECK:STDOUT:   %F.3: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [template]
// CHECK:STDOUT:   %F.4: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %struct.4: F = struct_value () [template]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [template]
// CHECK:STDOUT:   %F.5: type = fn_type @F.4 [template]
// CHECK:STDOUT:   %struct.5: F = struct_value () [template]
// CHECK:STDOUT:   %.6: type = interface_type @J [template]
// CHECK:STDOUT:   %Self.2: J = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.6: type = fn_type @F.5 [template]
// CHECK:STDOUT:   %struct.6: F = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type @J, F [template]
// CHECK:STDOUT:   %.8: <associated F in J> = assoc_entity element0, @J.%F.decl [template]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [template]
// CHECK:STDOUT:   %F.7: type = fn_type @F.6 [template]
// CHECK:STDOUT:   %struct.7: F = struct_value () [template]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [template]
// CHECK:STDOUT:   %F.8: type = fn_type @F.7 [template]
// CHECK:STDOUT:   %struct.8: F = struct_value () [template]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [template]
// CHECK:STDOUT:   %F.9: type = fn_type @F.8 [template]
// CHECK:STDOUT:   %struct.9: F = struct_value () [template]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [template]
// CHECK:STDOUT:   %F.10: type = fn_type @F.9 [template]
// CHECK:STDOUT:   %struct.10: F = struct_value () [template]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [template]
// CHECK:STDOUT:   %F.11: type = fn_type @F.10 [template]
// CHECK:STDOUT:   %struct.11: F = struct_value () [template]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [template]
// CHECK:STDOUT:   %F.12: type = fn_type @F.11 [template]
// CHECK:STDOUT:   %struct.12: F = struct_value () [template]
// CHECK:STDOUT:   %FDifferentParamName: type = class_type @FDifferentParamName [template]
// CHECK:STDOUT:   %F.13: type = fn_type @F.12 [template]
// CHECK:STDOUT:   %struct.13: F = struct_value () [template]
// CHECK:STDOUT:   %.9: type = interface_type @SelfNested [template]
// CHECK:STDOUT:   %Self.3: SelfNested = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.10: type = ptr_type Self [symbolic]
// CHECK:STDOUT:   %.11: type = struct_type {.x: Self, .y: i32} [symbolic]
// CHECK:STDOUT:   %.12: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.13: type = tuple_type (Self*, {.x: Self, .y: i32}) [symbolic]
// CHECK:STDOUT:   %.14: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.15: type = array_type %.14, Self [symbolic]
// CHECK:STDOUT:   %F.14: type = fn_type @F.13 [template]
// CHECK:STDOUT:   %struct.14: F = struct_value () [template]
// CHECK:STDOUT:   %.16: type = assoc_entity_type @SelfNested, F [template]
// CHECK:STDOUT:   %.17: <associated F in SelfNested> = assoc_entity element0, @SelfNested.%F.decl [template]
// CHECK:STDOUT:   %SelfNestedBadParam: type = class_type @SelfNestedBadParam [template]
// CHECK:STDOUT:   %.18: type = ptr_type SelfNestedBadParam [template]
// CHECK:STDOUT:   %.19: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.20: type = tuple_type (SelfNestedBadParam*, {.x: i32, .y: i32}) [template]
// CHECK:STDOUT:   %.21: type = array_type %.14, SelfNestedBadParam [template]
// CHECK:STDOUT:   %F.15: type = fn_type @F.14 [template]
// CHECK:STDOUT:   %struct.15: F = struct_value () [template]
// CHECK:STDOUT:   %.22: type = struct_type {.x: SelfNestedBadParam, .y: i32} [template]
// CHECK:STDOUT:   %.23: type = tuple_type (SelfNestedBadParam*, {.x: SelfNestedBadParam, .y: i32}) [template]
// CHECK:STDOUT:   %SelfNestedBadReturnType: type = class_type @SelfNestedBadReturnType [template]
// CHECK:STDOUT:   %.24: type = ptr_type SelfNestedBadReturnType [template]
// CHECK:STDOUT:   %.25: type = struct_type {.x: SelfNestedBadReturnType, .y: i32} [template]
// CHECK:STDOUT:   %.26: type = tuple_type (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32}) [template]
// CHECK:STDOUT:   %F.16: type = fn_type @F.15 [template]
// CHECK:STDOUT:   %struct.16: F = struct_value () [template]
// CHECK:STDOUT:   %.27: type = array_type %.14, SelfNestedBadReturnType [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
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
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   %NoF.decl: type = class_decl @NoF [template = constants.%NoF] {}
// CHECK:STDOUT:   %FNotFunction.decl: type = class_decl @FNotFunction [template = constants.%FNotFunction] {}
// CHECK:STDOUT:   %PossiblyF.decl: PossiblyF = fn_decl @PossiblyF [template = constants.%struct.2] {}
// CHECK:STDOUT:   %FAlias.decl: type = class_decl @FAlias [template = constants.%FAlias] {}
// CHECK:STDOUT:   %FExtraParam.decl: type = class_decl @FExtraParam [template = constants.%FExtraParam] {}
// CHECK:STDOUT:   %FExtraImplicitParam.decl: type = class_decl @FExtraImplicitParam [template = constants.%FExtraImplicitParam] {}
// CHECK:STDOUT:   %FExtraReturnType.decl: type = class_decl @FExtraReturnType [template = constants.%FExtraReturnType] {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%.6] {}
// CHECK:STDOUT:   %FMissingParam.decl: type = class_decl @FMissingParam [template = constants.%FMissingParam] {}
// CHECK:STDOUT:   %FMissingImplicitParam.decl: type = class_decl @FMissingImplicitParam [template = constants.%FMissingImplicitParam] {}
// CHECK:STDOUT:   %FMissingReturnType.decl: type = class_decl @FMissingReturnType [template = constants.%FMissingReturnType] {}
// CHECK:STDOUT:   %FDifferentParamType.decl: type = class_decl @FDifferentParamType [template = constants.%FDifferentParamType] {}
// CHECK:STDOUT:   %FDifferentImplicitParamType.decl: type = class_decl @FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType] {}
// CHECK:STDOUT:   %FDifferentReturnType.decl: type = class_decl @FDifferentReturnType [template = constants.%FDifferentReturnType] {}
// CHECK:STDOUT:   %FDifferentParamName.decl: type = class_decl @FDifferentParamName [template = constants.%FDifferentParamName] {}
// CHECK:STDOUT:   %SelfNested.decl: type = interface_decl @SelfNested [template = constants.%.9] {}
// CHECK:STDOUT:   %SelfNestedBadParam.decl: type = class_decl @SelfNestedBadParam [template = constants.%SelfNestedBadParam] {}
// CHECK:STDOUT:   %SelfNestedBadReturnType.decl: type = class_decl @SelfNestedBadReturnType [template = constants.%SelfNestedBadReturnType] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.loc7: <associated F in I> = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc7
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: J = bind_symbolic_name Self 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.5 [template = constants.%struct.6] {
// CHECK:STDOUT:     %self.loc89_20.1: bool = param self
// CHECK:STDOUT:     %self.loc89_20.2: bool = bind_name self, %self.loc89_20.1
// CHECK:STDOUT:     %b.loc89_32.1: bool = param b
// CHECK:STDOUT:     %b.loc89_32.2: bool = bind_name b, %b.loc89_32.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc89: <associated F in J> = assoc_entity element0, %F.decl [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc89
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @SelfNested {
// CHECK:STDOUT:   %Self: SelfNested = bind_symbolic_name Self 0 [symbolic = constants.%Self.3]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.13 [template = constants.%struct.14] {
// CHECK:STDOUT:     %Self.ref.loc184_12: SelfNested = name_ref Self, %Self [symbolic = constants.%Self.3]
// CHECK:STDOUT:     %.loc184_16.1: type = facet_type_access %Self.ref.loc184_12 [symbolic = constants.%Self.3]
// CHECK:STDOUT:     %.loc184_16.2: type = converted %Self.ref.loc184_12, %.loc184_16.1 [symbolic = constants.%Self.3]
// CHECK:STDOUT:     %.loc184_16.3: type = ptr_type Self [symbolic = constants.%.10]
// CHECK:STDOUT:     %Self.ref.loc184_24: SelfNested = name_ref Self, %Self [symbolic = constants.%Self.3]
// CHECK:STDOUT:     %.loc184_24.1: type = facet_type_access %Self.ref.loc184_24 [symbolic = constants.%Self.3]
// CHECK:STDOUT:     %.loc184_24.2: type = converted %Self.ref.loc184_24, %.loc184_24.1 [symbolic = constants.%Self.3]
// CHECK:STDOUT:     %.loc184_37: type = struct_type {.x: Self, .y: i32} [symbolic = constants.%.11]
// CHECK:STDOUT:     %.loc184_38.1: (type, type) = tuple_literal (%.loc184_16.3, %.loc184_37)
// CHECK:STDOUT:     %.loc184_38.2: type = converted %.loc184_38.1, constants.%.13 [symbolic = constants.%.13]
// CHECK:STDOUT:     %x.loc184_8.1: (Self*, {.x: Self, .y: i32}) = param x
// CHECK:STDOUT:     %x.loc184_8.2: (Self*, {.x: Self, .y: i32}) = bind_name x, %x.loc184_8.1
// CHECK:STDOUT:     %Self.ref.loc184_45: SelfNested = name_ref Self, %Self [symbolic = constants.%Self.3]
// CHECK:STDOUT:     %.loc184_51: i32 = int_literal 4 [template = constants.%.14]
// CHECK:STDOUT:     %.loc184_45.1: type = facet_type_access %Self.ref.loc184_45 [symbolic = constants.%Self.3]
// CHECK:STDOUT:     %.loc184_45.2: type = converted %Self.ref.loc184_45, %.loc184_45.1 [symbolic = constants.%Self.3]
// CHECK:STDOUT:     %.loc184_52: type = array_type %.loc184_51, Self [symbolic = constants.%.15]
// CHECK:STDOUT:     %return.var: ref [Self; 4] = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc184_53: <associated F in SelfNested> = assoc_entity element0, %F.decl [template = constants.%.17]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc184_53
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: NoF as I {
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: FNotFunction as I {
// CHECK:STDOUT:   %F.decl: type = class_decl @F.16 [template = constants.%F.2] {}
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: FAlias as I {
// CHECK:STDOUT:   %PossiblyF.ref: PossiblyF = name_ref PossiblyF, file.%PossiblyF.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %F: PossiblyF = bind_alias F, file.%PossiblyF.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: FExtraParam as I {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.3] {
// CHECK:STDOUT:     %b.loc58_10.1: bool = param b
// CHECK:STDOUT:     %b.loc58_10.2: bool = bind_name b, %b.loc58_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: FExtraImplicitParam as I {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.3 [template = constants.%struct.4] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [template = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self.loc71_10.1: FExtraImplicitParam = param self
// CHECK:STDOUT:     %self.loc71_10.2: FExtraImplicitParam = bind_name self, %self.loc71_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: FExtraReturnType as I {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.4 [template = constants.%struct.5] {
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: FMissingParam as J {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.6 [template = constants.%struct.7] {
// CHECK:STDOUT:     %self.loc100_10.1: bool = param self
// CHECK:STDOUT:     %self.loc100_10.2: bool = bind_name self, %self.loc100_10.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: FMissingImplicitParam as J {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.7 [template = constants.%struct.8] {
// CHECK:STDOUT:     %b.loc113_10.1: bool = param b
// CHECK:STDOUT:     %b.loc113_10.2: bool = bind_name b, %b.loc113_10.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: FMissingReturnType as J {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.8 [template = constants.%struct.9] {
// CHECK:STDOUT:     %self.loc126_10.1: bool = param self
// CHECK:STDOUT:     %self.loc126_10.2: bool = bind_name self, %self.loc126_10.1
// CHECK:STDOUT:     %b.loc126_22.1: bool = param b
// CHECK:STDOUT:     %b.loc126_22.2: bool = bind_name b, %b.loc126_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: FDifferentParamType as J {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.9 [template = constants.%struct.10] {
// CHECK:STDOUT:     %self.loc139_10.1: bool = param self
// CHECK:STDOUT:     %self.loc139_10.2: bool = bind_name self, %self.loc139_10.1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [template = constants.%FDifferentParamType]
// CHECK:STDOUT:     %b.loc139_22.1: FDifferentParamType = param b
// CHECK:STDOUT:     %b.loc139_22.2: FDifferentParamType = bind_name b, %b.loc139_22.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.11: FDifferentImplicitParamType as J {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.10 [template = constants.%struct.11] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self.loc152_10.1: FDifferentImplicitParamType = param self
// CHECK:STDOUT:     %self.loc152_10.2: FDifferentImplicitParamType = bind_name self, %self.loc152_10.1
// CHECK:STDOUT:     %b.loc152_22.1: bool = param b
// CHECK:STDOUT:     %b.loc152_22.2: bool = bind_name b, %b.loc152_22.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.12: FDifferentReturnType as J {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.11 [template = constants.%struct.12] {
// CHECK:STDOUT:     %self.loc165_10.1: bool = param self
// CHECK:STDOUT:     %self.loc165_10.2: bool = bind_name self, %self.loc165_10.1
// CHECK:STDOUT:     %b.loc165_22.1: bool = param b
// CHECK:STDOUT:     %b.loc165_22.2: bool = bind_name b, %b.loc165_22.1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [template = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %return.var: ref FDifferentReturnType = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.13: FDifferentParamName as J {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.12 [template = constants.%struct.13] {
// CHECK:STDOUT:     %self.loc179_10.1: bool = param self
// CHECK:STDOUT:     %self.loc179_10.2: bool = bind_name self, %self.loc179_10.1
// CHECK:STDOUT:     %not_b.loc179_22.1: bool = param not_b
// CHECK:STDOUT:     %not_b.loc179_22.2: bool = bind_name not_b, %not_b.loc179_22.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.14: SelfNestedBadParam as SelfNested {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.14 [template = constants.%struct.15] {
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc196_14: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %.loc196_32: type = ptr_type SelfNestedBadParam [template = constants.%.18]
// CHECK:STDOUT:     %.loc196_52: type = struct_type {.x: i32, .y: i32} [template = constants.%.19]
// CHECK:STDOUT:     %.loc196_53.1: (type, type) = tuple_literal (%.loc196_32, %.loc196_52)
// CHECK:STDOUT:     %.loc196_53.2: type = converted %.loc196_53.1, constants.%.20 [template = constants.%.20]
// CHECK:STDOUT:     %x.loc196_10.1: (SelfNestedBadParam*, {.x: i32, .y: i32}) = param x
// CHECK:STDOUT:     %x.loc196_10.2: (SelfNestedBadParam*, {.x: i32, .y: i32}) = bind_name x, %x.loc196_10.1
// CHECK:STDOUT:     %SelfNestedBadParam.ref.loc196_60: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %.loc196_80: i32 = int_literal 4 [template = constants.%.14]
// CHECK:STDOUT:     %.loc196_81: type = array_type %.loc196_80, SelfNestedBadParam [template = constants.%.21]
// CHECK:STDOUT:     %return.var: ref [SelfNestedBadParam; 4] = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.15: SelfNestedBadReturnType as SelfNested {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.15 [template = constants.%struct.16] {
// CHECK:STDOUT:     %SelfNestedBadReturnType.ref.loc208_14: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %.loc208_37: type = ptr_type SelfNestedBadReturnType [template = constants.%.24]
// CHECK:STDOUT:     %SelfNestedBadReturnType.ref.loc208_45: type = name_ref SelfNestedBadReturnType, file.%SelfNestedBadReturnType.decl [template = constants.%SelfNestedBadReturnType]
// CHECK:STDOUT:     %.loc208_77: type = struct_type {.x: SelfNestedBadReturnType, .y: i32} [template = constants.%.25]
// CHECK:STDOUT:     %.loc208_78.1: (type, type) = tuple_literal (%.loc208_37, %.loc208_77)
// CHECK:STDOUT:     %.loc208_78.2: type = converted %.loc208_78.1, constants.%.26 [template = constants.%.26]
// CHECK:STDOUT:     %x.loc208_10.1: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32}) = param x
// CHECK:STDOUT:     %x.loc208_10.2: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32}) = bind_name x, %x.loc208_10.1
// CHECK:STDOUT:     %SelfNestedBadParam.ref: type = name_ref SelfNestedBadParam, file.%SelfNestedBadParam.decl [template = constants.%SelfNestedBadParam]
// CHECK:STDOUT:     %.loc208_105: i32 = int_literal 4 [template = constants.%.14]
// CHECK:STDOUT:     %.loc208_106: type = array_type %.loc208_105, SelfNestedBadParam [template = constants.%.21]
// CHECK:STDOUT:     %return.var: ref [SelfNestedBadParam; 4] = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoF {
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NoF
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FNotFunction {
// CHECK:STDOUT:   impl_decl @impl.2 {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FNotFunction
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F.16;
// CHECK:STDOUT:
// CHECK:STDOUT: class @FAlias {
// CHECK:STDOUT:   impl_decl @impl.3 {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FAlias
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraParam {
// CHECK:STDOUT:   impl_decl @impl.4 {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraImplicitParam {
// CHECK:STDOUT:   impl_decl @impl.5 {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraImplicitParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FExtraReturnType {
// CHECK:STDOUT:   impl_decl @impl.6 {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FExtraReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingParam {
// CHECK:STDOUT:   impl_decl @impl.7 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingImplicitParam {
// CHECK:STDOUT:   impl_decl @impl.8 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingImplicitParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingReturnType {
// CHECK:STDOUT:   impl_decl @impl.9 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamType {
// CHECK:STDOUT:   impl_decl @impl.10 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentImplicitParamType {
// CHECK:STDOUT:   impl_decl @impl.11 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentImplicitParamType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentReturnType {
// CHECK:STDOUT:   impl_decl @impl.12 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamName {
// CHECK:STDOUT:   impl_decl @impl.13 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamName
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadParam {
// CHECK:STDOUT:   impl_decl @impl.14 {
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SelfNestedBadReturnType {
// CHECK:STDOUT:   impl_decl @impl.15 {
// CHECK:STDOUT:     %SelfNested.ref: type = name_ref SelfNested, file.%SelfNested.decl [template = constants.%.9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SelfNestedBadReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PossiblyF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(@impl.4.%b.loc58_10.2: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[@impl.5.%self.loc71_10.2: FExtraImplicitParam]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.4() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5[@J.%self.loc89_20.2: bool](@J.%b.loc89_32.2: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6[@impl.7.%self.loc100_10.2: bool]() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.7(@impl.8.%b.loc113_10.2: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.8[@impl.9.%self.loc126_10.2: bool](@impl.9.%b.loc126_22.2: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.9[@impl.10.%self.loc139_10.2: bool](@impl.10.%b.loc139_22.2: FDifferentParamType) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.10[@impl.11.%self.loc152_10.2: FDifferentImplicitParamType](@impl.11.%b.loc152_22.2: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.11[@impl.12.%self.loc165_10.2: bool](@impl.12.%b.loc165_22.2: bool) -> FDifferentReturnType;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.12[@impl.13.%self.loc179_10.2: bool](@impl.13.%not_b.loc179_22.2: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.13(@SelfNested.%x.loc184_8.2: (Self*, {.x: Self, .y: i32})) -> [Self; 4];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.14(@impl.14.%x.loc196_10.2: (SelfNestedBadParam*, {.x: i32, .y: i32})) -> [SelfNestedBadParam; 4];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.15(@impl.15.%x.loc208_10.2: (SelfNestedBadReturnType*, {.x: SelfNestedBadReturnType, .y: i32})) -> [SelfNestedBadParam; 4];
// CHECK:STDOUT:
