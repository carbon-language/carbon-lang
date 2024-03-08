// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I { fn F(); }

class NoF {
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:3: ERROR: Missing implementation of F in impl of interface I.
  // CHECK:STDERR:   impl as I {}
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-6]]:15: Associated function F declared here.
  // CHECK:STDERR: interface I { fn F(); }
  // CHECK:STDERR:               ^~~~~~~
  impl as I {}
}

class FNotFunction {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: ERROR: Associated function F implemented by non-function.
    // CHECK:STDERR:     class F;
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-17]]:15: Associated function F declared here.
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    class F;
  }
}

fn PossiblyF();

// TODO: Should this be permitted?
class FAlias {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:11: ERROR: Associated function F implemented by non-function.
    // CHECK:STDERR:     alias F = PossiblyF;
    // CHECK:STDERR:           ^
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-32]]:15: Associated function F declared here.
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    alias F = PossiblyF;
  }
}

class FExtraParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: ERROR: Function redeclaration differs because of parameter count of 1.
    // CHECK:STDERR:     fn F(b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-44]]:15: Previously declared with parameter count of 0.
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    fn F(b: bool);
  }
}

class FExtraImplicitParam {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: ERROR: Function redeclaration differs because of implicit parameter count of 1.
    // CHECK:STDERR:     fn F[self: Self]();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-56]]:15: Previously declared with implicit parameter count of 0.
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    fn F[self: Self]();
  }
}

// TODO: Should this be permitted?
class FExtraReturnType {
  impl as I {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: ERROR: Function redeclaration differs because return type is `bool`.
    // CHECK:STDERR:     fn F() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-69]]:15: Previously declared with no return type.
    // CHECK:STDERR: interface I { fn F(); }
    // CHECK:STDERR:               ^~~~~~~
    fn F() -> bool;
  }
}

interface J { fn F[self: bool](b: bool) -> bool; }

class FMissingParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: ERROR: Function redeclaration differs because of parameter count of 0.
    // CHECK:STDERR:     fn F[self: bool]() -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-7]]:15: Previously declared with parameter count of 1.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fn F[self: bool]() -> bool;
  }
}

class FMissingImplicitParam {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: ERROR: Function redeclaration differs because of implicit parameter count of 0.
    // CHECK:STDERR:     fn F(b: bool) -> bool;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-19]]:15: Previously declared with implicit parameter count of 1.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fn F(b: bool) -> bool;
  }
}

class FMissingReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:5: ERROR: Function redeclaration differs because no return type is provided.
    // CHECK:STDERR:     fn F[self: bool](b: bool);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-31]]:15: Previously declared with return type `bool`.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fn F[self: bool](b: bool);
  }
}

class FDifferentParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:22: ERROR: Function redeclaration differs at parameter 1.
    // CHECK:STDERR:     fn F[self: bool](b: Self) -> bool;
    // CHECK:STDERR:                      ^
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-43]]:32: Previous declaration's corresponding parameter here.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^
    fn F[self: bool](b: Self) -> bool;
  }
}

class FDifferentImplicitParamType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:10: ERROR: Function redeclaration differs at implicit parameter 1.
    // CHECK:STDERR:     fn F[self: Self](b: bool) -> bool;
    // CHECK:STDERR:          ^~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-55]]:20: Previous declaration's corresponding implicit parameter here.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                    ^~~~
    fn F[self: Self](b: bool) -> bool;
  }
}

class FDifferentReturnType {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:31: ERROR: Function returns incomplete type `FDifferentReturnType`.
    // CHECK:STDERR:     fn F[self: bool](b: bool) -> Self;
    // CHECK:STDERR:                               ^~~~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-5]]:1: Class is incomplete within its definition.
    // CHECK:STDERR: class FDifferentReturnType {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fn F[self: bool](b: bool) -> Self;
  }
}

// TODO: This should probably be permitted.
class FDifferentParamName {
  impl as J {
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE+6]]:22: ERROR: Function redeclaration differs at parameter 1.
    // CHECK:STDERR:     fn F[self: bool](not_b: bool) -> bool;
    // CHECK:STDERR:                      ^~~~~
    // CHECK:STDERR: fail_impl_bad_assoc_fn.carbon:[[@LINE-80]]:32: Previous declaration's corresponding parameter here.
    // CHECK:STDERR: interface J { fn F[self: bool](b: bool) -> bool; }
    // CHECK:STDERR:                                ^
    fn F[self: bool](not_b: bool) -> bool;
  }
}

// CHECK:STDOUT: --- fail_impl_bad_assoc_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in I> = assoc_entity element0, @I.%F [template]
// CHECK:STDOUT:   %NoF: type = class_type @NoF [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT:   %FNotFunction: type = class_type @FNotFunction [template]
// CHECK:STDOUT:   %F: type = class_type @F.13 [template]
// CHECK:STDOUT:   %FAlias: type = class_type @FAlias [template]
// CHECK:STDOUT:   %FExtraParam: type = class_type @FExtraParam [template]
// CHECK:STDOUT:   %FExtraImplicitParam: type = class_type @FExtraImplicitParam [template]
// CHECK:STDOUT:   %FExtraReturnType: type = class_type @FExtraReturnType [template]
// CHECK:STDOUT:   %.5: type = interface_type @J [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @J, <function> [template]
// CHECK:STDOUT:   %.7: <associated <function> in J> = assoc_entity element0, @J.%F [template]
// CHECK:STDOUT:   %FMissingParam: type = class_type @FMissingParam [template]
// CHECK:STDOUT:   %FMissingImplicitParam: type = class_type @FMissingImplicitParam [template]
// CHECK:STDOUT:   %FMissingReturnType: type = class_type @FMissingReturnType [template]
// CHECK:STDOUT:   %FDifferentParamType: type = class_type @FDifferentParamType [template]
// CHECK:STDOUT:   %FDifferentImplicitParamType: type = class_type @FDifferentImplicitParamType [template]
// CHECK:STDOUT:   %FDifferentReturnType: type = class_type @FDifferentReturnType [template]
// CHECK:STDOUT:   %FDifferentParamName: type = class_type @FDifferentParamName [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .NoF = %NoF.decl
// CHECK:STDOUT:     .FNotFunction = %FNotFunction.decl
// CHECK:STDOUT:     .PossiblyF = %PossiblyF
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   %NoF.decl: type = class_decl @NoF [template = constants.%NoF] {}
// CHECK:STDOUT:   %FNotFunction.decl: type = class_decl @FNotFunction [template = constants.%FNotFunction] {}
// CHECK:STDOUT:   %PossiblyF: <function> = fn_decl @PossiblyF [template] {}
// CHECK:STDOUT:   %FAlias.decl: type = class_decl @FAlias [template = constants.%FAlias] {}
// CHECK:STDOUT:   %FExtraParam.decl: type = class_decl @FExtraParam [template = constants.%FExtraParam] {}
// CHECK:STDOUT:   %FExtraImplicitParam.decl: type = class_decl @FExtraImplicitParam [template = constants.%FExtraImplicitParam] {}
// CHECK:STDOUT:   %FExtraReturnType.decl: type = class_decl @FExtraReturnType [template = constants.%FExtraReturnType] {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%.5] {}
// CHECK:STDOUT:   %FMissingParam.decl: type = class_decl @FMissingParam [template = constants.%FMissingParam] {}
// CHECK:STDOUT:   %FMissingImplicitParam.decl: type = class_decl @FMissingImplicitParam [template = constants.%FMissingImplicitParam] {}
// CHECK:STDOUT:   %FMissingReturnType.decl: type = class_decl @FMissingReturnType [template = constants.%FMissingReturnType] {}
// CHECK:STDOUT:   %FDifferentParamType.decl: type = class_decl @FDifferentParamType [template = constants.%FDifferentParamType] {}
// CHECK:STDOUT:   %FDifferentImplicitParamType.decl: type = class_decl @FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType] {}
// CHECK:STDOUT:   %FDifferentReturnType.decl: type = class_decl @FDifferentReturnType [template = constants.%FDifferentReturnType] {}
// CHECK:STDOUT:   %FDifferentParamName.decl: type = class_decl @FDifferentParamName [template = constants.%FDifferentParamName] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {}
// CHECK:STDOUT:   %.loc7: <associated <function> in I> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc7
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: J = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.5 [template] {
// CHECK:STDOUT:     %self.loc83_20.1: bool = param self
// CHECK:STDOUT:     %self.loc83_20.2: bool = bind_name self, %self.loc83_20.1
// CHECK:STDOUT:     %b.loc83_32.1: bool = param b
// CHECK:STDOUT:     %b.loc83_32.2: bool = bind_name b, %b.loc83_32.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc83: <associated <function> in J> = assoc_entity element0, %F [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc83
// CHECK:STDOUT:   witness = (%F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: NoF as I {
// CHECK:STDOUT:   %.1: <witness> = interface_witness @I, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: FNotFunction as I {
// CHECK:STDOUT:   %F.decl: type = class_decl @F.13 [template = constants.%F] {}
// CHECK:STDOUT:   %.1: <witness> = interface_witness @I, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: FAlias as I {
// CHECK:STDOUT:   %PossiblyF.ref: <function> = name_ref PossiblyF, file.%PossiblyF [template = file.%PossiblyF]
// CHECK:STDOUT:   %F: <function> = bind_alias F, file.%PossiblyF [template = file.%PossiblyF]
// CHECK:STDOUT:   %.1: <witness> = interface_witness @I, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: FExtraParam as I {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {
// CHECK:STDOUT:     %b.loc54_10.1: bool = param b
// CHECK:STDOUT:     %b.loc54_10.2: bool = bind_name b, %b.loc54_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @I, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.5: FExtraImplicitParam as I {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.3 [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FExtraImplicitParam [template = constants.%FExtraImplicitParam]
// CHECK:STDOUT:     %self.loc66_10.1: FExtraImplicitParam = param self
// CHECK:STDOUT:     %self.loc66_10.2: FExtraImplicitParam = bind_name self, %self.loc66_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @I, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6: FExtraReturnType as I {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.4 [template] {
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @I, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7: FMissingParam as J {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.6 [template] {
// CHECK:STDOUT:     %self.loc93_10.1: bool = param self
// CHECK:STDOUT:     %self.loc93_10.2: bool = bind_name self, %self.loc93_10.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @J, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.8: FMissingImplicitParam as J {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.7 [template] {
// CHECK:STDOUT:     %b.loc105_10.1: bool = param b
// CHECK:STDOUT:     %b.loc105_10.2: bool = bind_name b, %b.loc105_10.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @J, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: FMissingReturnType as J {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.8 [template] {
// CHECK:STDOUT:     %self.loc117_10.1: bool = param self
// CHECK:STDOUT:     %self.loc117_10.2: bool = bind_name self, %self.loc117_10.1
// CHECK:STDOUT:     %b.loc117_22.1: bool = param b
// CHECK:STDOUT:     %b.loc117_22.2: bool = bind_name b, %b.loc117_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @J, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.10: FDifferentParamType as J {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.9 [template] {
// CHECK:STDOUT:     %self.loc129_10.1: bool = param self
// CHECK:STDOUT:     %self.loc129_10.2: bool = bind_name self, %self.loc129_10.1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentParamType [template = constants.%FDifferentParamType]
// CHECK:STDOUT:     %b.loc129_22.1: FDifferentParamType = param b
// CHECK:STDOUT:     %b.loc129_22.2: FDifferentParamType = bind_name b, %b.loc129_22.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @J, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.11: FDifferentImplicitParamType as J {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.10 [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentImplicitParamType [template = constants.%FDifferentImplicitParamType]
// CHECK:STDOUT:     %self.loc141_10.1: FDifferentImplicitParamType = param self
// CHECK:STDOUT:     %self.loc141_10.2: FDifferentImplicitParamType = bind_name self, %self.loc141_10.1
// CHECK:STDOUT:     %b.loc141_22.1: bool = param b
// CHECK:STDOUT:     %b.loc141_22.2: bool = bind_name b, %b.loc141_22.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @J, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.12: FDifferentReturnType as J {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.11 [template] {
// CHECK:STDOUT:     %self.loc153_10.1: bool = param self
// CHECK:STDOUT:     %self.loc153_10.2: bool = bind_name self, %self.loc153_10.1
// CHECK:STDOUT:     %b.loc153_22.1: bool = param b
// CHECK:STDOUT:     %b.loc153_22.2: bool = bind_name b, %b.loc153_22.1
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FDifferentReturnType [template = constants.%FDifferentReturnType]
// CHECK:STDOUT:     %return.var: ref FDifferentReturnType = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @J, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.13: FDifferentParamName as J {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.12 [template] {
// CHECK:STDOUT:     %self.loc166_10.1: bool = param self
// CHECK:STDOUT:     %self.loc166_10.2: bool = bind_name self, %self.loc166_10.1
// CHECK:STDOUT:     %not_b.loc166_22.1: bool = param not_b
// CHECK:STDOUT:     %not_b.loc166_22.2: bool = bind_name not_b, %not_b.loc166_22.1
// CHECK:STDOUT:     %return.var: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.1: <witness> = interface_witness @J, (<error>) [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
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
// CHECK:STDOUT: class @F.13;
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
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingImplicitParam {
// CHECK:STDOUT:   impl_decl @impl.8 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingImplicitParam
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FMissingReturnType {
// CHECK:STDOUT:   impl_decl @impl.9 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FMissingReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamType {
// CHECK:STDOUT:   impl_decl @impl.10 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentImplicitParamType {
// CHECK:STDOUT:   impl_decl @impl.11 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentImplicitParamType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentReturnType {
// CHECK:STDOUT:   impl_decl @impl.12 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentReturnType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FDifferentParamName {
// CHECK:STDOUT:   impl_decl @impl.13 {
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%.5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FDifferentParamName
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PossiblyF();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(@impl.4.%b.loc54_10.2: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[@impl.5.%self.loc66_10.2: FExtraImplicitParam]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.4() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.5[@J.%self.loc83_20.2: bool](@J.%b.loc83_32.2: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.6[@impl.7.%self.loc93_10.2: bool]() -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.7(@impl.8.%b.loc105_10.2: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.8[@impl.9.%self.loc117_10.2: bool](@impl.9.%b.loc117_22.2: bool);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.9[@impl.10.%self.loc129_10.2: bool](@impl.10.%b.loc129_22.2: FDifferentParamType) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.10[@impl.11.%self.loc141_10.2: FDifferentImplicitParamType](@impl.11.%b.loc141_22.2: bool) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.11[@impl.12.%self.loc153_10.2: bool](@impl.12.%b.loc153_22.2: bool) -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.12[@impl.13.%self.loc166_10.2: bool](@impl.13.%not_b.loc166_22.2: bool) -> bool;
// CHECK:STDOUT:
