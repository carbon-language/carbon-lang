// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/partial.carbon

// --- base.carbon
library "[[@TEST_NAME]]";

base class C { }

//@dump-sem-ir-begin
fn A(p: partial C);
//@dump-sem-ir-end

// --- abstract.carbon
library "[[@TEST_NAME]]";

abstract class C { }

//@dump-sem-ir-begin
fn A(p: partial C);
//@dump-sem-ir-end

// --- fail_partial_nondynamic.carbon
library "[[@TEST_NAME]]";

class C { }

//@dump-sem-ir-begin
// CHECK:STDERR: fail_partial_nondynamic.carbon:[[@LINE+4]]:9: error: `partial` applied to final type `C` [PartialOnFinal]
// CHECK:STDERR: fn G(p: partial C);
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR:
fn G(p: partial C);
//@dump-sem-ir-end

// --- fail_partial_final.carbon
library "[[@TEST_NAME]]";

base class Base { }
class Derived {
  extend base: Base;
}

//@dump-sem-ir-begin
// CHECK:STDERR: fail_partial_final.carbon:[[@LINE+4]]:9: error: `partial` applied to final type `Derived` [PartialOnFinal]
// CHECK:STDERR: fn G(p: partial Derived);
// CHECK:STDERR:         ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(p: partial Derived);
//@dump-sem-ir-end

// --- fail_partial_decl.carbon
library "[[@TEST_NAME]]";

class C;

// TODO: This diagnostic could be more specific - the type might be non-final,
// but since we only have a declaration, we don't know.
//@dump-sem-ir-begin
// applied to final type `C` [PartialOnFinal] CHECK:STDERR: fn G(p: partial C);
// CHECK:STDERR: fail_partial_decl.carbon:[[@LINE+4]]:9: error: `partial` applied to final type `C` [PartialOnFinal]
// CHECK:STDERR: fn G(p: partial C);
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR:
fn G(p: partial C);
//@dump-sem-ir-end

// --- fail_partial_tuple.carbon
library "[[@TEST_NAME]]";

class C;

//@dump-sem-ir-begin
// CHECK:STDERR: fail_partial_tuple.carbon:[[@LINE+4]]:9: error: `partial` applied to final type `(C, C)` [PartialOnFinal]
// CHECK:STDERR: fn G(p: partial (C, C));
// CHECK:STDERR:         ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(p: partial (C, C));
//@dump-sem-ir-end

// --- fail_partial_struct.carbon
library "[[@TEST_NAME]]";

class C;

//@dump-sem-ir-begin
// CHECK:STDERR: fail_partial_struct.carbon:[[@LINE+4]]:9: error: `partial` applied to final type `{.x: C}` [PartialOnFinal]
// CHECK:STDERR: fn G(p: partial {.x: C});
// CHECK:STDERR:         ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(p: partial {.x: C});
//@dump-sem-ir-end

// --- fail_duplicate.carbon
library "[[@TEST_NAME]]";

base class C { }

//@dump-sem-ir-begin
// CHECK:STDERR: fail_duplicate.carbon:[[@LINE+4]]:9: error: `partial` applied to final type `partial C` [PartialOnFinal]
// CHECK:STDERR: fn F(p: partial (partial C));
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(p: partial (partial C));
//@dump-sem-ir-end

// --- fail_convert_to_nonpartial.carbon
library "[[@TEST_NAME]]";

base class C { }

fn G(p: partial C*) -> C* {
  // CHECK:STDERR: fail_convert_to_nonpartial.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `partial C*` to `C*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_convert_to_nonpartial.carbon:[[@LINE+4]]:3: note: type `partial C*` does not implement interface `Core.ImplicitAs(C*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// --- fail_derive_from_partial.carbon
library "[[@TEST_NAME]]";

base class C { }

class Derived {
  // CHECK:STDERR: fail_derive_from_partial.carbon:[[@LINE+4]]:16: error: deriving from final type `partial C`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: partial C;
  // CHECK:STDERR:                ^~~~~~~~~
  // CHECK:STDERR:
  extend base: partial C;
}

// --- fail_todo_partial_template_dependent.carbon
library "[[@TEST_NAME]]";

// TODO: This should be accepted because `T` might be final once we know what it
// is.

// CHECK:STDERR: fail_todo_partial_template_dependent.carbon:[[@LINE+4]]:28: error: `partial` applied to final type `T` [PartialOnFinal]
// CHECK:STDERR: fn G[template T:! type](p: partial T*);
// CHECK:STDERR:                            ^~~~~~~~~
// CHECK:STDERR:
fn G[template T:! type](p: partial T*);

// --- fail_partial_generic.carbon
library "[[@TEST_NAME]]";

// TODO: Maybe rephrase this to use some common/generic phrasing to refer to the
// generic type and its requirements, as distinct from the concrete type that
// might be used here in any specific.

// CHECK:STDERR: fail_partial_generic.carbon:[[@LINE+4]]:19: error: `partial` applied to final type `T` [PartialOnFinal]
// CHECK:STDERR: fn F[T:! type](p: partial T*);
// CHECK:STDERR:                   ^~~~~~~~~
// CHECK:STDERR:
fn F[T:! type](p: partial T*);

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.43f [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %p.patt: %pattern_type = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.43f = value_param call_param0
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = constants.%.43f] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc6_9.2: type = partial_type %C.ref [concrete = constants.%.43f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.43f = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param: %.43f);
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.43f [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %p.patt: %pattern_type = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.43f = value_param call_param0
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = constants.%.43f] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc6_9.2: type = partial_type %C.ref [concrete = constants.%.43f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.43f = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param: %.43f);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_partial_nondynamic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.43f [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.43f = value_param call_param0
// CHECK:STDOUT:     %.loc10_9.1: type = splice_block %.loc10_9.2 [concrete = constants.%.43f] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc10_9.2: type = partial_type %C.ref [concrete = constants.%.43f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.43f = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %.43f);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_partial_final.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %.604: type = partial_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.604 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.604 = value_param call_param0
// CHECK:STDOUT:     %.loc13_9.1: type = splice_block %.loc13_9.2 [concrete = constants.%.604] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %.loc13_9.2: type = partial_type %Derived.ref [concrete = constants.%.604]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.604 = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %.604);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_partial_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.43f [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.43f = value_param call_param0
// CHECK:STDOUT:     %.loc13_9.1: type = splice_block %.loc13_9.2 [concrete = constants.%.43f] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc13_9.2: type = partial_type %C.ref [concrete = constants.%.43f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.43f = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %.43f);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_partial_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %tuple.type.56b: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %.62c: type = partial_type %tuple.type.56b [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.62c [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.62c = value_param call_param0
// CHECK:STDOUT:     %.loc10_9.1: type = splice_block %.loc10_9.3 [concrete = constants.%.62c] {
// CHECK:STDOUT:       %C.ref.loc10_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %C.ref.loc10_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc10_22: %tuple.type.24b = tuple_literal (%C.ref.loc10_18, %C.ref.loc10_21) [concrete = constants.%tuple]
// CHECK:STDOUT:       %.loc10_9.2: type = converted %.loc10_22, constants.%tuple.type.56b [concrete = constants.%tuple.type.56b]
// CHECK:STDOUT:       %.loc10_9.3: type = partial_type %.loc10_9.2 [concrete = constants.%.62c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.62c = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %.62c);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_partial_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %C} [concrete]
// CHECK:STDOUT:   %.1ed: type = partial_type %struct_type.x [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.1ed [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.1ed = value_param call_param0
// CHECK:STDOUT:     %.loc10_9.1: type = splice_block %.loc10_9.2 [concrete = constants.%.1ed] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %struct_type.x: type = struct_type {.x: %C} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:       %.loc10_9.2: type = partial_type %struct_type.x [concrete = constants.%.1ed]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.1ed = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %.1ed);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_duplicate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %.a34: type = partial_type %.43f [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.a34 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.patt: %pattern_type = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.a34 = value_param call_param0
// CHECK:STDOUT:     %.loc10_9.1: type = splice_block %.loc10_9.2 [concrete = constants.%.a34] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc10_18: type = partial_type %C.ref [concrete = constants.%.43f]
// CHECK:STDOUT:       %.loc10_9.2: type = partial_type %.loc10_18 [concrete = constants.%.a34]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.a34 = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %.a34);
// CHECK:STDOUT:
