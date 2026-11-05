// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/generic.carbon

// --- generic_definition.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[T: Core.Copy](x: T) -> auto { return x; }
//@dump-sem-ir-end

// --- call.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy](x: T) -> auto { return x; }

//@dump-sem-ir-begin
fn G() -> i32 { return F(0 as i32); }
//@dump-sem-ir-end

// --- call_from_generic.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy](x: T) -> auto { return x; }

//@dump-sem-ir-begin
fn G[U: Core.Copy](u: U) -> U { return F(u); }
//@dump-sem-ir-end

fn H() -> i32 { return G(0 as i32); }

// --- deduced_from_call.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy](x: T) -> auto { return x; }
fn G[U: Core.Copy](u: U) -> auto { return F(u); }

fn H() -> i32 { return G(0 as i32); }
fn I[V: Core.Copy](v: V) -> V { return G(v); }

// --- return_type_from_definition.carbon
library "[[@TEST_NAME]]";

// The return type doesn't appear in the declaration, so is only computed by the
// definition's eval block.
//@dump-sem-ir-begin
fn F[T: Core.Copy](x: T) -> auto { return (x, x); }
//@dump-sem-ir-end

fn G() -> (i32, i32) { return F(0 as i32); }
fn H[U: Core.Copy](u: U) -> (U, U) { return F(u); }

// --- concrete_return_type.carbon
library "[[@TEST_NAME]]";

fn F[T: type](unused x: T) -> auto { return 0 as i32; }

fn G() -> i32 { return F(true); }
fn H[U: type](u: U) -> i32 { return F(u); }

// --- local_class.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy](x: T) -> auto {
  class C {
    var v: T;
  }
  return {.v = x} as C;
}

fn G() -> i32 {
  return F(0 as i32).v;
}

// --- member_of_generic_class.carbon
library "[[@TEST_NAME]]";

class C(T: Core.Copy) {
  fn Get(self) -> auto { return self.x; }
  var x: T;
}

fn G(c: C(i32)) -> i32 { return c.Get(); }

// --- recursion_after_returned_var.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy & Core.Destroy](x: T, b: bool) -> auto {
  returned var r: T = x;
  if (b) {
    let unused y: T = F(x, false);
  }
  return var;
}

fn G() -> i32 { return F(0 as i32, true); }

// --- fail_recursion_before_deduced.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy](x: T, b: bool) -> auto {
  if (b) {
    // CHECK:STDERR: fail_recursion_before_deduced.carbon:[[@LINE+3]]:12-22: error: call to `F` before its return type is known [CallBeforeReturnTypeDeduced]
    // CHECK:STDERR: fail_recursion_before_deduced.carbon:[[@LINE-3]]:1-43: note: return type of `F` is deduced from its definition [ReturnTypeDeducedFromDefinition]
    // CHECK:STDERR:
    return F(x, false);
  }
  return x;
}

// --- fail_recursion_with_different_arguments.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy](x: T, b: bool) -> auto {
  returned var r: T = x;
  if (b) {
    // CHECK:STDERR: fail_recursion_with_different_arguments.carbon:[[@LINE+3]]:5-22: error: call to `F` before its return type is known [CallBeforeReturnTypeDeduced]
    // CHECK:STDERR: fail_recursion_with_different_arguments.carbon:[[@LINE-4]]:1-43: note: return type of `F` is deduced from its definition [ReturnTypeDeducedFromDefinition]
    // CHECK:STDERR:
    F(0 as i32, false);
  }
  return var;
}

// CHECK:STDOUT: --- generic_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.2c66b4.2 = wrapper_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2f9c3.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.8bf: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2f9c3.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb5: %.8bf = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ac: <specific function> = specific_impl_function %impl.elem0.bb5, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_7.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_7.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:     %x.param_patt.loc4_21.1: @F.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc4_21.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc4_21.1: @F.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc4_21.1 [symbolic = %x.patt.loc4_21.2 (constants.%x.patt)]
// CHECK:STDOUT:     %return.param_patt.loc4_43.1: @F.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc4_43.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:     %return.patt.loc4_43.1: @F.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc4_43.1, %.loc4_43.6 [symbolic = %return.patt.loc4_43.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_7.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.param: @F.%T.as_type.loc4_23.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_23.1: type = splice_block %.loc4_23.2 [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc4_23.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc4_23.2: type = converted %T.ref, %T.as_type.loc4_23.2 [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.as_type.loc4_23.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc4_43.6: type = type_literal constants.%T.as_type [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc4_43.7: Core.Form = init_form %.loc4_43.6 [symbolic = %.loc4_43.4 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %return.param: ref @F.%T.as_type.loc4_23.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.as_type.loc4_23.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_7.2: %Copy.type) {
// CHECK:STDOUT:   %T.patt.loc4_7.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_7.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   %T.loc4_7.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc4_23.1: type = facet_access_type %T.loc4_7.1 [symbolic = %T.as_type.loc4_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc4_23.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %x.param_patt.loc4_21.2: @F.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc4_21.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc4_21.2: @F.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc4_21.2 [symbolic = %x.patt.loc4_21.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc4_23.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %.loc4_43.4: Core.Form = init_form %T.as_type.loc4_23.1 [symbolic = %.loc4_43.4 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc4_43.2: @F.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc4_43.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc4_43.2: @F.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc4_43.2, %T.as_type.loc4_23.1 [symbolic = %return.patt.loc4_43.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc4_7.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2f9c3.2)]
// CHECK:STDOUT:   %.loc4_43.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc4_7.1 [symbolic = %.loc4_43.5 (constants.%.8bf)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc4_7.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc4_43.2: @F.%.loc4_43.5 (%.8bf) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc4_43.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:   %specific_impl_fn.loc4_43.2: <specific function> = specific_impl_function %impl.elem0.loc4_43.2, @Copy.WithSelf.Op(%T.loc4_7.1) [symbolic = %specific_impl_fn.loc4_43.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.as_type.loc4_23.1 (%T.as_type)) -> out %return.param: @F.%T.as_type.loc4_23.1 (%T.as_type) [deduced] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.as_type.loc4_23.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc4_44: init @F.%T.as_type.loc4_23.1 (%T.as_type) = splice_block %Copy.WithSelf.Op.call {
// CHECK:STDOUT:       %impl.elem0.loc4_43.1: @F.%.loc4_43.5 (%.8bf) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc4_43.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:       %bound_method.loc4_43.1: <bound method> = bound_method %x.ref, %impl.elem0.loc4_43.1
// CHECK:STDOUT:       %.loc4_43.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %.loc4_43.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc4_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %specific_impl_fn.loc4_43.1: <specific function> = specific_impl_function %impl.elem0.loc4_43.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc4_43.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:       %bound_method.loc4_43.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc4_43.1
// CHECK:STDOUT:       %.loc4_43.3: ref @F.%T.as_type.loc4_23.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:       %Copy.WithSelf.Op.call: init @F.%T.as_type.loc4_23.1 (%T.as_type) to %.loc4_43.3 = call %bound_method.loc4_43.2(%x.ref)
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return %.loc4_44 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc4_7.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc4_23.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %x.param_patt.loc4_21.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc4_21.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.b93: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.396: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.84a: %Core.IntLiteral.as.As.impl.Convert.type.396 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e09: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b93 = facet_value Core.IntLiteral, (%As.impl_witness.e09) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2be: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %As.WithSelf.Convert.type.2be, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b0: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.94f: %Core.IntLiteral.as.As.impl.Convert.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.As.impl.Convert.94f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.94f, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Copy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.9e0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.396) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.84a)]
// CHECK:STDOUT:   %As.impl_witness_table.1c2 = impl_witness_table (%Core.import_ref.9e0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc6_11 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_11: Core.Form = init_form %i32.loc6_11 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %i32.loc6_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.979 = impl_witness_access constants.%As.impl_witness.e09, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.94f]
// CHECK:STDOUT:   %bound_method.loc6_28.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_28.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc6_28.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc6_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc6_28.2: %i32 = converted %int_0, %.loc6_28.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %Copy.facet.loc6_34.1: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc6_34.1: %Copy.type = converted constants.%i32, %Copy.facet.loc6_34.1 [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %Copy.facet.loc6_34.2: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %.loc6_34.2: %Copy.type = converted constants.%i32, %Copy.facet.loc6_34.2 [concrete = constants.%Copy.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Copy.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc6_28.2)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_from_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.f84: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type.c1c: type = facet_access_type %U.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.3: type = pattern_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %u.param_patt.f7a: %pattern_type.2c66b4.3 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.659: %pattern_type.2c66b4.3 = wrapper_binding_pattern u, %u.param_patt.f7a [symbolic]
// CHECK:STDOUT:   %.1ce700.3: Core.Form = init_form %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.3: %pattern_type.2c66b4.3 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f449d7.2: %pattern_type.2c66b4.3 = return_slot_pattern %return.param_patt.bf4475.3, %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83f06.2: <witness> = require_complete_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ea4: <specific function> = specific_function %F, @F(%U.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %u.param_patt.68b: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %u.patt.5ff: %pattern_type.d94 = wrapper_binding_pattern u, %u.param_patt.68b [concrete]
// CHECK:STDOUT:   %F.specific_fn.37a: <specific function> = specific_function %F, @F(%Copy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %U.patt.loc6_7.1: %pattern_type.e81 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_7.2 (constants.%U.patt.fe6)]
// CHECK:STDOUT:     %u.param_patt.loc6_21.1: @G.%pattern_type (%pattern_type.2c66b4.3) = value_param_pattern [symbolic = %u.param_patt.loc6_21.2 (constants.%u.param_patt.f7a)]
// CHECK:STDOUT:     %u.patt.loc6_21.1: @G.%pattern_type (%pattern_type.2c66b4.3) = wrapper_binding_pattern u, %u.param_patt.loc6_21.1 [symbolic = %u.patt.loc6_21.2 (constants.%u.patt.659)]
// CHECK:STDOUT:     %return.param_patt.loc6_29.1: @G.%pattern_type (%pattern_type.2c66b4.3) = out_param_pattern [symbolic = %return.param_patt.loc6_29.2 (constants.%return.param_patt.bf4475.3)]
// CHECK:STDOUT:     %return.patt.loc6_26.1: @G.%pattern_type (%pattern_type.2c66b4.3) = return_slot_pattern %return.param_patt.loc6_29.1, %.loc6_29.3 [symbolic = %return.patt.loc6_26.2 (constants.%return.patt.f449d7.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc6_29: %Copy.type = name_ref U, %U.loc6_7.2 [symbolic = %U.loc6_7.1 (constants.%U.f84)]
// CHECK:STDOUT:     %U.as_type.loc6_29: type = facet_access_type %U.ref.loc6_29 [symbolic = %U.as_type.loc6_23.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     %.loc6_29.3: type = converted %U.ref.loc6_29, %U.as_type.loc6_29 [symbolic = %U.as_type.loc6_23.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     %.loc6_29.4: Core.Form = init_form %.loc6_29.3 [symbolic = %.loc6_29.2 (constants.%.1ce700.3)]
// CHECK:STDOUT:     %.loc6_13: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc6_7.2: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc6_7.1 (constants.%U.f84)]
// CHECK:STDOUT:     %u.param: @G.%U.as_type.loc6_23.1 (%U.as_type.c1c) = value_param call_param0
// CHECK:STDOUT:     %.loc6_23.1: type = splice_block %.loc6_23.2 [symbolic = %U.as_type.loc6_23.1 (constants.%U.as_type.c1c)] {
// CHECK:STDOUT:       %U.ref.loc6_23: %Copy.type = name_ref U, %U.loc6_7.2 [symbolic = %U.loc6_7.1 (constants.%U.f84)]
// CHECK:STDOUT:       %U.as_type.loc6_23.2: type = facet_access_type %U.ref.loc6_23 [symbolic = %U.as_type.loc6_23.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:       %.loc6_23.2: type = converted %U.ref.loc6_23, %U.as_type.loc6_23.2 [symbolic = %U.as_type.loc6_23.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @G.%U.as_type.loc6_23.1 (%U.as_type.c1c) = wrapper_binding u, %u.param
// CHECK:STDOUT:     %return.param: ref @G.%U.as_type.loc6_23.1 (%U.as_type.c1c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @G.%U.as_type.loc6_23.1 (%U.as_type.c1c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%U.loc6_7.2: %Copy.type) {
// CHECK:STDOUT:   %U.patt.loc6_7.2: %pattern_type.e81 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_7.2 (constants.%U.patt.fe6)]
// CHECK:STDOUT:   %U.loc6_7.1: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc6_7.1 (constants.%U.f84)]
// CHECK:STDOUT:   %U.as_type.loc6_23.1: type = facet_access_type %U.loc6_7.1 [symbolic = %U.as_type.loc6_23.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc6_23.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.3)]
// CHECK:STDOUT:   %u.param_patt.loc6_21.2: @G.%pattern_type (%pattern_type.2c66b4.3) = value_param_pattern [symbolic = %u.param_patt.loc6_21.2 (constants.%u.param_patt.f7a)]
// CHECK:STDOUT:   %u.patt.loc6_21.2: @G.%pattern_type (%pattern_type.2c66b4.3) = wrapper_binding_pattern u, %u.param_patt.loc6_21.2 [symbolic = %u.patt.loc6_21.2 (constants.%u.patt.659)]
// CHECK:STDOUT:   %.loc6_29.2: Core.Form = init_form %U.as_type.loc6_23.1 [symbolic = %.loc6_29.2 (constants.%.1ce700.3)]
// CHECK:STDOUT:   %return.param_patt.loc6_29.2: @G.%pattern_type (%pattern_type.2c66b4.3) = out_param_pattern [symbolic = %return.param_patt.loc6_29.2 (constants.%return.param_patt.bf4475.3)]
// CHECK:STDOUT:   %return.patt.loc6_26.2: @G.%pattern_type (%pattern_type.2c66b4.3) = return_slot_pattern %return.param_patt.loc6_29.2, %U.as_type.loc6_23.1 [symbolic = %return.patt.loc6_26.2 (constants.%return.patt.f449d7.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc6_23.1 [symbolic = %require_complete (constants.%require_complete.d83f06.2)]
// CHECK:STDOUT:   %F.specific_fn.loc6_40.2: <specific function> = specific_function constants.%F, @F(%U.loc6_7.1) [symbolic = %F.specific_fn.loc6_40.2 (constants.%F.specific_fn.ea4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @G.%U.as_type.loc6_23.1 (%U.as_type.c1c)) -> out %return.param: @G.%U.as_type.loc6_23.1 (%U.as_type.c1c) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %u.ref: @G.%U.as_type.loc6_23.1 (%U.as_type.c1c) = name_ref u, %u
// CHECK:STDOUT:     %.loc6_43.1: %Copy.type = converted constants.%U.as_type.c1c, constants.%U.f84 [symbolic = %U.loc6_7.1 (constants.%U.f84)]
// CHECK:STDOUT:     %.loc6_43.2: %Copy.type = converted constants.%U.as_type.c1c, constants.%U.f84 [symbolic = %U.loc6_7.1 (constants.%U.f84)]
// CHECK:STDOUT:     %F.specific_fn.loc6_40.1: <specific function> = specific_function %F.ref, @F(constants.%U.f84) [symbolic = %F.specific_fn.loc6_40.2 (constants.%F.specific_fn.ea4)]
// CHECK:STDOUT:     %.loc6_29.1: ref @G.%U.as_type.loc6_23.1 (%U.as_type.c1c) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @G.%U.as_type.loc6_23.1 (%U.as_type.c1c) to %.loc6_29.1 = call %F.specific_fn.loc6_40.1(%u.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%U.f84) {
// CHECK:STDOUT:   %U.patt.loc6_7.2 => constants.%U.patt.fe6
// CHECK:STDOUT:   %U.loc6_7.1 => constants.%U.f84
// CHECK:STDOUT:   %U.as_type.loc6_23.1 => constants.%U.as_type.c1c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.3
// CHECK:STDOUT:   %u.param_patt.loc6_21.2 => constants.%u.param_patt.f7a
// CHECK:STDOUT:   %u.patt.loc6_21.2 => constants.%u.patt.659
// CHECK:STDOUT:   %.loc6_29.2 => constants.%.1ce700.3
// CHECK:STDOUT:   %return.param_patt.loc6_29.2 => constants.%return.param_patt.bf4475.3
// CHECK:STDOUT:   %return.patt.loc6_26.2 => constants.%return.patt.f449d7.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%Copy.facet) {
// CHECK:STDOUT:   %U.patt.loc6_7.2 => constants.%U.patt.fe6
// CHECK:STDOUT:   %U.loc6_7.1 => constants.%Copy.facet
// CHECK:STDOUT:   %U.as_type.loc6_23.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %u.param_patt.loc6_21.2 => constants.%u.param_patt.68b
// CHECK:STDOUT:   %u.patt.loc6_21.2 => constants.%u.patt.5ff
// CHECK:STDOUT:   %.loc6_29.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc6_29.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc6_26.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %F.specific_fn.loc6_40.2 => constants.%F.specific_fn.37a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_type_from_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.1f4da3.1: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.cecdb7.1: %pattern_type.2c66b4.2 = wrapper_binding_pattern x, %x.param_patt.1f4da3.1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83f06.1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %tuple.type.db8bf7.1: type = tuple_type (%T.as_type, %T.as_type) [symbolic]
// CHECK:STDOUT:   %.58c1c6.1: Core.Form = init_form %tuple.type.db8bf7.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.aacade.1: type = pattern_type %tuple.type.db8bf7.1 [symbolic]
// CHECK:STDOUT:   %return.param_patt.1c7f00.1: %pattern_type.aacade.1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.e7a3ca.1: %pattern_type.aacade.1 = return_slot_pattern %return.param_patt.1c7f00.1, %tuple.type.db8bf7.1 [symbolic]
// CHECK:STDOUT:   %require_complete.b868a8.1: <witness> = require_complete_type %tuple.type.db8bf7.1 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c332.1: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2f9c3.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.8bf935.1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2f9c3.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb582d.1: %.8bf935.1 = impl_witness_access %Copy.lookup_impl_witness.d9c332.1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2acc20.1: <specific function> = specific_impl_function %impl.elem0.bb582d.1, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.35f: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %.30d: Core.Form = init_form %tuple.type.35f [concrete]
// CHECK:STDOUT:   %pattern_type.354: type = pattern_type %tuple.type.35f [concrete]
// CHECK:STDOUT:   %return.param_patt.46d: %pattern_type.354 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.d16: %pattern_type.354 = return_slot_pattern %return.param_patt.46d, %tuple.type.35f [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %x.param_patt.051: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0f7: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt.051 [concrete]
// CHECK:STDOUT:   %complete_type.fa7: <witness> = complete_type_witness %tuple.type.35f [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %U.f84: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type.c1c: type = facet_access_type %U.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.3: type = pattern_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %tuple.type.db8bf7.2: type = tuple_type (%U.as_type.c1c, %U.as_type.c1c) [symbolic]
// CHECK:STDOUT:   %.58c1c6.2: Core.Form = init_form %tuple.type.db8bf7.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.aacade.2: type = pattern_type %tuple.type.db8bf7.2 [symbolic]
// CHECK:STDOUT:   %return.param_patt.1c7f00.2: %pattern_type.aacade.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.e7a3ca.2: %pattern_type.aacade.2 = return_slot_pattern %return.param_patt.1c7f00.2, %tuple.type.db8bf7.2 [symbolic]
// CHECK:STDOUT:   %require_complete.d83f06.2: <witness> = require_complete_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %require_complete.b868a8.2: <witness> = require_complete_type %tuple.type.db8bf7.2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.1f4da3.2: %pattern_type.2c66b4.3 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.cecdb7.2: %pattern_type.2c66b4.3 = wrapper_binding_pattern x, %x.param_patt.1f4da3.2 [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2f9c3.3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.f84) [symbolic]
// CHECK:STDOUT:   %.8bf935.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2f9c3.3, %U.f84 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c332.2: <witness> = lookup_impl_witness %U.f84, @Copy [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb582d.2: %.8bf935.2 = impl_witness_access %Copy.lookup_impl_witness.d9c332.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2acc20.2: <specific function> = specific_impl_function %impl.elem0.bb582d.2, @Copy.WithSelf.Op(%U.f84) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:     %x.param_patt.loc6_21.1: @F.%pattern_type.loc6_21 (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc6_21.2 (constants.%x.param_patt.1f4da3.1)]
// CHECK:STDOUT:     %x.patt.loc6_21.1: @F.%pattern_type.loc6_21 (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc6_21.1 [symbolic = %x.patt.loc6_21.2 (constants.%x.patt.cecdb7.1)]
// CHECK:STDOUT:     %return.param_patt.loc6_48.1: @F.%pattern_type.loc6_48 (%pattern_type.aacade.1) = out_param_pattern [symbolic = %return.param_patt.loc6_48.2 (constants.%return.param_patt.1c7f00.1)]
// CHECK:STDOUT:     %return.patt.loc6_48.1: @F.%pattern_type.loc6_48 (%pattern_type.aacade.1) = return_slot_pattern %return.param_patt.loc6_48.1, %.loc6_48.6 [symbolic = %return.patt.loc6_48.2 (constants.%return.patt.e7a3ca.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_13: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.param: @F.%T.as_type.loc6_23.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc6_23.1: type = splice_block %.loc6_23.2 [symbolic = %T.as_type.loc6_23.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc6_23.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_23.2: type = converted %T.ref, %T.as_type.loc6_23.2 [symbolic = %T.as_type.loc6_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.as_type.loc6_23.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc6_48.6: type = type_literal constants.%tuple.type.db8bf7.1 [symbolic = %tuple.type (constants.%tuple.type.db8bf7.1)]
// CHECK:STDOUT:     %.loc6_48.7: Core.Form = init_form %.loc6_48.6 [symbolic = %.loc6_48.5 (constants.%.58c1c6.1)]
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.db8bf7.1) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.db8bf7.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: %Copy.type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   %T.loc6_7.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc6_23.1: type = facet_access_type %T.loc6_7.1 [symbolic = %T.as_type.loc6_23.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc6_21: type = pattern_type %T.as_type.loc6_23.1 [symbolic = %pattern_type.loc6_21 (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %x.param_patt.loc6_21.2: @F.%pattern_type.loc6_21 (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc6_21.2 (constants.%x.param_patt.1f4da3.1)]
// CHECK:STDOUT:   %x.patt.loc6_21.2: @F.%pattern_type.loc6_21 (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc6_21.2 [symbolic = %x.patt.loc6_21.2 (constants.%x.patt.cecdb7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_21: <witness> = require_complete_type %T.as_type.loc6_23.1 [symbolic = %require_complete.loc6_21 (constants.%require_complete.d83f06.1)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.as_type.loc6_23.1, %T.as_type.loc6_23.1) [symbolic = %tuple.type (constants.%tuple.type.db8bf7.1)]
// CHECK:STDOUT:   %.loc6_48.5: Core.Form = init_form %tuple.type [symbolic = %.loc6_48.5 (constants.%.58c1c6.1)]
// CHECK:STDOUT:   %pattern_type.loc6_48: type = pattern_type %tuple.type [symbolic = %pattern_type.loc6_48 (constants.%pattern_type.aacade.1)]
// CHECK:STDOUT:   %return.param_patt.loc6_48.2: @F.%pattern_type.loc6_48 (%pattern_type.aacade.1) = out_param_pattern [symbolic = %return.param_patt.loc6_48.2 (constants.%return.param_patt.1c7f00.1)]
// CHECK:STDOUT:   %return.patt.loc6_48.2: @F.%pattern_type.loc6_48 (%pattern_type.aacade.1) = return_slot_pattern %return.param_patt.loc6_48.2, %tuple.type [symbolic = %return.patt.loc6_48.2 (constants.%return.patt.e7a3ca.1)]
// CHECK:STDOUT:   %require_complete.loc6_48: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc6_48 (constants.%require_complete.b868a8.1)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc6_7.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2f9c3.2)]
// CHECK:STDOUT:   %.loc6_44.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc6_7.1 [symbolic = %.loc6_44.3 (constants.%.8bf935.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc6_7.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c332.1)]
// CHECK:STDOUT:   %impl.elem0.loc6_44.2: @F.%.loc6_44.3 (%.8bf935.1) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_44.2 (constants.%impl.elem0.bb582d.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_44.2: <specific function> = specific_impl_function %impl.elem0.loc6_44.2, @Copy.WithSelf.Op(%T.loc6_7.1) [symbolic = %specific_impl_fn.loc6_44.2 (constants.%specific_impl_fn.2acc20.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.as_type.loc6_23.1 (%T.as_type)) -> out %return.param: @F.%tuple.type (%tuple.type.db8bf7.1) [deduced] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref.loc6_44: @F.%T.as_type.loc6_23.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %x.ref.loc6_47: @F.%T.as_type.loc6_23.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_48.1: @F.%tuple.type (%tuple.type.db8bf7.1) = tuple_literal (%x.ref.loc6_44, %x.ref.loc6_47)
// CHECK:STDOUT:     %.loc6_49.1: init @F.%tuple.type (%tuple.type.db8bf7.1) = splice_block %.loc6_49.2 {
// CHECK:STDOUT:       %impl.elem0.loc6_44.1: @F.%.loc6_44.3 (%.8bf935.1) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c332.1, element0 [symbolic = %impl.elem0.loc6_44.2 (constants.%impl.elem0.bb582d.1)]
// CHECK:STDOUT:       %bound_method.loc6_44.1: <bound method> = bound_method %x.ref.loc6_44, %impl.elem0.loc6_44.1
// CHECK:STDOUT:       %.loc6_44.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc6_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %.loc6_44.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc6_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %specific_impl_fn.loc6_44.1: <specific function> = specific_impl_function %impl.elem0.loc6_44.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc6_44.2 (constants.%specific_impl_fn.2acc20.1)]
// CHECK:STDOUT:       %bound_method.loc6_44.2: <bound method> = bound_method %x.ref.loc6_44, %specific_impl_fn.loc6_44.1
// CHECK:STDOUT:       %tuple.elem0: ref @F.%T.as_type.loc6_23.1 (%T.as_type) = tuple_access %return.param, element0
// CHECK:STDOUT:       %Copy.WithSelf.Op.call.loc6_44: init @F.%T.as_type.loc6_23.1 (%T.as_type) to %tuple.elem0 = call %bound_method.loc6_44.2(%x.ref.loc6_44)
// CHECK:STDOUT:       %.loc6_48.2: init @F.%T.as_type.loc6_23.1 (%T.as_type) to %tuple.elem0 = in_place_init %Copy.WithSelf.Op.call.loc6_44
// CHECK:STDOUT:       %impl.elem0.loc6_47: @F.%.loc6_44.3 (%.8bf935.1) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c332.1, element0 [symbolic = %impl.elem0.loc6_44.2 (constants.%impl.elem0.bb582d.1)]
// CHECK:STDOUT:       %bound_method.loc6_47.1: <bound method> = bound_method %x.ref.loc6_47, %impl.elem0.loc6_47
// CHECK:STDOUT:       %.loc6_47.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc6_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %.loc6_47.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc6_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %specific_impl_fn.loc6_47: <specific function> = specific_impl_function %impl.elem0.loc6_47, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc6_44.2 (constants.%specific_impl_fn.2acc20.1)]
// CHECK:STDOUT:       %bound_method.loc6_47.2: <bound method> = bound_method %x.ref.loc6_47, %specific_impl_fn.loc6_47
// CHECK:STDOUT:       %tuple.elem1: ref @F.%T.as_type.loc6_23.1 (%T.as_type) = tuple_access %return.param, element1
// CHECK:STDOUT:       %Copy.WithSelf.Op.call.loc6_47: init @F.%T.as_type.loc6_23.1 (%T.as_type) to %tuple.elem1 = call %bound_method.loc6_47.2(%x.ref.loc6_47)
// CHECK:STDOUT:       %.loc6_48.3: init @F.%T.as_type.loc6_23.1 (%T.as_type) to %tuple.elem1 = in_place_init %Copy.WithSelf.Op.call.loc6_47
// CHECK:STDOUT:       %.loc6_48.4: init @F.%tuple.type (%tuple.type.db8bf7.1) to %return.param = tuple_init (%.loc6_48.2, %.loc6_48.3)
// CHECK:STDOUT:       %.loc6_49.2: init @F.%tuple.type (%tuple.type.db8bf7.1) = converted %.loc6_48.1, %.loc6_48.4
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return %.loc6_49.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc6_23.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc6_21 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %x.param_patt.loc6_21.2 => constants.%x.param_patt.1f4da3.1
// CHECK:STDOUT:   %x.patt.loc6_21.2 => constants.%x.patt.cecdb7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Copy.facet) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%Copy.facet
// CHECK:STDOUT:   %T.as_type.loc6_23.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc6_21 => constants.%pattern_type.d94
// CHECK:STDOUT:   %x.param_patt.loc6_21.2 => constants.%x.param_patt.051
// CHECK:STDOUT:   %x.patt.loc6_21.2 => constants.%x.patt.0f7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_21 => constants.%complete_type.f8a
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.35f
// CHECK:STDOUT:   %.loc6_48.5 => constants.%.30d
// CHECK:STDOUT:   %pattern_type.loc6_48 => constants.%pattern_type.354
// CHECK:STDOUT:   %return.param_patt.loc6_48.2 => constants.%return.param_patt.46d
// CHECK:STDOUT:   %return.patt.loc6_48.2 => constants.%return.patt.d16
// CHECK:STDOUT:   %require_complete.loc6_48 => constants.%complete_type.fa7
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.f76
// CHECK:STDOUT:   %.loc6_44.3 => constants.%.2b9
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.8af
// CHECK:STDOUT:   %impl.elem0.loc6_44.2 => constants.%Int.as.Copy.impl.Op.cae
// CHECK:STDOUT:   %specific_impl_fn.loc6_44.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%U.f84) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%U.f84
// CHECK:STDOUT:   %T.as_type.loc6_23.1 => constants.%U.as_type.c1c
// CHECK:STDOUT:   %pattern_type.loc6_21 => constants.%pattern_type.2c66b4.3
// CHECK:STDOUT:   %x.param_patt.loc6_21.2 => constants.%x.param_patt.1f4da3.2
// CHECK:STDOUT:   %x.patt.loc6_21.2 => constants.%x.patt.cecdb7.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_21 => constants.%require_complete.d83f06.2
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.db8bf7.2
// CHECK:STDOUT:   %.loc6_48.5 => constants.%.58c1c6.2
// CHECK:STDOUT:   %pattern_type.loc6_48 => constants.%pattern_type.aacade.2
// CHECK:STDOUT:   %return.param_patt.loc6_48.2 => constants.%return.param_patt.1c7f00.2
// CHECK:STDOUT:   %return.patt.loc6_48.2 => constants.%return.patt.e7a3ca.2
// CHECK:STDOUT:   %require_complete.loc6_48 => constants.%require_complete.b868a8.2
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d2f9c3.3
// CHECK:STDOUT:   %.loc6_44.3 => constants.%.8bf935.2
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.d9c332.2
// CHECK:STDOUT:   %impl.elem0.loc6_44.2 => constants.%impl.elem0.bb582d.2
// CHECK:STDOUT:   %specific_impl_fn.loc6_44.2 => constants.%specific_impl_fn.2acc20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
