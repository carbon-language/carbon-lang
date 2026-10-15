// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/form.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/form.carbon

// --- ref_auto.carbon
library "[[@TEST_NAME]]";

var g: i32 = 0;

//@dump-sem-ir-begin
fn F() -> ref auto {
  return g;
}

fn G() -> ref i32 {
  return F();
}
//@dump-sem-ir-end

// --- ref_auto_param.carbon
library "[[@TEST_NAME]]";

class C {
  var a: i32;
  var b: i32;
}

fn F(ref c: C, select_a: bool) -> ref auto {
  if (select_a) {
    return c.a;
  }
  return c.b;
}

fn G(ref c: C) -> ref i32 {
  return F(ref c, true);
}

// --- ref_auto_const.carbon
library "[[@TEST_NAME]]";

var g: const i32 = 0 as i32;

// The deduced type retains `const`.
fn F() -> ref auto {
  return g;
}

fn G() -> ref const i32 {
  return F();
}

// --- val_auto.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F() -> val auto {
  return 1 as i32;
}

fn G() {
  let unused x: i32 = F();
}
//@dump-sem-ir-end

// --- var_auto.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F() -> var auto {
  return 1 as i32;
}

fn G() {
  var unused x: i32 = F();
}
//@dump-sem-ir-end

// --- var_auto_returned_var.carbon
library "[[@TEST_NAME]]";

fn F() -> var auto {
  returned var v: i32 = 1;
  return var;
}

// --- fail_ref_auto_non_ref.carbon
library "[[@TEST_NAME]]";

fn F() -> ref auto {
  // CHECK:STDERR: fail_ref_auto_non_ref.carbon:[[@LINE+7]]:3: error: cannot bind durable reference to non-reference value of type `i32` [ConversionFailureNonRefToRef]
  // CHECK:STDERR:   return 0 as i32;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_ref_auto_non_ref.carbon:[[@LINE+4]]:10: note: return type deduced as `i32` [ReturnTypeDeducedAs]
  // CHECK:STDERR:   return 0 as i32;
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR:
  return 0 as i32;
}

// --- fail_ref_auto_returned_var.carbon
library "[[@TEST_NAME]]";

fn F() -> ref auto {
  // CHECK:STDERR: fail_ref_auto_returned_var.carbon:[[@LINE+7]]:3: error: `returned var` declaration in function with non-initializing return form [ReturnedVarNotInit]
  // CHECK:STDERR:   returned var v: i32 = 1;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_ref_auto_returned_var.carbon:[[@LINE-4]]:1: note: return form declared here [ReturnFormHereNote]
  // CHECK:STDERR: fn F() -> ref auto {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  returned var v: i32 = 1;
  return var;
}

// --- fail_val_auto_returned_var.carbon
library "[[@TEST_NAME]]";

fn F() -> val auto {
  // CHECK:STDERR: fail_val_auto_returned_var.carbon:[[@LINE+7]]:3: error: `returned var` declaration in function with non-initializing return form [ReturnedVarNotInit]
  // CHECK:STDERR:   returned var v: i32 = 1;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_val_auto_returned_var.carbon:[[@LINE-4]]:1: note: return form declared here [ReturnFormHereNote]
  // CHECK:STDERR: fn F() -> val auto {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  returned var v: i32 = 1;
  return var;
}

// --- fail_ref_auto_forward_decl.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_ref_auto_forward_decl.carbon:[[@LINE+4]]:8: error: forward declaration of function with deduced return type [DeducedReturnTypeNotDefinition]
// CHECK:STDERR: fn F() -> ref auto;
// CHECK:STDERR:        ^~~~~~~~~~~
// CHECK:STDERR:
fn F() -> ref auto;

// --- fail_ref_auto_missing_return_value.carbon
library "[[@TEST_NAME]]";

fn F() -> ref auto {
  // CHECK:STDERR: fail_ref_auto_missing_return_value.carbon:[[@LINE+7]]:3: error: missing return value [ReturnStatementMissingExpr]
  // CHECK:STDERR:   return;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_ref_auto_missing_return_value.carbon:[[@LINE-4]]:1: note: return type of function is deduced [ReturnTypeIsDeducedNote]
  // CHECK:STDERR: fn F() -> ref auto {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return;
}

// --- fail_todo_form_literal.carbon
library "[[@TEST_NAME]]";

var g: i32 = 0;

// CHECK:STDERR: fail_todo_form_literal.carbon:[[@LINE+4]]:21: error: semantics TODO: `HandleAutoTypeLiteral` [SemanticsTodo]
// CHECK:STDERR: fn F() ->? form(ref auto) {
// CHECK:STDERR:                     ^~~~
// CHECK:STDERR:
fn F() ->? form(ref auto) {
  return g;
}

// CHECK:STDOUT: --- ref_auto.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.010: Core.Form = ref_form auto [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.30e: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %.b71: %pattern_type.d94 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.2c5: %pattern_type.d94 = return_slot_pattern %.b71, %i32 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc7_10.1: %pattern_type.d94 = ref_return_pattern [concrete = constants.%.b71]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %.loc7_10.1, %i32 [concrete = constants.%return.patt.2c5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: Core.Form = ref_form auto [concrete = constants.%.010]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_10.2: Core.Form = ref_form %i32 [concrete = constants.%.30e]
// CHECK:STDOUT:     %.loc7_10.3: ref %i32 = ref_return
// CHECK:STDOUT:     %return: ref %i32 = return_slot %.loc7_10.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %.loc10_11.1: %pattern_type.d94 = ref_return_pattern [concrete = constants.%.b71]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %.loc10_11.1, %.loc10_11.3 [concrete = constants.%return.patt.2c5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_11.2: Core.Form = ref_form %i32 [concrete = constants.%.30e]
// CHECK:STDOUT:     %.loc10_11.3: type = type_component_of %.loc10_11.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10_11.4: ref %i32 = ref_return
// CHECK:STDOUT:     %return: ref %i32 = return_slot %.loc10_11.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> ref %i32 [deduced] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %g.ref: ref %i32 = name_ref g, file.%g [concrete = file.%g.var]
// CHECK:STDOUT:   %.loc7_11: ref %i32 = splice_block %g.ref [concrete = file.%g.var] {}
// CHECK:STDOUT:   return %.loc7_11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> ref %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: ref %i32 = call %F.ref()
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_auto.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.0c3: Core.Form = value_form auto [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.b93: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.396: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.84a: %Core.IntLiteral.as.As.impl.Convert.type.396 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e09: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b0: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.94f: %Core.IntLiteral.as.As.impl.Convert.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b93 = facet_value Core.IntLiteral, (%As.impl_witness.e09) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2be: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %As.WithSelf.Convert.type.2be, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.94f [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.94f, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.acd: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %.45b: %pattern_type.d94 = value_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.f8b: %pattern_type.d94 = return_slot_pattern %.45b, %i32 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = value_binding_pattern x [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.9e0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.396) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.84a)]
// CHECK:STDOUT:   %As.impl_witness_table.1c2 = impl_witness_table (%Core.import_ref.9e0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc5_12.1: %pattern_type.d94 = value_return_pattern [concrete = constants.%.45b]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %.loc5_12.1, %i32.loc5_12 [concrete = constants.%return.patt.f8b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: Core.Form = value_form auto [concrete = constants.%.0c3]
// CHECK:STDOUT:     %i32.loc5_12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_12.4: Core.Form = value_form %i32.loc5_12 [concrete = constants.%.acd]
// CHECK:STDOUT:     %.loc5_12.5: %i32 = value_return
// CHECK:STDOUT:     %return: %i32 = return_slot %.loc5_12.5
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> val %i32 [deduced] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc5_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.979 = impl_witness_access constants.%As.impl_witness.e09, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.94f]
// CHECK:STDOUT:   %bound_method.loc5_12.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_12.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc5_12.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc5_12.2: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc5_12.3: %i32 = converted %int_1, %.loc5_12.2 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc5_18: %i32 = splice_block %.loc5_12.3 [concrete = constants.%int_1.68a] {}
// CHECK:STDOUT:   return %.loc5_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: %i32 = call %F.ref()
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: %i32 = wrapper_binding x, %F.call
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_auto.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.79e: Core.Form = init_form auto [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.b93: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.396: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.84a: %Core.IntLiteral.as.As.impl.Convert.type.396 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e09: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b0: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.94f: %Core.IntLiteral.as.As.impl.Convert.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b93 = facet_value Core.IntLiteral, (%As.impl_witness.e09) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2be: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %As.WithSelf.Convert.type.2be, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.94f [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.94f, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.5c4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.68a, %Int.as.Copy.impl.Op.cae [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.a13: <bound method> = bound_method %int_1.68a, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %self.param_patt.0c5: %pattern_type.d94 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.749: %pattern_type.d94 = wrapper_binding_pattern self, %self.param_patt.0c5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.9e0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.396) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.84a)]
// CHECK:STDOUT:   %As.impl_witness_table.1c2 = impl_witness_table (%Core.import_ref.9e0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d94 = ref_param_pattern [concrete = constants.%self.param_patt.0c5]
// CHECK:STDOUT:     %self.patt: %pattern_type.d94 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.749]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc9 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d94 = ref_param_pattern [concrete = constants.%self.param_patt.0c5]
// CHECK:STDOUT:     %self.patt: %pattern_type.d94 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.749]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc5_12 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: Core.Form = init_form auto [concrete = constants.%.79e]
// CHECK:STDOUT:     %i32.loc5_12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_12.3: Core.Form = init_form %i32.loc5_12 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %i32 [deduced] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc5_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc5_12.1: %.979 = impl_witness_access constants.%As.impl_witness.e09, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.94f]
// CHECK:STDOUT:   %bound_method.loc5_12.1: <bound method> = bound_method %int_1, %impl.elem0.loc5_12.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc5_12.1: <specific function> = specific_function %impl.elem0.loc5_12.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_12.2: <bound method> = bound_method %int_1, %specific_fn.loc5_12.1 [concrete = constants.%bound_method.5c4]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc5_12.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc5_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc5_12.2: %i32 = converted %int_1, %.loc5_12.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc5_18: init %i32 = splice_block %Int.as.Copy.impl.Op.call [concrete = constants.%int_1.68a] {
// CHECK:STDOUT:     %impl.elem0.loc5_12.2: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:     %bound_method.loc5_12.3: <bound method> = bound_method %.loc5_12.2, %impl.elem0.loc5_12.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:     %specific_fn.loc5_12.2: <specific function> = specific_function %impl.elem0.loc5_12.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc5_12.4: <bound method> = bound_method %.loc5_12.2, %specific_fn.loc5_12.2 [concrete = constants.%bound_method.a13]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc5_12.4(%.loc5_12.2) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return %.loc5_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %i32 = var_storage %x.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   assign %x.var, %F.call
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: ref %i32 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %x.var, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%x.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
