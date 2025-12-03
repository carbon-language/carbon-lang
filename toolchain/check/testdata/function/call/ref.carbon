// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/ref.carbon

// --- basics.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(ref x: i32);

fn G() {
  var y: i32 = 0;

  F(ref y);
}
//@dump-sem-ir-end

// --- class.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class C {
  fn F[ref self: Self]();
}

fn G() {
  var c: C;

  c.F();
}
//@dump-sem-ir-end

// --- fail_ref_not_durable_ref.carbon

library "[[@TEST_NAME]]";

fn F(ref x: i32);

fn G() {
  let y: i32 = 0;
  // CHECK:STDERR: fail_ref_not_durable_ref.carbon:[[@LINE+4]]:5: error: expression tagged with `ref` is not a durable reference [RefTagNotDurableRef]
  // CHECK:STDERR:   F(ref y);
  // CHECK:STDERR:     ^~~~~
  // CHECK:STDERR:
  F(ref y);
}

// --- fail_missing_ref.carbon

library "[[@TEST_NAME]]";

fn F(ref x: i32);

fn G() {
  var y: i32 = 0;
  // CHECK:STDERR: fail_missing_ref.carbon:[[@LINE+7]]:5: error: argument to `ref` parameter not marked with `ref` [RefParamNoRefTag]
  // CHECK:STDERR:   F(y);
  // CHECK:STDERR:     ^
  // CHECK:STDERR: fail_missing_ref.carbon:[[@LINE-7]]:6: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F(ref x: i32);
  // CHECK:STDERR:      ^~~~~~~~~~
  // CHECK:STDERR:
  F(y);
}

// --- fail_unnecessary_ref.carbon

library "[[@TEST_NAME]]";

fn F(x: i32);

fn G() {
  var y: i32 = 0;

  // CHECK:STDERR: fail_unnecessary_ref.carbon:[[@LINE+4]]:24: error: `ref` tag is not an argument to a `ref` parameter [RefTagNoRefParam]
  // CHECK:STDERR:   var z: (i32,) = (ref y,);
  // CHECK:STDERR:                        ^
  // CHECK:STDERR:
  var z: (i32,) = (ref y,);
  // CHECK:STDERR: fail_unnecessary_ref.carbon:[[@LINE+7]]:9: error: `ref` tag is not an argument to a `ref` parameter [RefTagNoRefParam]
  // CHECK:STDERR:   F(ref y);
  // CHECK:STDERR:         ^
  // CHECK:STDERR: fail_unnecessary_ref.carbon:[[@LINE-13]]:6: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F(x: i32);
  // CHECK:STDERR:      ^~~~~~
  // CHECK:STDERR:
  F(ref y);
}

// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d4e: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.424: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d4e = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = ref_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %i32 = ref_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.7ce = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.7ce = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %i32 = var %y.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_3.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_3.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_3.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc8_3: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %y.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %i32 = ref_binding y, %y.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %y.ref: ref %i32 = name_ref y, %y
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%y.ref)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %y.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.424
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_3.3: <bound method> = bound_method %y.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_3.3(%y.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.eeb: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.88a: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.eeb = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: ref %C = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %C.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.F(%self.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %C.F.type = name_ref F, @C.%C.F.decl [concrete = constants.%C.F]
// CHECK:STDOUT:   %C.F.bound: <bound method> = bound_method %c.ref, %F.ref
// CHECK:STDOUT:   %C.F.call: init %empty_tuple.type = call %C.F.bound(%c.ref)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.88a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%c.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
