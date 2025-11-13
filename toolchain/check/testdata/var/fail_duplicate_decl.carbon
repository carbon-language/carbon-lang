// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_duplicate_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_duplicate_decl.carbon

// --- fail_basic.carbon

fn Main() {
  var x: () = ();
  // CHECK:STDERR: fail_basic.carbon:[[@LINE+7]]:7: error: duplicate name `x` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_basic.carbon:[[@LINE-4]]:7: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  var x: () = ();
}

// CHECK:STDOUT: --- fail_basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %empty_tuple.type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.bb8: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.72f: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.bb8 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.72f, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc3: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt.loc3: %pattern_type.cb1 = var_pattern %x.patt.loc3 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc3: ref %empty_tuple.type = var %x.var_patt.loc3
// CHECK:STDOUT:   %.loc3_16.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc3_16.2: init %empty_tuple.type = tuple_init () to %x.var.loc3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc3_3: init %empty_tuple.type = converted %.loc3_16.1, %.loc3_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc3, %.loc3_3
// CHECK:STDOUT:   %.loc3_11.1: type = splice_block %.loc3_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc3_11.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc3_11.3: type = converted %.loc3_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc3: ref %empty_tuple.type = ref_binding x, %x.var.loc3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt.loc11: %pattern_type.cb1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt.loc11: %pattern_type.cb1 = var_pattern %x.patt.loc11 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var.loc11: ref %empty_tuple.type = var %x.var_patt.loc11
// CHECK:STDOUT:   %.loc11_16.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_16.2: init %empty_tuple.type = tuple_init () to %x.var.loc11 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_3: init %empty_tuple.type = converted %.loc11_16.1, %.loc11_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var.loc11, %.loc11_3
// CHECK:STDOUT:   %.loc11_11.1: type = splice_block %.loc11_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc11_11.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11_11.3: type = converted %.loc11_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.loc11: ref %empty_tuple.type = ref_binding x, %x.var.loc11
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc11: <bound method> = bound_method %x.var.loc11, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %x.var.loc11, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc11: %ptr.843 = addr_of %x.var.loc11
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc11: init %empty_tuple.type = call %bound_method.loc11(%addr.loc11)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc3: <bound method> = bound_method %x.var.loc3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.72f, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc3: <bound method> = bound_method %x.var.loc3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc3: %ptr.843 = addr_of %x.var.loc3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc3: init %empty_tuple.type = call %bound_method.loc3(%addr.loc3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
