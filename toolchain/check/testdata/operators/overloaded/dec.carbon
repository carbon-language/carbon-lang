// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/dec.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/dec.carbon

// This file was generated from unary_stmt.carbon.tmpl. Run make_tests.sh to regenerate.

package User;

class C {};

impl C as Core.Dec {
  fn Op[addr self: C*]();
}

fn TestOp() {
  var c: C = {};
  --c;
}

// CHECK:STDOUT: --- dec.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %Op.type.633: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.7f9.%Op.decl) [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.type.9e0: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.cf9: %Op.type.9e0 = struct_value () [concrete]
// CHECK:STDOUT:   %Dec.facet: %Dec.type = facet_value %C, (%impl_witness) [concrete]
// CHECK:STDOUT:   %TestOp.type: type = fn_type @TestOp [concrete]
// CHECK:STDOUT:   %TestOp: %TestOp.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.369: type = fn_type_with_self_type %Op.type.633, %Dec.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Dec = %Core.Dec
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Dec: type = import_ref Core//prelude/operators/arithmetic, Dec, loaded [concrete = constants.%Dec.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TestOp = %TestOp.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.7f9 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Dec.ref: type = name_ref Dec, imports.%Core.Dec [concrete = constants.%Dec.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.7f9.%Op.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %TestOp.decl: %TestOp.type = fn_decl @TestOp [concrete = constants.%TestOp] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.7f9: %C.ref as %Dec.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.9e0 = fn_decl @Op.2 [concrete = constants.%Op.cf9] {
// CHECK:STDOUT:     %self.patt: %ptr.019 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.019 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc18_9: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc18_21: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[addr %self.param_patt: %ptr.019]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestOp() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc22_3.1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc22_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc22_15.2: init %C = class_init (), %c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc22_3.2: init %C = converted %.loc22_15.1, %.loc22_15.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc22_3.2
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   %impl.elem0: %.369 = impl_witness_access constants.%impl_witness, element0 [concrete = constants.%Op.cf9]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %c.ref
// CHECK:STDOUT:   %Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
