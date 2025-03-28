// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/overloaded.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/overloaded.carbon

class X {
  var n: i32;
}

impl i32 as Core.As(X) {
  fn Convert[self: i32]() -> X { return {.n = self}; }
}

impl X as Core.As(i32) {
  fn Convert[self: X]() -> i32 { return self.n; }
}

let n: i32 = ((4 as i32) as X) as i32;

// CHECK:STDOUT: --- overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %X.elem: type = unbound_element_type %X, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.602: type = facet_type <@As, @As(%X)> [concrete]
// CHECK:STDOUT:   %Convert.type.35b: type = fn_type @Convert.1, @As(%X) [concrete]
// CHECK:STDOUT:   %impl_witness.491: <witness> = impl_witness (@impl.d8c.%Convert.decl) [concrete]
// CHECK:STDOUT:   %Convert.type.0e3: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.311: %Convert.type.0e3 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.206: %As.type.602 = facet_value %i32, (%impl_witness.491) [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.1, @As(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.662: <witness> = impl_witness (@impl.18a.%Convert.decl) [concrete]
// CHECK:STDOUT:   %Convert.type.c23: type = fn_type @Convert.3 [concrete]
// CHECK:STDOUT:   %Convert.8bb: %Convert.type.c23 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.79a: %As.type.fd4 = facet_value %X, (%impl_witness.662) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %impl_witness.882: <witness> = impl_witness (imports.%Core.import_ref.78a), @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.7, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.d7e: %As.type.fd4 = facet_value Core.IntLiteral, (%impl_witness.882) [concrete]
// CHECK:STDOUT:   %.249: type = fn_type_with_self_type %Convert.type.99b, %As.facet.d7e [concrete]
// CHECK:STDOUT:   %Convert.bound.e80: <bound method> = bound_method %int_4.0c1, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.197, @Convert.7(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %.e84: type = fn_type_with_self_type %Convert.type.35b, %As.facet.206 [concrete]
// CHECK:STDOUT:   %Convert.bound.483: <bound method> = bound_method %int_4.940, %Convert.311 [concrete]
// CHECK:STDOUT:   %.c95: type = fn_type_with_self_type %Convert.type.99b, %As.facet.79a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   impl_decl @impl.d8c [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %As.ref: %As.type.90f = name_ref As, imports.%Core.As [concrete = constants.%As.generic]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%X)> [concrete = constants.%As.type.602]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc15: <witness> = impl_witness (@impl.d8c.%Convert.decl) [concrete = constants.%impl_witness.491]
// CHECK:STDOUT:   impl_decl @impl.18a [concrete] {} {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %As.ref: %As.type.90f = name_ref As, imports.%Core.As [concrete = constants.%As.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32)> [concrete = constants.%As.type.fd4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc19: <witness> = impl_witness (@impl.18a.%Convert.decl) [concrete = constants.%impl_witness.662]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: %i32 = bind_name n, @__global_init.%.loc23_32.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.d8c: %i32 as %As.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.0e3 = fn_decl @Convert.2 [concrete = constants.%Convert.311] {
// CHECK:STDOUT:     %self.patt: %i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %self.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc16_20: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %i32 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %X = out_param call_param1
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.18a: %X.ref as %As.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.c23 = fn_decl @Convert.3 [concrete = constants.%Convert.8bb] {
// CHECK:STDOUT:     %self.patt: %X = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %X = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %self: %X = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .X = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %.loc12_8: %X.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc12_3: %X.elem = var_pattern %.loc12_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %X.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .n = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: %i32]() -> %return.param_patt: %X {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %i32 = name_ref self, %self
// CHECK:STDOUT:   %.loc16_51.1: %struct_type.n = struct_literal (%self.ref)
// CHECK:STDOUT:   %.loc16_51.2: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc16_51.3: init %i32 = initialize_from %self.ref to %.loc16_51.2
// CHECK:STDOUT:   %.loc16_51.4: init %X = class_init (%.loc16_51.3), %return
// CHECK:STDOUT:   %.loc16_52: init %X = converted %.loc16_51.1, %.loc16_51.4
// CHECK:STDOUT:   return %.loc16_52 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: %X]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %X = name_ref self, %self
// CHECK:STDOUT:   %n.ref: %X.elem = name_ref n, @X.%.loc12_8 [concrete = @X.%.loc12_8]
// CHECK:STDOUT:   %.loc20_45.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc20_45.2: %i32 = bind_value %.loc20_45.1
// CHECK:STDOUT:   return %.loc20_45.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_32.loc23_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc23_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc23_18: %.249 = impl_witness_access constants.%impl_witness.882, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc23_18.1: <bound method> = bound_method %int_4, %impl.elem0.loc23_18 [concrete = constants.%Convert.bound.e80]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc23_18, @Convert.7(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_18.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc23_18.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc23_18.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc23_18.2: %i32 = converted %int_4, %.loc23_18.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %impl.elem0.loc23_26: %.e84 = impl_witness_access constants.%impl_witness.491, element0 [concrete = constants.%Convert.311]
// CHECK:STDOUT:   %bound_method.loc23_26: <bound method> = bound_method %.loc23_18.2, %impl.elem0.loc23_26 [concrete = constants.%Convert.bound.483]
// CHECK:STDOUT:   %.loc23_26.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Convert.call.loc23_26: init %X = call %bound_method.loc23_26(%.loc23_18.2) to %.loc23_26.1
// CHECK:STDOUT:   %.loc23_26.2: init %X = converted %.loc23_18.2, %Convert.call.loc23_26
// CHECK:STDOUT:   %int_32.loc23_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc23_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc23_32: %.c95 = impl_witness_access constants.%impl_witness.662, element0 [concrete = constants.%Convert.8bb]
// CHECK:STDOUT:   %bound_method.loc23_32: <bound method> = bound_method %.loc23_26.2, %impl.elem0.loc23_32
// CHECK:STDOUT:   %.loc23_26.3: ref %X = temporary %.loc23_26.1, %.loc23_26.2
// CHECK:STDOUT:   %.loc23_26.4: %X = bind_value %.loc23_26.3
// CHECK:STDOUT:   %Convert.call.loc23_32: init %i32 = call %bound_method.loc23_32(%.loc23_26.4)
// CHECK:STDOUT:   %.loc23_32.1: %i32 = value_of_initializer %Convert.call.loc23_32
// CHECK:STDOUT:   %.loc23_32.2: %i32 = converted %.loc23_26.2, %.loc23_32.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
