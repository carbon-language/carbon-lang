// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/basic.carbon

fn F(b: bool, n: i32, m: i32) -> i32 {
  var x: [i32; 1] = (0,);
  return if b then x[m] else x[n];
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %i32 [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_0.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param1
// CHECK:STDOUT:     %m.patt: %i32 = binding_pattern m
// CHECK:STDOUT:     %m.param_patt: %i32 = value_param_pattern %m.patt, runtime_param2
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_9.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc11_9.2: type = converted %bool.make_type, %.loc11_9.1 [template = bool]
// CHECK:STDOUT:     %int_32.loc11_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_18: init type = call constants.%Int(%int_32.loc11_18) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_18.1: type = value_of_initializer %int.make_type_signed.loc11_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_18.2: type = converted %int.make_type_signed.loc11_18, %.loc11_18.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_26: init type = call constants.%Int(%int_32.loc11_26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_26.1: type = value_of_initializer %int.make_type_signed.loc11_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_26.2: type = converted %int.make_type_signed.loc11_26, %.loc11_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_34: init type = call constants.%Int(%int_32.loc11_34) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_34.1: type = value_of_initializer %int.make_type_signed.loc11_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_34.2: type = converted %int.make_type_signed.loc11_34, %.loc11_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %m.param: %i32 = value_param runtime_param2
// CHECK:STDOUT:     %m: %i32 = bind_name m, %m.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param3
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param_patt: bool, %n.param_patt: %i32, %m.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc12_11.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_11.2: type = converted %int.make_type_signed.loc12, %.loc12_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %x.var: ref %array_type = var x
// CHECK:STDOUT:   %x: ref %array_type = bind_name x, %x.var
// CHECK:STDOUT:   %int_0.loc12_22: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc12_24.1: %tuple.type = tuple_literal (%int_0.loc12_22)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.loc12_22, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0.loc12_22) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_24.2: init %i32 = converted %int_0.loc12_22, %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:   %int_0.loc12_24: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc12_24.3: ref %i32 = array_index %x.var, %int_0.loc12_24
// CHECK:STDOUT:   %.loc12_24.4: init %i32 = initialize_from %.loc12_24.2 to %.loc12_24.3 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_24.5: init %array_type = array_init (%.loc12_24.4) to %x.var [template = constants.%array]
// CHECK:STDOUT:   %.loc12_25: init %array_type = converted %.loc12_24.1, %.loc12_24.5 [template = constants.%array]
// CHECK:STDOUT:   assign %x.var, %.loc12_25
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %x.ref.loc13_20: ref %array_type = name_ref x, %x
// CHECK:STDOUT:   %m.ref: %i32 = name_ref m, %m
// CHECK:STDOUT:   %int_32.loc13_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13_23: init type = call constants.%Int(%int_32.loc13_23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_23.1: type = value_of_initializer %int.make_type_signed.loc13_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_23.2: type = converted %int.make_type_signed.loc13_23, %.loc13_23.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_23.3: ref %i32 = array_index %x.ref.loc13_20, %m.ref
// CHECK:STDOUT:   %.loc13_23.4: %i32 = bind_value %.loc13_23.3
// CHECK:STDOUT:   br !if.expr.result(%.loc13_23.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %x.ref.loc13_30: ref %array_type = name_ref x, %x
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %int_32.loc13_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13_33: init type = call constants.%Int(%int_32.loc13_33) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_33.1: type = value_of_initializer %int.make_type_signed.loc13_33 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_33.2: type = converted %int.make_type_signed.loc13_33, %.loc13_33.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_33.3: ref %i32 = array_index %x.ref.loc13_30, %n.ref
// CHECK:STDOUT:   %.loc13_33.4: %i32 = bind_value %.loc13_33.3
// CHECK:STDOUT:   br !if.expr.result(%.loc13_33.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc13_10: %i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc13_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
