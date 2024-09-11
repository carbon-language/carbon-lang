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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_9.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc11_9.2: type = converted %bool.make_type, %.loc11_9.1 [template = bool]
// CHECK:STDOUT:     %b.loc11_6.1: bool = param b
// CHECK:STDOUT:     @F.%b: bool = bind_name b, %b.loc11_6.1
// CHECK:STDOUT:     %int.make_type_32.loc11_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_18.1: type = value_of_initializer %int.make_type_32.loc11_18 [template = i32]
// CHECK:STDOUT:     %.loc11_18.2: type = converted %int.make_type_32.loc11_18, %.loc11_18.1 [template = i32]
// CHECK:STDOUT:     %n.loc11_15.1: i32 = param n
// CHECK:STDOUT:     @F.%n: i32 = bind_name n, %n.loc11_15.1
// CHECK:STDOUT:     %int.make_type_32.loc11_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_26.1: type = value_of_initializer %int.make_type_32.loc11_26 [template = i32]
// CHECK:STDOUT:     %.loc11_26.2: type = converted %int.make_type_32.loc11_26, %.loc11_26.1 [template = i32]
// CHECK:STDOUT:     %m.loc11_23.1: i32 = param m
// CHECK:STDOUT:     @F.%m: i32 = bind_name m, %m.loc11_23.1
// CHECK:STDOUT:     %int.make_type_32.loc11_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_34.1: type = value_of_initializer %int.make_type_32.loc11_34 [template = i32]
// CHECK:STDOUT:     %.loc11_34.2: type = converted %int.make_type_32.loc11_34, %.loc11_34.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool, %n: i32, %m: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_16: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_11.2: type = converted %int.make_type_32, %.loc12_11.1 [template = i32]
// CHECK:STDOUT:   %.loc12_17: type = array_type %.loc12_16, i32 [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref %.3 = var x
// CHECK:STDOUT:   %x: ref %.3 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc12_22: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_24.1: %.6 = tuple_literal (%.loc12_22)
// CHECK:STDOUT:   %.loc12_24.2: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_24.3: ref i32 = array_index %x.var, %.loc12_24.2
// CHECK:STDOUT:   %.loc12_24.4: init i32 = initialize_from %.loc12_22 to %.loc12_24.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_24.5: init %.3 = array_init (%.loc12_24.4) to %x.var [template = constants.%array]
// CHECK:STDOUT:   %.loc12_25: init %.3 = converted %.loc12_24.1, %.loc12_24.5 [template = constants.%array]
// CHECK:STDOUT:   assign %x.var, %.loc12_25
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %x.ref.loc13_20: ref %.3 = name_ref x, %x
// CHECK:STDOUT:   %m.ref: i32 = name_ref m, %m
// CHECK:STDOUT:   %.loc13_23.1: ref i32 = array_index %x.ref.loc13_20, %m.ref
// CHECK:STDOUT:   %.loc13_23.2: i32 = bind_value %.loc13_23.1
// CHECK:STDOUT:   br !if.expr.result(%.loc13_23.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %x.ref.loc13_30: ref %.3 = name_ref x, %x
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc13_33.1: ref i32 = array_index %x.ref.loc13_30, %n.ref
// CHECK:STDOUT:   %.loc13_33.2: i32 = bind_value %.loc13_33.1
// CHECK:STDOUT:   br !if.expr.result(%.loc13_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc13_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc13_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
