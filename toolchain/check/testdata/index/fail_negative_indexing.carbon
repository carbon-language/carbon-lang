// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (12, 6);
// TODO: Only diagnose this once, and make the diagnostic clearer.
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+3]]:16: ERROR: Cannot access member of interface Negate in type i32 that does not implement that interface.
// CHECK:STDERR: var b: i32 = a[-10];
// CHECK:STDERR:                ^~~
var b: i32 = a[-10];

// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %tuple: (i32, i32) = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 10 [template]
// CHECK:STDOUT:   %.7: type = interface_type @Negate [template]
// CHECK:STDOUT:   %Self: Negate = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Op: type = fn_type @Op [template]
// CHECK:STDOUT:   %.8: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Op = struct_value () [template]
// CHECK:STDOUT:   %.9: type = assoc_entity_type @Negate, Op [template]
// CHECK:STDOUT:   %.10: <associated Op in Negate> = assoc_entity element0, file.%import_ref.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc7_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_17.2: type = converted %.loc7_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref (i32, i32) = var a
// CHECK:STDOUT:   %a: ref (i32, i32) = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir4, inst+67, loaded [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir4, inst+69, unloaded
// CHECK:STDOUT:   %import_ref.3: <associated Op in Negate> = import_ref ir4, inst+84, loaded [template = constants.%.10]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir4, inst+80, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref ir4, inst+80, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Negate {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   .Op = file.%import_ref.3
// CHECK:STDOUT:   witness = (file.%import_ref.4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op[%self: Self]() -> Self;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_27.1: (i32, i32) = tuple_literal (%.loc7_22, %.loc7_26)
// CHECK:STDOUT:   %.loc7_27.2: ref i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc7_27.3: init i32 = initialize_from %.loc7_22 to %.loc7_27.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_27.4: ref i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc7_27.5: init i32 = initialize_from %.loc7_26 to %.loc7_27.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_27.6: init (i32, i32) = tuple_init (%.loc7_27.3, %.loc7_27.5) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_28: init (i32, i32) = converted %.loc7_27.1, %.loc7_27.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc7_28
// CHECK:STDOUT:   %a.ref: ref (i32, i32) = name_ref a, file.%a
// CHECK:STDOUT:   %.loc12_17: i32 = int_literal 10 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_19: ref <error> = tuple_index %a.ref, <error>
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
