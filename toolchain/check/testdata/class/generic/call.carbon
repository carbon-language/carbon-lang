// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- call.carbon

library "call";

class Class(T:! type, N:! i32) {}

var a: Class(i32*, 5);

// Requires an implicit conversion to type `type`.
var b: Class((), 0);

// --- fail_too_few.carbon

library "too_few";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_too_few.carbon:[[@LINE+7]]:8: ERROR: 1 argument(s) passed to function expecting 2 argument(s).
// CHECK:STDERR: var a: Class(i32*);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_too_few.carbon:[[@LINE-5]]:1: Calling function declared here.
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Class(i32*);

// --- fail_too_many.carbon

library "too_many";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_too_many.carbon:[[@LINE+7]]:8: ERROR: 3 argument(s) passed to function expecting 2 argument(s).
// CHECK:STDERR: var a: Class(i32*, 1, 2);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_too_many.carbon:[[@LINE-5]]:1: Calling function declared here.
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Class(i32*, 1, 2);

// --- fail_no_conversion.carbon

library "no_conversion";

class Class(T:! type, N:! i32) {}

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+6]]:8: ERROR: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: var a: Class(5, i32*);
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE-5]]:1: Initializing parameter 1 of function declared here.
// CHECK:STDERR: class Class(T:! type, N:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var a: Class(5, i32*);

// CHECK:STDOUT: --- call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %Class.1: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %struct.2: Class = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, (file.%.loc6_17.3, file.%.loc6_20) [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %Class.4: type = class_type @Class, (file.%.loc9_13, file.%.loc9_18) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Class.decl: Class = class_decl @Class [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.loc4_23.1: i32 = param N
// CHECK:STDOUT:     %N.loc4_23.2: i32 = bind_symbolic_name N 1, %N.loc4_23.1 [symbolic = constants.%N]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref.loc6: Class = name_ref Class, %Class.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc6_17.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_17.2: type = converted %int.make_type_32.loc6, %.loc6_17.1 [template = i32]
// CHECK:STDOUT:   %.loc6_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_20: i32 = int_literal 5 [template = constants.%.4]
// CHECK:STDOUT:   %Class.loc6: type = class_type @Class, (%.loc6_17.3, %.loc6_20) [template = constants.%Class.3]
// CHECK:STDOUT:   %a.var: ref Class = var a
// CHECK:STDOUT:   %a: ref Class = bind_name a, %a.var
// CHECK:STDOUT:   %Class.ref.loc9: Class = name_ref Class, %Class.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc9_15: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_18: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_13: type = converted %.loc9_15, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %Class.loc9: type = class_type @Class, (%.loc9_13, %.loc9_18) [template = constants.%Class.4]
// CHECK:STDOUT:   %b.var: ref Class = var b
// CHECK:STDOUT:   %b: ref Class = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_few.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %Class.1: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %struct.2: Class = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Class.decl: Class = class_decl @Class [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.loc4_23.1: i32 = param N
// CHECK:STDOUT:     %N.loc4_23.2: i32 = bind_symbolic_name N 1, %N.loc4_23.1 [symbolic = constants.%N]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: Class = name_ref Class, %Class.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_32.loc13, %.loc13_17.1 [template = i32]
// CHECK:STDOUT:   %.loc13_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %Class: type = class_type @Class [template = constants.%Class.2]
// CHECK:STDOUT:   %a.var: ref Class = var a
// CHECK:STDOUT:   %a: ref Class = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_many.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %Class.1: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %struct.2: Class = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Class.decl: Class = class_decl @Class [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.loc4_23.1: i32 = param N
// CHECK:STDOUT:     %N.loc4_23.2: i32 = bind_symbolic_name N 1, %N.loc4_23.1 [symbolic = constants.%N]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: Class = name_ref Class, %Class.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc13_17.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_17.2: type = converted %int.make_type_32.loc13, %.loc13_17.1 [template = i32]
// CHECK:STDOUT:   %.loc13_17.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_20: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_23: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %Class: type = class_type @Class [template = constants.%Class.2]
// CHECK:STDOUT:   %a.var: ref Class = var a
// CHECK:STDOUT:   %a: ref Class = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %Class.1: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %struct.2: Class = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Class.decl: Class = class_decl @Class [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc4_13.1: type = param T
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T 0, %T.loc4_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %N.loc4_23.1: i32 = param N
// CHECK:STDOUT:     %N.loc4_23.2: i32 = bind_symbolic_name N 1, %N.loc4_23.1 [symbolic = constants.%N]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.ref: Class = name_ref Class, %Class.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc12_14: i32 = int_literal 5 [template = constants.%.3]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc12_20.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_20.2: type = converted %int.make_type_32.loc12, %.loc12_20.1 [template = i32]
// CHECK:STDOUT:   %.loc12_20.3: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %Class: type = class_type @Class [template = constants.%Class.2]
// CHECK:STDOUT:   %a.var: ref Class = var a
// CHECK:STDOUT:   %a: ref Class = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
