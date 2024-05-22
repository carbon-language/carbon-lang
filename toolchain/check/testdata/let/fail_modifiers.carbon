// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `protected` not allowed on `let` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected let b: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected let b: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `default` not allowed on `let` declaration outside of an interface.
// CHECK:STDERR: default let c: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default let c: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `final` not allowed on `let` declaration outside of an interface.
// CHECK:STDERR: final let d: i32 = 1;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
final let d: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `virtual` not allowed on `let` declaration.
// CHECK:STDERR: virtual let e: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual let e: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: `default` not allowed on `let` declaration outside of an interface.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: ERROR: `final` not allowed on declaration with `default`.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `default` previously appeared here.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default final let f: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: `default` not allowed on `let` declaration outside of an interface.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: ERROR: `default` repeated on declaration.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `default` previously appeared here.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default default let g: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: `protected` not allowed on `let` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:11: ERROR: `private` not allowed on declaration with `protected`.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `protected` previously appeared here.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected private let h: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+10]]:1: ERROR: `protected` not allowed on `let` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: ERROR: `protected` repeated on declaration.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `protected` previously appeared here.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
protected protected let i: i32 = 1;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc11_18.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_18.2: type = converted %int.make_type_32.loc11, %.loc11_18.1 [template = i32]
// CHECK:STDOUT:   %.loc11_24: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc11_24
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc17_16.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_16.2: type = converted %int.make_type_32.loc17, %.loc17_16.1 [template = i32]
// CHECK:STDOUT:   %.loc17_22: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc17_22
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc23_14.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_14.2: type = converted %int.make_type_32.loc23, %.loc23_14.1 [template = i32]
// CHECK:STDOUT:   %.loc23_20: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %d: i32 = bind_name d, %.loc23_20
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc29: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc29_16.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:   %.loc29_16.2: type = converted %int.make_type_32.loc29, %.loc29_16.1 [template = i32]
// CHECK:STDOUT:   %.loc29_22: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %e: i32 = bind_name e, %.loc29_22
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc42: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc42_22.1: type = value_of_initializer %int.make_type_32.loc42 [template = i32]
// CHECK:STDOUT:   %.loc42_22.2: type = converted %int.make_type_32.loc42, %.loc42_22.1 [template = i32]
// CHECK:STDOUT:   %.loc42_28: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %f: i32 = bind_name f, %.loc42_28
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc55: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc55_24.1: type = value_of_initializer %int.make_type_32.loc55 [template = i32]
// CHECK:STDOUT:   %.loc55_24.2: type = converted %int.make_type_32.loc55, %.loc55_24.1 [template = i32]
// CHECK:STDOUT:   %.loc55_30: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %g: i32 = bind_name g, %.loc55_30
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc68: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc68_26.1: type = value_of_initializer %int.make_type_32.loc68 [template = i32]
// CHECK:STDOUT:   %.loc68_26.2: type = converted %int.make_type_32.loc68, %.loc68_26.1 [template = i32]
// CHECK:STDOUT:   %.loc68_32: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %h: i32 = bind_name h, %.loc68_32
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc80: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc80_28.1: type = value_of_initializer %int.make_type_32.loc80 [template = i32]
// CHECK:STDOUT:   %.loc80_28.2: type = converted %int.make_type_32.loc80, %.loc80_28.1 [template = i32]
// CHECK:STDOUT:   %.loc80_34: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %i: i32 = bind_name i, %.loc80_34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
