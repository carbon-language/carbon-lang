// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

abstract class Abstract {
  var a: i32;
}

class Derived {
  // TODO: `extend base: Base;`
  base: Abstract;

  var d: i32;
}

fn Make() -> Derived {
  // TODO: This should be valid, and should construct an instance of `partial Abstract` as the base.
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+3]]:26: ERROR: Cannot construct instance of abstract class. Consider using `partial Abstract` instead.
  // CHECK:STDERR:   return {.base = {.a = 1}, .d = 7};
  // CHECK:STDERR:                          ^
  return {.base = {.a = 1}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.a);
}

// CHECK:STDOUT: --- fail_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.a: i32}
// CHECK:STDOUT:   %.loc16_1.1: type = struct_type {.base: Abstract, .d: i32}
// CHECK:STDOUT:   %.loc16_1.2: type = struct_type {.base: {.a: i32}*, .d: i32}
// CHECK:STDOUT:   %.loc16_1.3: type = ptr_type {.base: {.a: i32}*, .d: i32}
// CHECK:STDOUT:   %.loc11: type = ptr_type {.base: Abstract, .d: i32}
// CHECK:STDOUT:   %.loc23: type = struct_type {.base: {.a: i32}, .d: i32}
// CHECK:STDOUT:   %.loc26_35.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc26_35.2: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc26_35.3: type = ptr_type (i32, i32)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Abstract = %Abstract.decl, .Derived = %Derived.decl, .Make = %Make, .Access = %Access}
// CHECK:STDOUT:   %Abstract.decl = class_decl @Abstract, ()
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract
// CHECK:STDOUT:   %Derived.decl = class_decl @Derived, ()
// CHECK:STDOUT:   %Derived: type = class_type @Derived
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make
// CHECK:STDOUT:   %Access: <function> = fn_decl @Access
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_element_type Abstract, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound element of class Abstract> = field_decl a, element0
// CHECK:STDOUT:   %a: <unbound element of class Abstract> = bind_name a, %.loc8_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Abstract.ref: type = name_ref Abstract, file.%Abstract
// CHECK:STDOUT:   %.loc13_17.1: type = unbound_element_type Derived, Abstract
// CHECK:STDOUT:   %.loc13_17.2: <unbound element of class Derived> = base_decl Abstract, element0
// CHECK:STDOUT:   %.loc15_8.1: type = unbound_element_type Derived, i32
// CHECK:STDOUT:   %.loc15_8.2: <unbound element of class Derived> = field_decl d, element1
// CHECK:STDOUT:   %d: <unbound element of class Derived> = bind_name d, %.loc15_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc13_17.2
// CHECK:STDOUT:   .d = %d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_25: i32 = int_literal 1
// CHECK:STDOUT:   %.loc23_26: {.a: i32} = struct_literal (%.loc23_25)
// CHECK:STDOUT:   %.loc23_34: i32 = int_literal 7
// CHECK:STDOUT:   %.loc23_35: {.base: {.a: i32}, .d: i32} = struct_literal (%.loc23_26, %.loc23_34)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d: Derived) -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc27_11: Derived = name_ref d, %d
// CHECK:STDOUT:   %.loc27_12.1: ref i32 = class_element_access %d.ref.loc27_11, element1
// CHECK:STDOUT:   %.loc27_12.2: i32 = bind_value %.loc27_12.1
// CHECK:STDOUT:   %d.ref.loc27_16: Derived = name_ref d, %d
// CHECK:STDOUT:   %.loc27_17.1: ref Abstract = class_element_access %d.ref.loc27_16, element0
// CHECK:STDOUT:   %.loc27_17.2: Abstract = bind_value %.loc27_17.1
// CHECK:STDOUT:   %.loc27_22.1: ref i32 = class_element_access %.loc27_17.2, element0
// CHECK:STDOUT:   %.loc27_22.2: i32 = bind_value %.loc27_22.1
// CHECK:STDOUT:   %.loc27_24.1: (i32, i32) = tuple_literal (%.loc27_12.2, %.loc27_22.2)
// CHECK:STDOUT:   %.loc27_24.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc27_24.3: init i32 = initialize_from %.loc27_12.2 to %.loc27_24.2
// CHECK:STDOUT:   %.loc27_24.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc27_24.5: init i32 = initialize_from %.loc27_22.2 to %.loc27_24.4
// CHECK:STDOUT:   %.loc27_24.6: init (i32, i32) = tuple_init (%.loc27_24.3, %.loc27_24.5) to %return
// CHECK:STDOUT:   %.loc27_24.7: init (i32, i32) = converted %.loc27_24.1, %.loc27_24.6
// CHECK:STDOUT:   return %.loc27_24.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
