// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {
  var b: i32;
}

class Derived {
  // TODO: `extend base: Base;`
  base: Base;

  var d: i32;
}

fn Make() -> Derived {
  return {.base = {.b = 4}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.b);
}

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9: type = struct_type {.b: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.b: i32}
// CHECK:STDOUT:   %.loc16_1.1: type = struct_type {.base: Base, .d: i32}
// CHECK:STDOUT:   %.loc16_1.2: type = struct_type {.base: {.b: i32}*, .d: i32}
// CHECK:STDOUT:   %.loc16_1.3: type = ptr_type {.base: {.b: i32}*, .d: i32}
// CHECK:STDOUT:   %.loc11: type = ptr_type {.base: Base, .d: i32}
// CHECK:STDOUT:   %.loc19: type = struct_type {.base: {.b: i32}, .d: i32}
// CHECK:STDOUT:   %.loc22_35.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc22_35.2: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc22_35.3: type = ptr_type (i32, i32)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Base = %Base.decl, .Derived = %Derived.decl, .Make = %Make, .Access = %Access}
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Base: type = class_type @Base
// CHECK:STDOUT:   %Derived.decl = class_decl @Derived, ()
// CHECK:STDOUT:   %Derived: type = class_type @Derived
// CHECK:STDOUT:   %Make: <function> = fn_decl @Make
// CHECK:STDOUT:   %Access: <function> = fn_decl @Access
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc8_8.1: type = unbound_element_type Base, i32
// CHECK:STDOUT:   %.loc8_8.2: <unbound field of class Base> = field b, element0
// CHECK:STDOUT:   %b: <unbound field of class Base> = bind_name b, %.loc8_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base
// CHECK:STDOUT:   %.loc13_13.1: type = unbound_element_type Derived, Base
// CHECK:STDOUT:   %.loc13_13.2: <unbound field of class Derived> = base Base, element0
// CHECK:STDOUT:   %.loc15_8.1: type = unbound_element_type Derived, i32
// CHECK:STDOUT:   %.loc15_8.2: <unbound field of class Derived> = field d, element1
// CHECK:STDOUT:   %d: <unbound field of class Derived> = bind_name d, %.loc15_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc13_13.2
// CHECK:STDOUT:   .d = %d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_25: i32 = int_literal 4
// CHECK:STDOUT:   %.loc19_26.1: {.b: i32} = struct_literal (%.loc19_25)
// CHECK:STDOUT:   %.loc19_34: i32 = int_literal 7
// CHECK:STDOUT:   %.loc19_35.1: {.base: {.b: i32}, .d: i32} = struct_literal (%.loc19_26.1, %.loc19_34)
// CHECK:STDOUT:   %.loc19_35.2: ref Base = class_element_access %return, element0
// CHECK:STDOUT:   %.loc19_26.2: ref i32 = class_element_access %.loc19_35.2, element0
// CHECK:STDOUT:   %.loc19_26.3: init i32 = initialize_from %.loc19_25 to %.loc19_26.2
// CHECK:STDOUT:   %.loc19_26.4: init Base = class_init (%.loc19_26.3), %.loc19_35.2
// CHECK:STDOUT:   %.loc19_26.5: init Base = converted %.loc19_26.1, %.loc19_26.4
// CHECK:STDOUT:   %.loc19_35.3: ref i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc19_35.4: init i32 = initialize_from %.loc19_34 to %.loc19_35.3
// CHECK:STDOUT:   %.loc19_35.5: init Derived = class_init (%.loc19_26.5, %.loc19_35.4), %return
// CHECK:STDOUT:   %.loc19_35.6: init Derived = converted %.loc19_35.1, %.loc19_35.5
// CHECK:STDOUT:   return %.loc19_35.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d: Derived) -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc23_11: Derived = name_ref d, %d
// CHECK:STDOUT:   %.loc23_12.1: ref i32 = class_element_access %d.ref.loc23_11, element1
// CHECK:STDOUT:   %.loc23_12.2: i32 = bind_value %.loc23_12.1
// CHECK:STDOUT:   %d.ref.loc23_16: Derived = name_ref d, %d
// CHECK:STDOUT:   %.loc23_17.1: ref Base = class_element_access %d.ref.loc23_16, element0
// CHECK:STDOUT:   %.loc23_17.2: Base = bind_value %.loc23_17.1
// CHECK:STDOUT:   %.loc23_22.1: ref i32 = class_element_access %.loc23_17.2, element0
// CHECK:STDOUT:   %.loc23_22.2: i32 = bind_value %.loc23_22.1
// CHECK:STDOUT:   %.loc23_24.1: (i32, i32) = tuple_literal (%.loc23_12.2, %.loc23_22.2)
// CHECK:STDOUT:   %.loc23_24.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc23_24.3: init i32 = initialize_from %.loc23_12.2 to %.loc23_24.2
// CHECK:STDOUT:   %.loc23_24.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc23_24.5: init i32 = initialize_from %.loc23_22.2 to %.loc23_24.4
// CHECK:STDOUT:   %.loc23_24.6: init (i32, i32) = tuple_init (%.loc23_24.3, %.loc23_24.5) to %return
// CHECK:STDOUT:   %.loc23_24.7: init (i32, i32) = converted %.loc23_24.1, %.loc23_24.6
// CHECK:STDOUT:   return %.loc23_24.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
