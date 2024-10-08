// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/adapter_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/adapter_conversion.carbon

// --- adapt_class.carbon

library "[[@TEST_NAME]]";

class A {
  var x: i32;
  var y: i32;

  fn Make() -> A {
    return {.x = 1, .y = 2};
  }
}

class B {
  adapt A;
}

var a_ref: A = {.x = 1, .y = 2};
let a_val: A = a_ref;

// An `as` conversion to an adapter type preserves the expression category.
let b_val: B = a_val as B;
let b_ptr: B* = &(a_ref as B);

var b_factory: B = A.Make() as B;

// --- adapt_i32.carbon

library "[[@TEST_NAME]]";

class A {
  adapt i32;
}

let a: A = (1 as i32) as A;
let n: i32 = a as i32;

// --- multi_level_adapt.carbon

library "[[@TEST_NAME]]";

class A { adapt {}; }
class B { adapt A; }
class C { adapt B; }
class D { adapt C; }

let d: D = {} as D;

// --- fail_init_class.carbon

library "[[@TEST_NAME]]";

class A {
  var x: i32;
  var y: i32;
}

class B {
  adapt A;
}

let b_value: B = ({.x = 1, .y = 2} as A) as B;

// TODO: Here, we treat `{.x = 1, .y = 2} as A` as a value expression, not an
// initializing expression, so `(...) as B` is a value expression too, requiring
// a copy to perform initialization. It's not clear whether that is the right
// behavior.

// CHECK:STDERR: fail_init_class.carbon:[[@LINE+4]]:17: error: cannot copy value of type `B`
// CHECK:STDERR: var b_init: B = ({.x = 1, .y = 2} as A) as B;
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var b_init: B = ({.x = 1, .y = 2} as A) as B;

// --- fail_adapt_init_from_struct.carbon

library "[[@TEST_NAME]]";

class A {
  var x: i32;
}

class B {
  adapt A;
}

// We do not try to implicitly convert from the first operand of `as` to the
// adapted type of the second operand.

// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+6]]:12: error: cannot convert from `{.x: i32}` to `B` with `as`
// CHECK:STDERR: var b: B = {.x = 1} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~
// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+3]]:12: note: type `{.x: i32}` does not implement interface `As`
// CHECK:STDERR: var b: B = {.x = 1} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~
var b: B = {.x = 1} as B;

// CHECK:STDOUT: --- adapt_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A, i32 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct: %A = struct_value (%.6, %.7) [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.8: <witness> = complete_type_witness %A [template]
// CHECK:STDOUT:   %.9: type = ptr_type %B [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .a_val = @__global_init.%a_val
// CHECK:STDOUT:     .b_val = @__global_init.%b_val
// CHECK:STDOUT:     .b_ptr = @__global_init.%b_ptr
// CHECK:STDOUT:     .b_factory = %b_factory
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %A.ref.loc17: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %a_ref.var: ref %A = var a_ref
// CHECK:STDOUT:   %a_ref: ref %A = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %A.ref.loc18: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %B.ref.loc21: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %B.ref.loc22: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc22: type = ptr_type %B [template = constants.%.9]
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %b_factory.var: ref %B = var b_factory
// CHECK:STDOUT:   %b_factory: ref %B = bind_name b_factory, %b_factory.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_32.loc5, %.loc5_10.1 [template = i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl x, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_10.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_10.2: type = converted %int.make_type_32.loc6, %.loc6_10.1 [template = i32]
// CHECK:STDOUT:   %.loc6_8: %.2 = field_decl y, element1 [template]
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %return: ref %A = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT:   .y = %.loc6_8
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %A [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_18: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_26: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_27.1: %.3 = struct_literal (%.loc9_18, %.loc9_26)
// CHECK:STDOUT:   %.loc9_27.2: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc9_27.3: init i32 = initialize_from %.loc9_18 to %.loc9_27.2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_27.4: ref i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc9_27.5: init i32 = initialize_from %.loc9_26 to %.loc9_27.4 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_27.6: init %A = class_init (%.loc9_27.3, %.loc9_27.5), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_28: init %A = converted %.loc9_27.1, %.loc9_27.6 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc9_28 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17_22: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_30: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc17_31.1: %.3 = struct_literal (%.loc17_22, %.loc17_30)
// CHECK:STDOUT:   %.loc17_31.2: ref i32 = class_element_access file.%a_ref.var, element0
// CHECK:STDOUT:   %.loc17_31.3: init i32 = initialize_from %.loc17_22 to %.loc17_31.2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_31.4: ref i32 = class_element_access file.%a_ref.var, element1
// CHECK:STDOUT:   %.loc17_31.5: init i32 = initialize_from %.loc17_30 to %.loc17_31.4 [template = constants.%.7]
// CHECK:STDOUT:   %.loc17_31.6: init %A = class_init (%.loc17_31.3, %.loc17_31.5), file.%a_ref.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc17_32: init %A = converted %.loc17_31.1, %.loc17_31.6 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc17_32
// CHECK:STDOUT:   %a_ref.ref.loc18: ref %A = name_ref a_ref, file.%a_ref
// CHECK:STDOUT:   %.loc18: %A = bind_value %a_ref.ref.loc18
// CHECK:STDOUT:   %a_val: %A = bind_name a_val, %.loc18
// CHECK:STDOUT:   %a_val.ref: %A = name_ref a_val, %a_val
// CHECK:STDOUT:   %B.ref.loc21: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21_22.1: %B = as_compatible %a_val.ref
// CHECK:STDOUT:   %.loc21_22.2: %B = converted %a_val.ref, %.loc21_22.1
// CHECK:STDOUT:   %b_val: %B = bind_name b_val, %.loc21_22.2
// CHECK:STDOUT:   %a_ref.ref.loc22: ref %A = name_ref a_ref, file.%a_ref
// CHECK:STDOUT:   %B.ref.loc22: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc22_25.1: ref %B = as_compatible %a_ref.ref.loc22
// CHECK:STDOUT:   %.loc22_25.2: ref %B = converted %a_ref.ref.loc22, %.loc22_25.1
// CHECK:STDOUT:   %.loc22_17: %.9 = addr_of %.loc22_25.2
// CHECK:STDOUT:   %b_ptr: %.9 = bind_name b_ptr, %.loc22_17
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, @A.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc24_5: ref %B = splice_block file.%b_factory.var {}
// CHECK:STDOUT:   %Make.call: init %A = call %Make.ref() to %.loc24_5
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc24_29.1: init %B = as_compatible %Make.call
// CHECK:STDOUT:   %.loc24_29.2: init %B = converted %Make.call, %.loc24_29.1
// CHECK:STDOUT:   assign file.%b_factory.var, %.loc24_29.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness i32 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc9_8.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc9_8.2: type = converted %int.make_type_32, %.loc9_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_12.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc5_12.2: type = converted %int.make_type_32, %.loc5_12.1 [template = i32]
// CHECK:STDOUT:   adapt_decl i32
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness i32 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_13: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_18.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_18.2: type = converted %int.make_type_32.loc8, %.loc8_18.1 [template = i32]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc8_23.1: %A = as_compatible %.loc8_13 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_23.2: %A = converted %.loc8_13, %.loc8_23.1 [template = constants.%.3]
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc8_23.2
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %int.make_type_32.loc9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc9_19.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:   %.loc9_19.2: type = converted %int.make_type_32.loc9, %.loc9_19.1 [template = i32]
// CHECK:STDOUT:   %.loc9_16.1: i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc9_16.2: i32 = converted %a.ref, %.loc9_16.1
// CHECK:STDOUT:   %n: i32 = bind_name n, %.loc9_16.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multi_level_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %D.ref: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc4_18: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc4_19: type = converted %.loc4_18, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   adapt_decl %.1
// CHECK:STDOUT:   %.loc4_21: <witness> = complete_type_witness %.1 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   adapt_decl %B
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   adapt_decl %C
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_13: %.1 = struct_literal ()
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %struct: %.1 = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_15.1: %D = as_compatible %struct [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_15.2: %D = converted %.loc9_13, %.loc9_15.1 [template = constants.%struct]
// CHECK:STDOUT:   %d: %D = bind_name d, %.loc9_15.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_init_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %A [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct: %A = struct_value (%.7, %.8) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .b_value = @__global_init.%b_value
// CHECK:STDOUT:     .b_init = %b_init
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %B.ref.loc13: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %b_init.var: ref %B = var b_init
// CHECK:STDOUT:   %b_init: ref %B = bind_name b_init, %b_init.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_32.loc5, %.loc5_10.1 [template = i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl x, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_10.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_10.2: type = converted %int.make_type_32.loc6, %.loc6_10.1 [template = i32]
// CHECK:STDOUT:   %.loc6_8: %.2 = field_decl y, element1 [template]
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT:   .y = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %A [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_25: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_33: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_34.1: %.3 = struct_literal (%.loc13_25, %.loc13_33)
// CHECK:STDOUT:   %A.ref.loc13: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc13_34.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc13_34.3: ref i32 = class_element_access %.loc13_34.2, element0
// CHECK:STDOUT:   %.loc13_34.4: init i32 = initialize_from %.loc13_25 to %.loc13_34.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_34.5: ref i32 = class_element_access %.loc13_34.2, element1
// CHECK:STDOUT:   %.loc13_34.6: init i32 = initialize_from %.loc13_33 to %.loc13_34.5 [template = constants.%.8]
// CHECK:STDOUT:   %.loc13_34.7: init %A = class_init (%.loc13_34.4, %.loc13_34.6), %.loc13_34.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_34.8: ref %A = temporary %.loc13_34.2, %.loc13_34.7
// CHECK:STDOUT:   %.loc13_36: ref %A = converted %.loc13_34.1, %.loc13_34.8
// CHECK:STDOUT:   %B.ref.loc13: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13_42.1: ref %B = as_compatible %.loc13_36
// CHECK:STDOUT:   %.loc13_42.2: ref %B = converted %.loc13_36, %.loc13_42.1
// CHECK:STDOUT:   %.loc13_42.3: %B = bind_value %.loc13_42.2
// CHECK:STDOUT:   %b_value: %B = bind_name b_value, %.loc13_42.3
// CHECK:STDOUT:   %.loc24_24: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc24_32: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc24_33.1: %.3 = struct_literal (%.loc24_24, %.loc24_32)
// CHECK:STDOUT:   %A.ref.loc24: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc24_33.2: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc24_33.3: ref i32 = class_element_access %.loc24_33.2, element0
// CHECK:STDOUT:   %.loc24_33.4: init i32 = initialize_from %.loc24_24 to %.loc24_33.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc24_33.5: ref i32 = class_element_access %.loc24_33.2, element1
// CHECK:STDOUT:   %.loc24_33.6: init i32 = initialize_from %.loc24_32 to %.loc24_33.5 [template = constants.%.8]
// CHECK:STDOUT:   %.loc24_33.7: init %A = class_init (%.loc24_33.4, %.loc24_33.6), %.loc24_33.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc24_33.8: ref %A = temporary %.loc24_33.2, %.loc24_33.7
// CHECK:STDOUT:   %.loc24_35: ref %A = converted %.loc24_33.1, %.loc24_33.8
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc24_41.1: ref %B = as_compatible %.loc24_35
// CHECK:STDOUT:   %.loc24_41.2: ref %B = converted %.loc24_35, %.loc24_41.1
// CHECK:STDOUT:   %.loc24_41.3: %B = bind_value %.loc24_41.2
// CHECK:STDOUT:   assign file.%b_init.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_init_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %A [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %As.type.2: type = interface_type @As, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @As(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %As.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = interface_type @As, @As(%B) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @As(%B) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.10: type = assoc_entity_type %As.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.11: %.10 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.12: %.8 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %struct: %.3 = struct_value (%.7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %As.type.1 = import_ref Core//prelude/operators/as, inst+5, loaded [template = constants.%As]
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//prelude/operators/as, inst+11, unloaded
// CHECK:STDOUT:   %import_ref.4: @As.%.1 (%.8) = import_ref Core//prelude/operators/as, inst+29, loaded [symbolic = @As.%.2 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+22, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Core//prelude/operators/as, inst+22, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %b.var: ref %B = var b
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = interface_type @As, @As(%Dest) [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @As(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.8)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.8) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_32, %.loc5_10.1 [template = i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %A [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %As.type: type = interface_type @As, @As(%Dest) [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc21_18: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_19.1: %.3 = struct_literal (%.loc21_18)
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %As.type: type = interface_type @As, @As(constants.%B) [template = constants.%As.type.3]
// CHECK:STDOUT:   %.loc21_21.1: %.10 = specific_constant imports.%import_ref.4, @As(constants.%B) [template = constants.%.11]
// CHECK:STDOUT:   %Convert.ref: %.10 = name_ref Convert, %.loc21_21.1 [template = constants.%.11]
// CHECK:STDOUT:   %struct: %.3 = struct_value (%.loc21_18) [template = constants.%struct]
// CHECK:STDOUT:   %.loc21_19.2: %.3 = converted %.loc21_19.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %.loc21_21.2: %B = converted %.loc21_19.1, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@As.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%B) {
// CHECK:STDOUT:   %Dest => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.10
// CHECK:STDOUT:   %.2 => constants.%.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
