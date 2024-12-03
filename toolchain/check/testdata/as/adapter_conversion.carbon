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

// CHECK:STDERR: fail_init_class.carbon:[[@LINE+4]]:17: error: cannot copy value of type `B` [CopyOfUncopyableType]
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

// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+6]]:12: error: cannot convert from `{.x: Core.IntLiteral}` to `B` with `as` [ExplicitAsConversionFailure]
// CHECK:STDERR: var b: B = {.x = 1} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~
// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+3]]:12: note: type `{.x: Core.IntLiteral}` does not implement interface `As(B)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: B = {.x = 1} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~
var b: B = {.x = 1} as B;

// CHECK:STDOUT: --- adapt_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.x.y.1: type = struct_type {.x: %i32, .y: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.1 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.x.y.2: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %A = struct_value (%int.1.2, %int.2.2) [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.12: type = ptr_type %B [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %.loc22: type = ptr_type %B [template = constants.%.12]
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:   %b_factory.var: ref %B = var b_factory
// CHECK:STDOUT:   %b_factory: ref %B = bind_name b_factory, %b_factory.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int.32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_signed.loc5, %.loc5_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8: %.1 = field_decl x, element0 [template]
// CHECK:STDOUT:   %int.32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int.32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.2: type = converted %int.make_type_signed.loc6, %.loc6_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8: %.1 = field_decl y, element1 [template]
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %A = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %A = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %return.param: ref %A = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.1 [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT:   .y = %.loc6_8
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.1 [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc9_27.1: %struct_type.x.y.2 = struct_literal (%int.1, %int.2)
// CHECK:STDOUT:   %.loc9_27.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc9_27.1: <bound method> = bound_method %int.1, %.loc9_27.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_27.1: <specific function> = specific_function %Convert.bound.loc9_27.1, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_27.1: init %i32 = call %Convert.specific_fn.loc9_27.1(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc9_27.3: init %i32 = converted %int.1, %int.convert_checked.loc9_27.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc9_27.4: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc9_27.5: init %i32 = initialize_from %.loc9_27.3 to %.loc9_27.4 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc9_27.6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc9_27.2: <bound method> = bound_method %int.2, %.loc9_27.6 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_27.2: <specific function> = specific_function %Convert.bound.loc9_27.2, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_27.2: init %i32 = call %Convert.specific_fn.loc9_27.2(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc9_27.7: init %i32 = converted %int.2, %int.convert_checked.loc9_27.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc9_27.8: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc9_27.9: init %i32 = initialize_from %.loc9_27.7 to %.loc9_27.8 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc9_27.10: init %A = class_init (%.loc9_27.5, %.loc9_27.9), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_28: init %A = converted %.loc9_27.1, %.loc9_27.10 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc9_28 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc17_31.1: %struct_type.x.y.2 = struct_literal (%int.1, %int.2)
// CHECK:STDOUT:   %.loc17_31.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc17_31.1: <bound method> = bound_method %int.1, %.loc17_31.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_31.1: <specific function> = specific_function %Convert.bound.loc17_31.1, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc17_31.1: init %i32 = call %Convert.specific_fn.loc17_31.1(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc17_31.3: init %i32 = converted %int.1, %int.convert_checked.loc17_31.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc17_31.4: ref %i32 = class_element_access file.%a_ref.var, element0
// CHECK:STDOUT:   %.loc17_31.5: init %i32 = initialize_from %.loc17_31.3 to %.loc17_31.4 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc17_31.6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc17_31.2: <bound method> = bound_method %int.2, %.loc17_31.6 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_31.2: <specific function> = specific_function %Convert.bound.loc17_31.2, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_31.2: init %i32 = call %Convert.specific_fn.loc17_31.2(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc17_31.7: init %i32 = converted %int.2, %int.convert_checked.loc17_31.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc17_31.8: ref %i32 = class_element_access file.%a_ref.var, element1
// CHECK:STDOUT:   %.loc17_31.9: init %i32 = initialize_from %.loc17_31.7 to %.loc17_31.8 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc17_31.10: init %A = class_init (%.loc17_31.5, %.loc17_31.9), file.%a_ref.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc17_32: init %A = converted %.loc17_31.1, %.loc17_31.10 [template = constants.%struct]
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
// CHECK:STDOUT:   %.loc22_17: %.12 = addr_of %.loc22_25.2
// CHECK:STDOUT:   %b_ptr: %.12 = bind_name b_ptr, %.loc22_17
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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %i32 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @As(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.7, @impl.5(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.7(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_8.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_8.2: type = converted %int.make_type_signed, %.loc9_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_12.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_12.2: type = converted %int.make_type_signed, %.loc5_12.1 [template = constants.%i32]
// CHECK:STDOUT:   adapt_decl %.loc5_12.2 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %i32 [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int.32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_18.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_18.2: type = converted %int.make_type_signed.loc8, %.loc8_18.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_15.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int.1, %.loc8_15.1 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.7(constants.%int.32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc8_15.2: %i32 = value_of_initializer %int.convert_checked [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc8_15.3: %i32 = converted %int.1, %.loc8_15.2 [template = constants.%int.1.2]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc8_23.1: %A = as_compatible %.loc8_15.3 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc8_23.2: %A = converted %.loc8_15.3, %.loc8_23.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %a: %A = bind_name a, %.loc8_23.2
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %int.32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%int.32.loc9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_19.1: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_19.2: type = converted %int.make_type_signed.loc9, %.loc9_19.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_16.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc9_16.2: %i32 = converted %a.ref, %.loc9_16.1
// CHECK:STDOUT:   %n: %i32 = bind_name n, %.loc9_16.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multi_level_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
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
// CHECK:STDOUT:   %.loc4_18: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc4_19: type = converted %.loc4_18, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc4_19 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   adapt_decl %B.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   adapt_decl %C.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %struct: %empty_struct_type = struct_value () [template = constants.%struct]
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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.y.1: type = struct_type {.x: %i32, .y: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.1 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.x.y.2: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %A = struct_value (%int.1.2, %int.2.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int.32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int.32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_signed.loc5, %.loc5_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8: %.1 = field_decl x, element0 [template]
// CHECK:STDOUT:   %int.32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int.32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.2: type = converted %int.make_type_signed.loc6, %.loc6_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8: %.1 = field_decl y, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.1 [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT:   .y = %.loc6_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.1 [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2.loc13: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc13_34.1: %struct_type.x.y.2 = struct_literal (%int.1.loc13, %int.2.loc13)
// CHECK:STDOUT:   %A.ref.loc13: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc13_34.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_34.1: <bound method> = bound_method %int.1.loc13, %.loc13_34.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_34.1: <specific function> = specific_function %Convert.bound.loc13_34.1, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_34.1: init %i32 = call %Convert.specific_fn.loc13_34.1(%int.1.loc13) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc13_34.3: init %i32 = converted %int.1.loc13, %int.convert_checked.loc13_34.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc13_34.4: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc13_34.5: ref %i32 = class_element_access %.loc13_34.4, element0
// CHECK:STDOUT:   %.loc13_34.6: init %i32 = initialize_from %.loc13_34.3 to %.loc13_34.5 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc13_34.7: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_34.2: <bound method> = bound_method %int.2.loc13, %.loc13_34.7 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_34.2: <specific function> = specific_function %Convert.bound.loc13_34.2, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_34.2: init %i32 = call %Convert.specific_fn.loc13_34.2(%int.2.loc13) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc13_34.8: init %i32 = converted %int.2.loc13, %int.convert_checked.loc13_34.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc13_34.9: ref %i32 = class_element_access %.loc13_34.4, element1
// CHECK:STDOUT:   %.loc13_34.10: init %i32 = initialize_from %.loc13_34.8 to %.loc13_34.9 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc13_34.11: init %A = class_init (%.loc13_34.6, %.loc13_34.10), %.loc13_34.4 [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_34.12: ref %A = temporary %.loc13_34.4, %.loc13_34.11
// CHECK:STDOUT:   %.loc13_36: ref %A = converted %.loc13_34.1, %.loc13_34.12
// CHECK:STDOUT:   %B.ref.loc13: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13_42.1: ref %B = as_compatible %.loc13_36
// CHECK:STDOUT:   %.loc13_42.2: ref %B = converted %.loc13_36, %.loc13_42.1
// CHECK:STDOUT:   %.loc13_42.3: %B = bind_value %.loc13_42.2
// CHECK:STDOUT:   %b_value: %B = bind_name b_value, %.loc13_42.3
// CHECK:STDOUT:   %int.1.loc24: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2.loc24: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc24_33.1: %struct_type.x.y.2 = struct_literal (%int.1.loc24, %int.2.loc24)
// CHECK:STDOUT:   %A.ref.loc24: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc24_33.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc24_33.1: <bound method> = bound_method %int.1.loc24, %.loc24_33.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_33.1: <specific function> = specific_function %Convert.bound.loc24_33.1, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc24_33.1: init %i32 = call %Convert.specific_fn.loc24_33.1(%int.1.loc24) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc24_33.3: init %i32 = converted %int.1.loc24, %int.convert_checked.loc24_33.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc24_33.4: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc24_33.5: ref %i32 = class_element_access %.loc24_33.4, element0
// CHECK:STDOUT:   %.loc24_33.6: init %i32 = initialize_from %.loc24_33.3 to %.loc24_33.5 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc24_33.7: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc24_33.2: <bound method> = bound_method %int.2.loc24, %.loc24_33.7 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_33.2: <specific function> = specific_function %Convert.bound.loc24_33.2, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc24_33.2: init %i32 = call %Convert.specific_fn.loc24_33.2(%int.2.loc24) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc24_33.8: init %i32 = converted %int.2.loc24, %int.convert_checked.loc24_33.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc24_33.9: ref %i32 = class_element_access %.loc24_33.4, element1
// CHECK:STDOUT:   %.loc24_33.10: init %i32 = initialize_from %.loc24_33.8 to %.loc24_33.9 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc24_33.11: init %A = class_init (%.loc24_33.6, %.loc24_33.10), %.loc24_33.4 [template = constants.%struct]
// CHECK:STDOUT:   %.loc24_33.12: ref %A = temporary %.loc24_33.4, %.loc24_33.11
// CHECK:STDOUT:   %.loc24_35: ref %A = converted %.loc24_33.1, %.loc24_33.12
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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.1 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: Core.IntLiteral} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_signed, %.loc5_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8: %.1 = field_decl x, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.1 [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.1 [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %.loc21_19: %struct_type.x.2 = struct_literal (%int.1)
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21_21: %B = converted %.loc21_19, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
