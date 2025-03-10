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

// --- init_class_value.carbon

library "[[@TEST_NAME]]";

class A {
  var x: i32;
  var y: i32;
}

class B {
  adapt A;
}

let b_value: B = ({.x = 1, .y = 2} as A) as B;

// --- fail_init_class_variable.carbon

library "[[@TEST_NAME]]";

class A {
  var x: i32;
  var y: i32;
}

class B {
  adapt A;
}

// TODO: Here, we treat `{.x = 1, .y = 2} as A` as a value expression, not an
// initializing expression, so `(...) as B` is a value expression too, requiring
// a copy to perform initialization. It's not clear whether that is the right
// behavior.

// CHECK:STDERR: fail_init_class_variable.carbon:[[@LINE+4]]:17: error: cannot copy value of type `B` [CopyOfUncopyableType]
// CHECK:STDERR: var b_init: B = ({.x = 1, .y = 2} as A) as B;
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var b_init: B = ({.x = 1, .y = 2} as A) as B;

// --- init_tuple_value.carbon

library "[[@TEST_NAME]]";

class Noncopyable {
  // TODO: Ensure this remains non-copyable once we have rules for class copyability.
}

class A {
  adapt (i32, Noncopyable);
}

fn F(a: A) {
  let a_value: A = (a as (i32, Noncopyable)) as A;
}

// --- fail_init_tuple_variable.carbon

library "[[@TEST_NAME]]";

class Noncopyable {
  // TODO: Ensure this remains non-copyable once we have rules for class copyability.
}

class A {
  adapt (i32, Noncopyable);
}

fn F(a: A) {
  // TODO: Here, we treat `a as (i32, Noncopyable)` as a value expression, not an
  // initializing expression, so `(...) as A` is a value expression too, requiring
  // a copy to perform initialization. It's not clear whether that is the right
  // behavior.

  // CHECK:STDERR: fail_init_tuple_variable.carbon:[[@LINE+7]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var a_init: A = (a as (i32, Noncopyable)) as A;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_init_tuple_variable.carbon:[[@LINE+4]]:19: note: in copy of `A` [InCopy]
  // CHECK:STDERR:   var a_init: A = (a as (i32, Noncopyable)) as A;
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a_init: A = (a as (i32, Noncopyable)) as A;
}

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

// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+7]]:12: error: cannot convert value of type `{.x: Core.IntLiteral}` to `B` with `as` [ExplicitAsConversionFailure]
// CHECK:STDERR: var b: B = {.x = 1} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~
// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+4]]:12: note: type `{.x: Core.IntLiteral}` does not implement interface `Core.As(B)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: B = {.x = 1} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~
// CHECK:STDERR:
var b: B = {.x = 1} as B;

// CHECK:STDOUT: --- adapt_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y.871: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.70a: <witness> = complete_type_witness %struct_type.x.y.871 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.y.4cf: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .a_val = %a_val
// CHECK:STDOUT:     .b_val = %b_val
// CHECK:STDOUT:     .b_ptr = %b_ptr
// CHECK:STDOUT:     .b_factory = %b_factory
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ref.patt: %A = binding_pattern a_ref
// CHECK:STDOUT:     %.loc17: %A = var_pattern %a_ref.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ref.var: ref %A = var a_ref
// CHECK:STDOUT:   %A.ref.loc17: type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a_ref: ref %A = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_val.patt: %A = binding_pattern a_val
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref.loc18: type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a_val: ref %A = bind_name a_val, @__global_init.%a_ref.ref.loc18
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_val.patt: %B = binding_pattern b_val
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.ref.loc21: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_val: ref %B = bind_name b_val, @__global_init.%.loc21_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ptr.patt: %ptr.e79 = binding_pattern b_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22: type = splice_block %ptr [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:     %B.ref.loc22: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr: type = ptr_type %B [concrete = constants.%ptr.e79]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr: %ptr.e79 = bind_name b_ptr, @__global_init.%addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_factory.patt: %B = binding_pattern b_factory
// CHECK:STDOUT:     %.loc24: %B = var_pattern %b_factory.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_factory.var: ref %B = var b_factory
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_factory: ref %B = bind_name b_factory, %b_factory.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5_8: %A.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %A.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc5: ref %A.elem = var <none>
// CHECK:STDOUT:   %.loc6_8: %A.elem = field_decl y, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6_3: %A.elem = var_pattern %.loc6_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %A.elem = var <none>
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %A = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %A = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %return.param: ref %A = out_param call_param0
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.871 [concrete = constants.%complete_type.70a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT:   .y = %.loc6_8
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.871 [concrete = constants.%complete_type.70a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param_patt: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc9_27.1: %struct_type.x.y.4cf = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc9_27.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_27.1: <bound method> = bound_method %int_1, %impl.elem0.loc9_27.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc9_27.1: <specific function> = specific_function %impl.elem0.loc9_27.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_27.2: <bound method> = bound_method %int_1, %specific_fn.loc9_27.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc9_27.1: init %i32 = call %bound_method.loc9_27.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_27.2: init %i32 = converted %int_1, %int.convert_checked.loc9_27.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_27.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc9_27.4: init %i32 = initialize_from %.loc9_27.2 to %.loc9_27.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc9_27.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_27.3: <bound method> = bound_method %int_2, %impl.elem0.loc9_27.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc9_27.2: <specific function> = specific_function %impl.elem0.loc9_27.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_27.4: <bound method> = bound_method %int_2, %specific_fn.loc9_27.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc9_27.2: init %i32 = call %bound_method.loc9_27.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_27.5: init %i32 = converted %int_2, %int.convert_checked.loc9_27.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_27.6: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc9_27.7: init %i32 = initialize_from %.loc9_27.5 to %.loc9_27.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_27.8: init %A = class_init (%.loc9_27.4, %.loc9_27.7), %return [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc9_28: init %A = converted %.loc9_27.1, %.loc9_27.8 [concrete = constants.%A.val]
// CHECK:STDOUT:   return %.loc9_28 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc17_31.1: %struct_type.x.y.4cf = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc17_31.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_31.1: <bound method> = bound_method %int_1, %impl.elem0.loc17_31.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc17_31.1: <specific function> = specific_function %impl.elem0.loc17_31.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_31.2: <bound method> = bound_method %int_1, %specific_fn.loc17_31.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc17_31.1: init %i32 = call %bound_method.loc17_31.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_31.2: init %i32 = converted %int_1, %int.convert_checked.loc17_31.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_31.3: ref %i32 = class_element_access file.%a_ref.var, element0
// CHECK:STDOUT:   %.loc17_31.4: init %i32 = initialize_from %.loc17_31.2 to %.loc17_31.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc17_31.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_31.3: <bound method> = bound_method %int_2, %impl.elem0.loc17_31.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc17_31.2: <specific function> = specific_function %impl.elem0.loc17_31.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_31.4: <bound method> = bound_method %int_2, %specific_fn.loc17_31.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc17_31.2: init %i32 = call %bound_method.loc17_31.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_31.5: init %i32 = converted %int_2, %int.convert_checked.loc17_31.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_31.6: ref %i32 = class_element_access file.%a_ref.var, element1
// CHECK:STDOUT:   %.loc17_31.7: init %i32 = initialize_from %.loc17_31.5 to %.loc17_31.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_31.8: init %A = class_init (%.loc17_31.4, %.loc17_31.7), file.%a_ref.var [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc17_1: init %A = converted %.loc17_31.1, %.loc17_31.8 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc17_1
// CHECK:STDOUT:   %a_ref.ref.loc18: ref %A = name_ref a_ref, file.%a_ref
// CHECK:STDOUT:   %a_val.ref: ref %A = name_ref a_val, file.%a_val
// CHECK:STDOUT:   %B.ref.loc21: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc21_22.1: ref %B = as_compatible %a_val.ref
// CHECK:STDOUT:   %.loc21_22.2: ref %B = converted %a_val.ref, %.loc21_22.1
// CHECK:STDOUT:   %a_ref.ref.loc22: ref %A = name_ref a_ref, file.%a_ref
// CHECK:STDOUT:   %B.ref.loc22: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc22_25.1: ref %B = as_compatible %a_ref.ref.loc22
// CHECK:STDOUT:   %.loc22_25.2: ref %B = converted %a_ref.ref.loc22, %.loc22_25.1
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc22_25.2
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, @A.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc24_1: ref %B = splice_block file.%b_factory.var {}
// CHECK:STDOUT:   %Make.call: init %A = call %Make.ref() to %.loc24_1
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc24_29.1: init %B = as_compatible %Make.call
// CHECK:STDOUT:   %.loc24_29.2: init %B = converted %Make.call, %.loc24_29.1
// CHECK:STDOUT:   assign file.%b_factory.var, %.loc24_29.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.1, @As(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.882: <witness> = impl_witness (imports.%Core.import_ref.78a), @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%impl_witness.882) [concrete]
// CHECK:STDOUT:   %.249: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.e78: %A = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref: type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: %A = bind_name a, @__global_init.%.loc8_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: %i32 = bind_name n, @__global_init.%.loc9_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %i32.builtin [concrete = constants.%complete_type.f8a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.249 = impl_witness_access constants.%impl_witness.882, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc8_15.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_15.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_15.2: %i32 = converted %int_1, %.loc8_15.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc8_23.1: %A = as_compatible %.loc8_15.2 [concrete = constants.%int_1.e78]
// CHECK:STDOUT:   %.loc8_23.2: %A = converted %.loc8_15.2, %.loc8_23.1 [concrete = constants.%int_1.e78]
// CHECK:STDOUT:   %a.ref: %A = name_ref a, file.%a
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9_16.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc9_16.2: %i32 = converted %a.ref, %.loc9_16.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multi_level_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.ref: type = name_ref D, %D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %d: %D = bind_name d, @__global_init.%.loc9_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc4_18: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc4_19: type = converted %.loc4_18, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc4_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   adapt_decl %B.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   adapt_decl %C.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_15.1: %D = as_compatible %empty_struct [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc9_15.2: %D = converted %.loc9_13, %.loc9_15.1 [concrete = constants.%D.val]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_class_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.y.871: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.70a: <witness> = complete_type_witness %struct_type.x.y.871 [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.y.4cf: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .b_value = %b_value
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_value.patt: %B = binding_pattern b_value
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_value: ref %B = bind_name b_value, @__global_init.%.loc13_42.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5_8: %A.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %A.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc5: ref %A.elem = var <none>
// CHECK:STDOUT:   %.loc6_8: %A.elem = field_decl y, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6_3: %A.elem = var_pattern %.loc6_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %A.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.871 [concrete = constants.%complete_type.70a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT:   .y = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.871 [concrete = constants.%complete_type.70a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc13_34.1: %struct_type.x.y.4cf = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %impl.elem0.loc13_34.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc13_34.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc13_34.1: <specific function> = specific_function %impl.elem0.loc13_34.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_1, %specific_fn.loc13_34.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc13_34.1: init %i32 = call %bound_method.loc13_34.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_34.2: init %i32 = converted %int_1, %int.convert_checked.loc13_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_34.3: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc13_34.4: ref %i32 = class_element_access %.loc13_34.3, element0
// CHECK:STDOUT:   %.loc13_34.5: init %i32 = initialize_from %.loc13_34.2 to %.loc13_34.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc13_34.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_34.3: <bound method> = bound_method %int_2, %impl.elem0.loc13_34.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc13_34.2: <specific function> = specific_function %impl.elem0.loc13_34.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.4: <bound method> = bound_method %int_2, %specific_fn.loc13_34.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc13_34.2: init %i32 = call %bound_method.loc13_34.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_34.6: init %i32 = converted %int_2, %int.convert_checked.loc13_34.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_34.7: ref %i32 = class_element_access %.loc13_34.3, element1
// CHECK:STDOUT:   %.loc13_34.8: init %i32 = initialize_from %.loc13_34.6 to %.loc13_34.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_34.9: init %A = class_init (%.loc13_34.5, %.loc13_34.8), %.loc13_34.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc13_34.10: ref %A = temporary %.loc13_34.3, %.loc13_34.9
// CHECK:STDOUT:   %.loc13_36: ref %A = converted %.loc13_34.1, %.loc13_34.10
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc13_42.1: ref %B = as_compatible %.loc13_36
// CHECK:STDOUT:   %.loc13_42.2: ref %B = converted %.loc13_36, %.loc13_42.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_init_class_variable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.y.871: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.70a: <witness> = complete_type_witness %struct_type.x.y.871 [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.y.4cf: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .b_init = %b_init
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_init.patt: %B = binding_pattern b_init
// CHECK:STDOUT:     %.loc22: %B = var_pattern %b_init.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_init.var: ref %B = var b_init
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_init: ref %B = bind_name b_init, %b_init.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5_8: %A.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %A.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc5: ref %A.elem = var <none>
// CHECK:STDOUT:   %.loc6_8: %A.elem = field_decl y, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6_3: %A.elem = var_pattern %.loc6_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %A.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.871 [concrete = constants.%complete_type.70a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT:   .y = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.871 [concrete = constants.%complete_type.70a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc22_33.1: %struct_type.x.y.4cf = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %impl.elem0.loc22_33.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_33.1: <bound method> = bound_method %int_1, %impl.elem0.loc22_33.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc22_33.1: <specific function> = specific_function %impl.elem0.loc22_33.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_33.2: <bound method> = bound_method %int_1, %specific_fn.loc22_33.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc22_33.1: init %i32 = call %bound_method.loc22_33.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_33.2: init %i32 = converted %int_1, %int.convert_checked.loc22_33.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_33.3: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc22_33.4: ref %i32 = class_element_access %.loc22_33.3, element0
// CHECK:STDOUT:   %.loc22_33.5: init %i32 = initialize_from %.loc22_33.2 to %.loc22_33.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc22_33.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_33.3: <bound method> = bound_method %int_2, %impl.elem0.loc22_33.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc22_33.2: <specific function> = specific_function %impl.elem0.loc22_33.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_33.4: <bound method> = bound_method %int_2, %specific_fn.loc22_33.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc22_33.2: init %i32 = call %bound_method.loc22_33.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_33.6: init %i32 = converted %int_2, %int.convert_checked.loc22_33.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_33.7: ref %i32 = class_element_access %.loc22_33.3, element1
// CHECK:STDOUT:   %.loc22_33.8: init %i32 = initialize_from %.loc22_33.6 to %.loc22_33.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_33.9: init %A = class_init (%.loc22_33.5, %.loc22_33.8), %.loc22_33.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc22_33.10: ref %A = temporary %.loc22_33.3, %.loc22_33.9
// CHECK:STDOUT:   %.loc22_35: ref %A = converted %.loc22_33.1, %.loc22_33.10
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc22_41.1: ref %B = as_compatible %.loc22_35
// CHECK:STDOUT:   %.loc22_41.2: ref %B = converted %.loc22_35, %.loc22_41.1
// CHECK:STDOUT:   %.loc22_41.3: %B = bind_value %.loc22_41.2
// CHECK:STDOUT:   assign file.%b_init.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_tuple_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.560: type = tuple_type (%i32, %Noncopyable) [concrete]
// CHECK:STDOUT:   %complete_type.ce5: <witness> = complete_type_witness %tuple.type.560 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Noncopyable = %Noncopyable.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Noncopyable.decl: type = class_decl @Noncopyable [concrete = constants.%Noncopyable] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref.loc12: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc9_26: %tuple.type.24b = tuple_literal (%i32, %Noncopyable.ref)
// CHECK:STDOUT:   %.loc9_27: type = converted %.loc9_26, constants.%tuple.type.560 [concrete = constants.%tuple.type.560]
// CHECK:STDOUT:   adapt_decl %.loc9_27 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %tuple.type.560 [concrete = constants.%complete_type.ce5]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .Noncopyable = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_value.patt: %A = binding_pattern a_value
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc13_43.1: %tuple.type.24b = tuple_literal (%i32, %Noncopyable.ref)
// CHECK:STDOUT:   %.loc13_43.2: type = converted %.loc13_43.1, constants.%tuple.type.560 [concrete = constants.%tuple.type.560]
// CHECK:STDOUT:   %.loc13_23.1: %tuple.type.560 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc13_23.2: %tuple.type.560 = converted %a.ref, %.loc13_23.1
// CHECK:STDOUT:   %A.ref.loc13_49: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc13_46.1: %A = as_compatible %.loc13_23.2
// CHECK:STDOUT:   %.loc13_46.2: %A = converted %.loc13_23.2, %.loc13_46.1
// CHECK:STDOUT:   %A.ref.loc13_16: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a_value: %A = bind_name a_value, %.loc13_46.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_init_tuple_variable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.560: type = tuple_type (%i32, %Noncopyable) [concrete]
// CHECK:STDOUT:   %complete_type.ce5: <witness> = complete_type_witness %tuple.type.560 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Noncopyable = %Noncopyable.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Noncopyable.decl: type = class_decl @Noncopyable [concrete = constants.%Noncopyable] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref.loc12: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc9_26: %tuple.type.24b = tuple_literal (%i32, %Noncopyable.ref)
// CHECK:STDOUT:   %.loc9_27: type = converted %.loc9_26, constants.%tuple.type.560 [concrete = constants.%tuple.type.560]
// CHECK:STDOUT:   adapt_decl %.loc9_27 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %tuple.type.560 [concrete = constants.%complete_type.ce5]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .Noncopyable = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_init.patt: %A = binding_pattern a_init
// CHECK:STDOUT:     %.loc25_3.1: %A = var_pattern %a_init.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_init.var: ref %A = var a_init
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc25_42.1: %tuple.type.24b = tuple_literal (%i32, %Noncopyable.ref)
// CHECK:STDOUT:   %.loc25_42.2: type = converted %.loc25_42.1, constants.%tuple.type.560 [concrete = constants.%tuple.type.560]
// CHECK:STDOUT:   %.loc25_22.1: %tuple.type.560 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc25_22.2: %tuple.type.560 = converted %a.ref, %.loc25_22.1
// CHECK:STDOUT:   %A.ref.loc25_48: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc25_45.1: %A = as_compatible %.loc25_22.2
// CHECK:STDOUT:   %.loc25_45.2: %A = converted %.loc25_22.2, %.loc25_45.1
// CHECK:STDOUT:   %.loc25_3.2: %tuple.type.560 = as_compatible %.loc25_45.2
// CHECK:STDOUT:   %tuple.elem0.loc25_3.1: %i32 = tuple_access %.loc25_3.2, element0
// CHECK:STDOUT:   %.loc25_3.3: ref %tuple.type.560 = as_compatible %a_init.var
// CHECK:STDOUT:   %tuple.elem0.loc25_3.2: ref %i32 = tuple_access %.loc25_3.3, element0
// CHECK:STDOUT:   %.loc25_3.4: init %i32 = initialize_from %tuple.elem0.loc25_3.1 to %tuple.elem0.loc25_3.2
// CHECK:STDOUT:   %tuple.elem1: %Noncopyable = tuple_access %.loc25_3.2, element1
// CHECK:STDOUT:   assign %a_init.var, <error>
// CHECK:STDOUT:   %A.ref.loc25_15: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a_init: ref %A = bind_name a_init, %a_init.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_init_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.x.c96: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %B = binding_pattern b
// CHECK:STDOUT:     %.loc22: %B = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var b
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5_8: %A.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %A.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %A.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc22_19: %struct_type.x.c96 = struct_literal (%int_1)
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc22_21: %B = converted %.loc22_19, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
