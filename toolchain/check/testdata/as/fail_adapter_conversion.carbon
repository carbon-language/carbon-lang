// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/fail_adapter_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/fail_adapter_conversion.carbon

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

// --- fail_init_tuple.carbon

library "[[@TEST_NAME]]";

class Noncopyable {
  // TODO: Ensure this remains non-copyable once we have rules for class copyability.
}

class A {
  adapt (i32, Noncopyable);
}

fn F(a: A) {
  let a_value: A = (a as (i32, Noncopyable)) as A;

  // TODO: Here, we treat `a as (i32, Noncopyable)` as a value expression, not an
  // initializing expression, so `(...) as A` is a value expression too, requiring
  // a copy to perform initialization. It's not clear whether that is the right
  // behavior.

  // CHECK:STDERR: fail_init_tuple.carbon:[[@LINE+4]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var a_init: A = (a as (i32, Noncopyable)) as A;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+6]]:12: error: cannot convert from `{.x: Core.IntLiteral}` to `B` with `as` [ExplicitAsConversionFailure]
// CHECK:STDERR: var b: B = {.x = 1} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~
// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+3]]:12: note: type `{.x: Core.IntLiteral}` does not implement interface `Core.As(B)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: B = {.x = 1} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~
var b: B = {.x = 1} as B;

// CHECK:STDOUT: --- fail_init_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.y.1: type = struct_type {.x: %i32, .y: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.x.y.1 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.x.y.2: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.19: <witness> = impl_witness (imports.%import_ref.14), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
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
// CHECK:STDOUT:   %b_init.var: ref %B = var b_init
// CHECK:STDOUT:   %b_init: ref %B = bind_name b_init, %b_init.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5: %A.elem = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc6: %A.elem = field_decl y, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.1 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5
// CHECK:STDOUT:   .y = %.loc6
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y.1 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc13: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc13_34.1: %struct_type.x.y.2 = struct_literal (%int_1.loc13, %int_2.loc13)
// CHECK:STDOUT:   %A.ref.loc13: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %impl.elem0.loc13_34.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_34.1: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_34.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_34.1: <specific function> = specific_function %Convert.bound.loc13_34.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_34.1: init %i32 = call %Convert.specific_fn.loc13_34.1(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_34.2: init %i32 = converted %int_1.loc13, %int.convert_checked.loc13_34.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_34.3: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc13_34.4: ref %i32 = class_element_access %.loc13_34.3, element0
// CHECK:STDOUT:   %.loc13_34.5: init %i32 = initialize_from %.loc13_34.2 to %.loc13_34.4 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_34.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_34.2: <bound method> = bound_method %int_2.loc13, %impl.elem0.loc13_34.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_34.2: <specific function> = specific_function %Convert.bound.loc13_34.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_34.2: init %i32 = call %Convert.specific_fn.loc13_34.2(%int_2.loc13) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_34.6: init %i32 = converted %int_2.loc13, %int.convert_checked.loc13_34.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_34.7: ref %i32 = class_element_access %.loc13_34.3, element1
// CHECK:STDOUT:   %.loc13_34.8: init %i32 = initialize_from %.loc13_34.6 to %.loc13_34.7 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_34.9: init %A = class_init (%.loc13_34.5, %.loc13_34.8), %.loc13_34.3 [template = constants.%A.val]
// CHECK:STDOUT:   %.loc13_34.10: ref %A = temporary %.loc13_34.3, %.loc13_34.9
// CHECK:STDOUT:   %.loc13_36: ref %A = converted %.loc13_34.1, %.loc13_34.10
// CHECK:STDOUT:   %B.ref.loc13: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13_42.1: ref %B = as_compatible %.loc13_36
// CHECK:STDOUT:   %.loc13_42.2: ref %B = converted %.loc13_36, %.loc13_42.1
// CHECK:STDOUT:   %.loc13_42.3: %B = bind_value %.loc13_42.2
// CHECK:STDOUT:   %b_value: %B = bind_name b_value, %.loc13_42.3
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc24: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc24_33.1: %struct_type.x.y.2 = struct_literal (%int_1.loc24, %int_2.loc24)
// CHECK:STDOUT:   %A.ref.loc24: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %impl.elem0.loc24_33.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc24_33.1: <bound method> = bound_method %int_1.loc24, %impl.elem0.loc24_33.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_33.1: <specific function> = specific_function %Convert.bound.loc24_33.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc24_33.1: init %i32 = call %Convert.specific_fn.loc24_33.1(%int_1.loc24) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc24_33.2: init %i32 = converted %int_1.loc24, %int.convert_checked.loc24_33.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc24_33.3: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc24_33.4: ref %i32 = class_element_access %.loc24_33.3, element0
// CHECK:STDOUT:   %.loc24_33.5: init %i32 = initialize_from %.loc24_33.2 to %.loc24_33.4 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc24_33.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc24_33.2: <bound method> = bound_method %int_2.loc24, %impl.elem0.loc24_33.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_33.2: <specific function> = specific_function %Convert.bound.loc24_33.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc24_33.2: init %i32 = call %Convert.specific_fn.loc24_33.2(%int_2.loc24) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc24_33.6: init %i32 = converted %int_2.loc24, %int.convert_checked.loc24_33.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc24_33.7: ref %i32 = class_element_access %.loc24_33.3, element1
// CHECK:STDOUT:   %.loc24_33.8: init %i32 = initialize_from %.loc24_33.6 to %.loc24_33.7 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc24_33.9: init %A = class_init (%.loc24_33.5, %.loc24_33.8), %.loc24_33.3 [template = constants.%A.val]
// CHECK:STDOUT:   %.loc24_33.10: ref %A = temporary %.loc24_33.3, %.loc24_33.9
// CHECK:STDOUT:   %.loc24_35: ref %A = converted %.loc24_33.1, %.loc24_33.10
// CHECK:STDOUT:   %B.ref.loc24: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc24_41.1: ref %B = as_compatible %.loc24_35
// CHECK:STDOUT:   %.loc24_41.2: ref %B = converted %.loc24_35, %.loc24_41.1
// CHECK:STDOUT:   %.loc24_41.3: %B = bind_value %.loc24_41.2
// CHECK:STDOUT:   assign file.%b_init.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_init_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %Noncopyable) [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %tuple.type.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Noncopyable = %Noncopyable.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Noncopyable.decl: type = class_decl @Noncopyable [template = constants.%Noncopyable] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %A = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param runtime_param0
// CHECK:STDOUT:     %A.ref.loc12: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [template = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc9_26: %tuple.type.1 = tuple_literal (%i32, %Noncopyable.ref)
// CHECK:STDOUT:   %.loc9_27: type = converted %.loc9_26, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   adapt_decl %.loc9_27 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %tuple.type.2 [template = constants.%complete_type.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref.loc13: %A = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref.loc13: type = name_ref Noncopyable, file.%Noncopyable.decl [template = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc13_43.1: %tuple.type.1 = tuple_literal (%i32.loc13, %Noncopyable.ref.loc13)
// CHECK:STDOUT:   %.loc13_43.2: type = converted %.loc13_43.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc13_23.1: %tuple.type.2 = as_compatible %a.ref.loc13
// CHECK:STDOUT:   %.loc13_23.2: %tuple.type.2 = converted %a.ref.loc13, %.loc13_23.1
// CHECK:STDOUT:   %A.ref.loc13: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc13_46.1: %A = as_compatible %.loc13_23.2
// CHECK:STDOUT:   %.loc13_46.2: %A = converted %.loc13_23.2, %.loc13_46.1
// CHECK:STDOUT:   %a_value: %A = bind_name a_value, %.loc13_46.2
// CHECK:STDOUT:   %a_init.var: ref %A = var a_init
// CHECK:STDOUT:   %a_init: ref %A = bind_name a_init, %a_init.var
// CHECK:STDOUT:   %a.ref.loc24: %A = name_ref a, %a
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref.loc24: type = name_ref Noncopyable, file.%Noncopyable.decl [template = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc24_42.1: %tuple.type.1 = tuple_literal (%i32.loc24, %Noncopyable.ref.loc24)
// CHECK:STDOUT:   %.loc24_42.2: type = converted %.loc24_42.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc24_22.1: %tuple.type.2 = as_compatible %a.ref.loc24
// CHECK:STDOUT:   %.loc24_22.2: %tuple.type.2 = converted %a.ref.loc24, %.loc24_22.1
// CHECK:STDOUT:   %A.ref.loc24: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc24_45.1: %A = as_compatible %.loc24_22.2
// CHECK:STDOUT:   %.loc24_45.2: %A = converted %.loc24_22.2, %.loc24_45.1
// CHECK:STDOUT:   %.loc24_49.1: %tuple.type.2 = as_compatible %.loc24_45.2
// CHECK:STDOUT:   %tuple.elem0.loc24_49.1: %i32 = tuple_access %.loc24_49.1, element0
// CHECK:STDOUT:   %.loc24_49.2: ref %tuple.type.2 = as_compatible %a_init.var
// CHECK:STDOUT:   %tuple.elem0.loc24_49.2: ref %i32 = tuple_access %.loc24_49.2, element0
// CHECK:STDOUT:   %.loc24_49.3: init %i32 = initialize_from %tuple.elem0.loc24_49.1 to %tuple.elem0.loc24_49.2
// CHECK:STDOUT:   %tuple.elem1: %Noncopyable = tuple_access %.loc24_49.1, element1
// CHECK:STDOUT:   assign %a_init.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_init_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.1: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.x.1 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.x.2: type = struct_type {.x: Core.IntLiteral} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.5
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
// CHECK:STDOUT:   %b.var: ref %B = var b
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc5: %A.elem = field_decl x, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.1 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %.loc5
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   adapt_decl %A.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.1 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc21_19: %struct_type.x.2 = struct_literal (%int_1)
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21_21: %B = converted %.loc21_19, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
