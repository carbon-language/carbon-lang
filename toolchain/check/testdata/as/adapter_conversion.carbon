// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
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
//@dump-sem-ir-begin
let b_val: B = a_val as B;
let b_ptr: B* = &(a_ref as B);

var b_factory: B = A.Make() as B;
//@dump-sem-ir-end

// --- adapt_i32.carbon

library "[[@TEST_NAME]]";

class A {
  adapt i32;
}

//@dump-sem-ir-begin
let a: A = (1 as i32) as A;
let n: i32 = a as i32;
//@dump-sem-ir-end

// --- multi_level_adapt.carbon

library "[[@TEST_NAME]]";

class A { adapt {}; }
class B { adapt A; }
class C { adapt B; }
class D { adapt C; }

//@dump-sem-ir-begin
let d: D = {} as D;
//@dump-sem-ir-end


// --- init_class_value.carbon

library "[[@TEST_NAME]]";

class A {
  var x: i32;
  var y: i32;
}

class B {
  adapt A;
}

//@dump-sem-ir-begin
let b_value: B = ({.x = 1, .y = 2} as A) as B;
//@dump-sem-ir-end

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
  adapt ({}, Noncopyable);
}

fn F(a: A) {
  //@dump-sem-ir-begin
  let a_value: A = (a as ({}, Noncopyable)) as A;
  //@dump-sem-ir-end
}

// --- fail_init_tuple_variable.carbon

library "[[@TEST_NAME]]";

class Noncopyable {
  // TODO: Ensure this remains non-copyable once we have rules for class copyability.
}

class A {
  adapt ({}, Noncopyable);
}

fn F(a: A) {
  // TODO: Here, we treat `a as (i32, Noncopyable)` as a value expression, not an
  // initializing expression, so `(...) as A` is a value expression too, requiring
  // a copy to perform initialization. It's not clear whether that is the right
  // behavior.

  // CHECK:STDERR: fail_init_tuple_variable.carbon:[[@LINE+7]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var a_init: A = (a as ({}, Noncopyable)) as A;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_init_tuple_variable.carbon:[[@LINE+4]]:19: note: in copy of `A` [InCopy]
  // CHECK:STDERR:   var a_init: A = (a as ({}, Noncopyable)) as A;
  // CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a_init: A = (a as ({}, Noncopyable)) as A;
}

// --- fail_adapt_init_from_struct.carbon

library "[[@TEST_NAME]]";

class A {
  var x: ();
}

class B {
  adapt A;
}

// We do not try to implicitly convert from the first operand of `as` to the
// adapted type of the second operand.

// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+7]]:12: error: cannot convert expression of type `{.x: ()}` to `B` with `as` [ConversionFailure]
// CHECK:STDERR: var b: B = {.x = ()} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+4]]:12: note: type `{.x: ()}` does not implement interface `Core.As(B)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: B = {.x = ()} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~~
// CHECK:STDERR:
var b: B = {.x = ()} as B;

// CHECK:STDOUT: --- adapt_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %B = var file.%a_ref.var_patt [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %a_ref.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_val.patt: %pattern_type.049 = binding_pattern b_val [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.ref.loc22: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_val: ref %B = bind_name b_val, @__global_init.%.loc22_22.2 [concrete = constants.%a_ref.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ptr.patt: %pattern_type.960 = binding_pattern b_ptr [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23: type = splice_block %ptr [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:     %B.ref.loc23: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr: type = ptr_type %B.ref.loc23 [concrete = constants.%ptr.e79]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr: %ptr.e79 = bind_name b_ptr, @__global_init.%addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_factory.patt: %pattern_type.049 = binding_pattern b_factory [concrete]
// CHECK:STDOUT:     %b_factory.var_patt: %pattern_type.049 = var_pattern %b_factory.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_factory.var: ref %B = var %b_factory.var_patt [concrete]
// CHECK:STDOUT:   %B.ref.loc25: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_factory: ref %B = bind_name b_factory, %b_factory.var [concrete = %b_factory.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a_val.ref: ref %A = name_ref a_val, file.%a_val [concrete = file.%a_ref.var]
// CHECK:STDOUT:   %B.ref.loc22: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc22_22.1: ref %B = as_compatible %a_val.ref [concrete = constants.%a_ref.var]
// CHECK:STDOUT:   %.loc22_22.2: ref %B = converted %a_val.ref, %.loc22_22.1 [concrete = constants.%a_ref.var]
// CHECK:STDOUT:   %a_ref.ref.loc23: ref %A = name_ref a_ref, file.%a_ref [concrete = file.%a_ref.var]
// CHECK:STDOUT:   %B.ref.loc23: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc23_25.1: ref %B = as_compatible %a_ref.ref.loc23 [concrete = constants.%a_ref.var]
// CHECK:STDOUT:   %.loc23_25.2: ref %B = converted %a_ref.ref.loc23, %.loc23_25.1 [concrete = constants.%a_ref.var]
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc23_25.2 [concrete = constants.%addr]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, @A.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc25_1: ref %B = splice_block file.%b_factory.var [concrete = file.%b_factory.var] {}
// CHECK:STDOUT:   %Make.call: init %A = call %Make.ref() to %.loc25_1
// CHECK:STDOUT:   %B.ref.loc25: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc25_29.1: init %B = as_compatible %Make.call
// CHECK:STDOUT:   %.loc25_29.2: init %B = converted %Make.call, %.loc25_29.1
// CHECK:STDOUT:   assign file.%b_factory.var, %.loc25_29.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.1, @As(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.5, @impl.686(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.6b4: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%As.impl_witness.6b4) [concrete]
// CHECK:STDOUT:   %.982: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.197 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.e78: %A = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.78a: @impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/types/int, loc28_39, loaded [symbolic = @impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @impl.686 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref: type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: %A = bind_name a, @__global_init.%.loc9_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: %i32 = bind_name n, @__global_init.%.loc10_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc9_15.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_15.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc9_15.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_15.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_15.2: %i32 = converted %int_1, %.loc9_15.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc9_23.1: %A = as_compatible %.loc9_15.2 [concrete = constants.%int_1.e78]
// CHECK:STDOUT:   %.loc9_23.2: %A = converted %.loc9_15.2, %.loc9_23.1 [concrete = constants.%int_1.e78]
// CHECK:STDOUT:   %a.ref: %A = name_ref a, file.%a
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc10_16.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc10_16.2: %i32 = converted %a.ref, %.loc10_16.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multi_level_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type = binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.ref: type = name_ref D, %D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %d: %D = bind_name d, @__global_init.%.loc10_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_15.1: %D = as_compatible %empty_struct [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc10_15.2: %D = converted %.loc10_13, %.loc10_15.1 [concrete = constants.%D.val]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_class_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.y.4cf: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_value.patt: %pattern_type.049 = binding_pattern b_value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_value: ref %B = bind_name b_value, @__global_init.%.loc14_42.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc14_34.1: %struct_type.x.y.4cf = struct_literal (%int_1, %int_2)
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %impl.elem0.loc14_34.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc14_34.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc14_34.1: <specific function> = specific_function %impl.elem0.loc14_34.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_1, %specific_fn.loc14_34.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc14_34.1: init %i32 = call %bound_method.loc14_34.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_34.2: init %i32 = converted %int_1, %int.convert_checked.loc14_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_34.3: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc14_34.4: ref %i32 = class_element_access %.loc14_34.3, element0
// CHECK:STDOUT:   %.loc14_34.5: init %i32 = initialize_from %.loc14_34.2 to %.loc14_34.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc14_34.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_34.3: <bound method> = bound_method %int_2, %impl.elem0.loc14_34.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc14_34.2: <specific function> = specific_function %impl.elem0.loc14_34.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.4: <bound method> = bound_method %int_2, %specific_fn.loc14_34.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc14_34.2: init %i32 = call %bound_method.loc14_34.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_34.6: init %i32 = converted %int_2, %int.convert_checked.loc14_34.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_34.7: ref %i32 = class_element_access %.loc14_34.3, element1
// CHECK:STDOUT:   %.loc14_34.8: init %i32 = initialize_from %.loc14_34.6 to %.loc14_34.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_34.9: init %A = class_init (%.loc14_34.5, %.loc14_34.8), %.loc14_34.3 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc14_34.10: ref %A = temporary %.loc14_34.3, %.loc14_34.9
// CHECK:STDOUT:   %.loc14_36: ref %A = converted %.loc14_34.1, %.loc14_34.10
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc14_42.1: ref %B = as_compatible %.loc14_36
// CHECK:STDOUT:   %.loc14_42.2: ref %B = converted %.loc14_36, %.loc14_42.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_tuple_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %tuple.type.c8c: type = tuple_type (%empty_struct_type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.a10: type = tuple_type (%empty_struct_type, %Noncopyable) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_value.patt: %pattern_type = binding_pattern a_value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %.loc14_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc14_42.1: %tuple.type.c8c = tuple_literal (%.loc14_28, %Noncopyable.ref)
// CHECK:STDOUT:   %.loc14_42.2: type = converted %.loc14_28, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc14_42.3: type = converted %.loc14_42.1, constants.%tuple.type.a10 [concrete = constants.%tuple.type.a10]
// CHECK:STDOUT:   %.loc14_23.1: %tuple.type.a10 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc14_23.2: %tuple.type.a10 = converted %a.ref, %.loc14_23.1
// CHECK:STDOUT:   %A.ref.loc14_48: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc14_45.1: %A = as_compatible %.loc14_23.2
// CHECK:STDOUT:   %.loc14_45.2: %A = converted %.loc14_23.2, %.loc14_45.1
// CHECK:STDOUT:   %A.ref.loc14_16: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a_value: %A = bind_name a_value, %.loc14_45.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
