// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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

// --- init_class_variable.carbon

library "[[@TEST_NAME]]";

class A {
  var x: i32;
  var y: i32;
}

class B {
  adapt A;
}

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
  let unused a_value: A = (a as ({}, Noncopyable)) as A;
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

  // CHECK:STDERR: fail_init_tuple_variable.carbon:[[@LINE+10]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var unused a_init: A = (a as ({}, Noncopyable)) as A;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_init_tuple_variable.carbon:[[@LINE+7]]:3: note: type `Noncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused a_init: A = (a as ({}, Noncopyable)) as A;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_init_tuple_variable.carbon:[[@LINE+4]]:26: note: in copy of `A` [InCopy]
  // CHECK:STDERR:   var unused a_init: A = (a as ({}, Noncopyable)) as A;
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused a_init: A = (a as ({}, Noncopyable)) as A;
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
// CHECK:STDERR: fail_adapt_init_from_struct.carbon:[[@LINE+4]]:12: note: type `{.x: ()}` does not implement interface `Core.As(B)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var b: B = {.x = ()} as B;
// CHECK:STDERR:            ^~~~~~~~~~~~~~
// CHECK:STDERR:
var b: B = {.x = ()} as B;

// CHECK:STDOUT: --- adapt_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.Make.type: type = fn_type @A.Make [concrete]
// CHECK:STDOUT:   %A.Make: %A.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.3c8: type = pattern_type %ptr.432 [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %B = var file.%a_ref.var_patt [concrete]
// CHECK:STDOUT:   %addr: %ptr.432 = addr_of %a_ref.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_val.patt: %pattern_type.e39 = value_binding_pattern b_val [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.ref.loc22: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_val: %B = value_binding b_val, @__global_init.%.loc22_22.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ptr.patt: %pattern_type.3c8 = value_binding_pattern b_ptr [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23: type = splice_block %ptr [concrete = constants.%ptr.432] {
// CHECK:STDOUT:     %B.ref.loc23: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr: type = ptr_type %B.ref.loc23 [concrete = constants.%ptr.432]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr: %ptr.432 = value_binding b_ptr, @__global_init.%addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_factory.patt: %pattern_type.e39 = ref_binding_pattern b_factory [concrete]
// CHECK:STDOUT:     %b_factory.var_patt: %pattern_type.e39 = var_pattern %b_factory.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_factory.var: ref %B = var %b_factory.var_patt [concrete]
// CHECK:STDOUT:   %B.ref.loc25: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_factory: ref %B = ref_binding b_factory, %b_factory.var [concrete = %b_factory.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a_val.ref: %A = name_ref a_val, file.%a_val
// CHECK:STDOUT:   %B.ref.loc22: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc22_22.1: %B = as_compatible %a_val.ref
// CHECK:STDOUT:   %.loc22_22.2: %B = converted %a_val.ref, %.loc22_22.1
// CHECK:STDOUT:   %a_ref.ref.loc23: ref %A = name_ref a_ref, file.%a_ref [concrete = file.%a_ref.var]
// CHECK:STDOUT:   %B.ref.loc23: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc23_25.1: ref %B = as_compatible %a_ref.ref.loc23 [concrete = constants.%a_ref.var]
// CHECK:STDOUT:   %.loc23_25.2: ref %B = converted %a_ref.ref.loc23, %.loc23_25.1 [concrete = constants.%a_ref.var]
// CHECK:STDOUT:   %addr: %ptr.432 = addr_of %.loc23_25.2 [concrete = constants.%addr]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Make.ref: %A.Make.type = name_ref Make, @A.%A.Make.decl [concrete = constants.%A.Make]
// CHECK:STDOUT:   %.loc25_1: ref %B = splice_block file.%b_factory.var [concrete = file.%b_factory.var] {}
// CHECK:STDOUT:   %A.Make.call: init %A to %.loc25_1 = call %Make.ref()
// CHECK:STDOUT:   %B.ref.loc25: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc25_29.1: init %B = as_compatible %A.Make.call
// CHECK:STDOUT:   %.loc25_29.2: init %B = converted %A.Make.call, %.loc25_29.1
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
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.155: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.03e: <witness> = impl_witness imports.%As.impl_witness_table.1c5, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e79: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8a4: %Core.IntLiteral.as.As.impl.Convert.type.e79 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.155 = facet_value Core.IntLiteral, (%As.impl_witness.03e) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.8ea: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.fbc: type = fn_type_with_self_type %As.WithSelf.Convert.type.8ea, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.8a4 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.8a4, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.403: %A = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.1c5 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref: type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: %A = value_binding a, @__global_init.%.loc9_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.6b6 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %n: %i32 = value_binding n, @__global_init.%.loc10_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.fbc = impl_witness_access constants.%As.impl_witness.03e, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.8a4]
// CHECK:STDOUT:   %bound_method.loc9_15.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_15.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc9_15.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc9_15.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc9_15.2: %i32 = converted %int_1, %.loc9_15.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc9_23.1: %A = as_compatible %.loc9_15.2 [concrete = constants.%int_1.403]
// CHECK:STDOUT:   %.loc9_23.2: %A = converted %.loc9_15.2, %.loc9_23.1 [concrete = constants.%int_1.403]
// CHECK:STDOUT:   %a.ref: %A = name_ref a, file.%a
// CHECK:STDOUT:   %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc10_16.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc10_16.2: %i32 = converted %a.ref, %.loc10_16.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- multi_level_adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.ref: type = name_ref D, %D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %d: %D = value_binding d, @__global_init.%.loc10_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.y.4cf: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x.y.4cf = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%int_1.0c6, %int_2.295) [concrete]
// CHECK:STDOUT:   %.07b: ref %B = temporary invalid, %B.val [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_value.patt: %pattern_type.e39 = value_binding_pattern b_value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc14_42.1: ref %B = temporary @__global_init.%.loc14_34.3, @__global_init.%.loc14_42.2 [concrete = constants.%.07b]
// CHECK:STDOUT:   %.loc14_42.2: %B = acquire_value %.loc14_42.1 [concrete = constants.%B.val]
// CHECK:STDOUT:   %b_value: %B = value_binding b_value, %.loc14_42.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc14_34.1: %struct_type.x.y.4cf = struct_literal (%int_1, %int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %impl.elem0.loc14_34.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc14_34.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc14_34.1: <specific function> = specific_function %impl.elem0.loc14_34.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_1, %specific_fn.loc14_34.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_34.1: init %i32 = call %bound_method.loc14_34.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc14_34.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_34.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc14_34.3: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc14_34.4: ref %i32 = class_element_access %.loc14_34.3, element0
// CHECK:STDOUT:   %.loc14_34.5: init %i32 to %.loc14_34.4 = in_place_init %.loc14_34.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc14_34.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc14_34.3: <bound method> = bound_method %int_2, %impl.elem0.loc14_34.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc14_34.2: <specific function> = specific_function %impl.elem0.loc14_34.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.4: <bound method> = bound_method %int_2, %specific_fn.loc14_34.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_34.2: init %i32 = call %bound_method.loc14_34.4(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc14_34.6: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_34.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc14_34.7: ref %i32 = class_element_access %.loc14_34.3, element1
// CHECK:STDOUT:   %.loc14_34.8: init %i32 to %.loc14_34.7 = in_place_init %.loc14_34.6 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc14_34.9: init %A to %.loc14_34.3 = class_init (%.loc14_34.5, %.loc14_34.8) [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc14_36: init %A = converted %.loc14_34.1, %.loc14_34.9 [concrete = constants.%A.val]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc14_42.1: init %B = as_compatible %.loc14_36 [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc14_42.2: init %B = converted %.loc14_36, %.loc14_42.1 [concrete = constants.%B.val]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_tuple_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.c8c: type = tuple_type (%empty_struct_type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.c8c = tuple_value (%empty_struct, %Noncopyable) [concrete]
// CHECK:STDOUT:   %tuple.type.c68: type = tuple_type (%empty_struct_type, %Noncopyable) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_value.patt: %pattern_type = value_binding_pattern a_value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %.loc14_35: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc14_49.1: %tuple.type.c8c = tuple_literal (%.loc14_35, %Noncopyable.ref) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc14_49.2: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc14_49.3: type = converted %.loc14_49.1, constants.%tuple.type.c68 [concrete = constants.%tuple.type.c68]
// CHECK:STDOUT:   %.loc14_30.1: %tuple.type.c68 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc14_30.2: %tuple.type.c68 = converted %a.ref, %.loc14_30.1
// CHECK:STDOUT:   %A.ref.loc14_55: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc14_52.1: %A = as_compatible %.loc14_30.2
// CHECK:STDOUT:   %.loc14_52.2: %A = converted %.loc14_30.2, %.loc14_52.1
// CHECK:STDOUT:   %A.ref.loc14_23: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a_value: %A = value_binding a_value, %.loc14_52.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
