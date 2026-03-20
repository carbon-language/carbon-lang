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
// CHECK:STDOUT:   %pattern_type.1f4: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %ptr.27c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.191: type = pattern_type %ptr.27c [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %B = var file.%a_ref.var_patt [concrete]
// CHECK:STDOUT:   %addr: %ptr.27c = addr_of %a_ref.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_val.patt: %pattern_type.1f4 = value_binding_pattern b_val [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.ref.loc73: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_val: %B = value_binding b_val, @__global_init.%.loc76.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_ptr.patt: %pattern_type.191 = value_binding_pattern b_ptr [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc83: type = splice_block %ptr [concrete = constants.%ptr.27c] {
// CHECK:STDOUT:     %B.ref.loc82: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr: type = ptr_type %B.ref.loc82 [concrete = constants.%ptr.27c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_ptr: %ptr.27c = value_binding b_ptr, @__global_init.%addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_factory.patt: %pattern_type.1f4 = ref_binding_pattern b_factory [concrete]
// CHECK:STDOUT:     %b_factory.var_patt: %pattern_type.1f4 = var_pattern %b_factory.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b_factory.var: ref %B = var %b_factory.var_patt [concrete]
// CHECK:STDOUT:   %B.ref.loc95: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b_factory: ref %B = ref_binding b_factory, %b_factory.var [concrete = %b_factory.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a_val.ref: %A = name_ref a_val, file.%a_val
// CHECK:STDOUT:   %B.ref.loc77: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc76.1: %B = as_compatible %a_val.ref
// CHECK:STDOUT:   %.loc76.2: %B = converted %a_val.ref, %.loc76.1
// CHECK:STDOUT:   %a_ref.ref.loc87: ref %A = name_ref a_ref, file.%a_ref [concrete = file.%a_ref.var]
// CHECK:STDOUT:   %B.ref.loc89: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc88.1: ref %B = as_compatible %a_ref.ref.loc87 [concrete = constants.%a_ref.var]
// CHECK:STDOUT:   %.loc88.2: ref %B = converted %a_ref.ref.loc87, %.loc88.1 [concrete = constants.%a_ref.var]
// CHECK:STDOUT:   %addr: %ptr.27c = addr_of %.loc88.2 [concrete = constants.%addr]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Make.ref: %A.Make.type = name_ref Make, @A.%A.Make.decl [concrete = constants.%A.Make]
// CHECK:STDOUT:   %.loc92: ref %B = splice_block file.%b_factory.var [concrete = file.%b_factory.var] {}
// CHECK:STDOUT:   %A.Make.call: init %A to %.loc92 = call %Make.ref()
// CHECK:STDOUT:   %B.ref.loc103: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc102.1: init %B = as_compatible %A.Make.call
// CHECK:STDOUT:   %.loc102.2: init %B = converted %A.Make.call, %.loc102.1
// CHECK:STDOUT:   assign file.%b_factory.var, %.loc102.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e5b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %As.WithSelf.Convert.type.e5b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.360: %A = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc197, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1ab = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref: type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: %A = value_binding a, @__global_init.%.loc21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %n: %i32 = value_binding n, @__global_init.%.loc30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc18.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc18.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18.2: %i32 = converted %int_1, %.loc18.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc21.1: %A = as_compatible %.loc18.2 [concrete = constants.%int_1.360]
// CHECK:STDOUT:   %.loc21.2: %A = converted %.loc18.2, %.loc21.1 [concrete = constants.%int_1.360]
// CHECK:STDOUT:   %a.ref: %A = name_ref a, file.%a
// CHECK:STDOUT:   %i32.loc31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc30.1: %i32 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc30.2: %i32 = converted %a.ref, %.loc30.1
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
// CHECK:STDOUT:   %d: %D = value_binding d, @__global_init.%.loc40.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc39: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc40.1: %D = as_compatible %empty_struct [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc40.2: %D = converted %.loc39, %.loc40.1 [concrete = constants.%D.val]
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
// CHECK:STDOUT:   %pattern_type.1f4: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.y.4cf: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x.y.4cf = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_value.patt: %pattern_type.1f4 = value_binding_pattern b_value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc45.1: ref %B = temporary @__global_init.%.loc41.3, @__global_init.%.loc45.2
// CHECK:STDOUT:   %.loc45.2: %B = acquire_value %.loc45.1
// CHECK:STDOUT:   %b_value: %B = value_binding b_value, %.loc45.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc41.1: %struct_type.x.y.4cf = struct_literal (%int_1, %int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %impl.elem0.loc41.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc41.1: <bound method> = bound_method %int_1, %impl.elem0.loc41.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc41.1: <specific function> = specific_function %impl.elem0.loc41.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41.2: <bound method> = bound_method %int_1, %specific_fn.loc41.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41.1: init %i32 = call %bound_method.loc41.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc41.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc41.3: ref %A = temporary_storage
// CHECK:STDOUT:   %.loc41.4: ref %i32 = class_element_access %.loc41.3, element0
// CHECK:STDOUT:   %.loc41.5: init %i32 to %.loc41.4 = in_place_init %.loc41.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc41.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc41.3: <bound method> = bound_method %int_2, %impl.elem0.loc41.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc41.2: <specific function> = specific_function %impl.elem0.loc41.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41.4: <bound method> = bound_method %int_2, %specific_fn.loc41.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41.2: init %i32 = call %bound_method.loc41.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc41.6: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc41.7: ref %i32 = class_element_access %.loc41.3, element1
// CHECK:STDOUT:   %.loc41.8: init %i32 to %.loc41.7 = in_place_init %.loc41.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc41.9: init %A to %.loc41.3 = class_init (%.loc41.5, %.loc41.8) [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc42: init %A = converted %.loc41.1, %.loc41.9 [concrete = constants.%A.val]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc45.1: init %B = as_compatible %.loc42 [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc45.2: init %B = converted %.loc42, %.loc45.1 [concrete = constants.%B.val]
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
// CHECK:STDOUT:   %tuple.type.037: type = tuple_type (%empty_struct_type, %Noncopyable) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_value.patt: %pattern_type = value_binding_pattern a_value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %.loc39: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %.loc42.1: %tuple.type.c8c = tuple_literal (%.loc39, %Noncopyable.ref) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc42.2: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc42.3: type = converted %.loc42.1, constants.%tuple.type.037 [concrete = constants.%tuple.type.037]
// CHECK:STDOUT:   %.loc36.1: %tuple.type.037 = as_compatible %a.ref
// CHECK:STDOUT:   %.loc36.2: %tuple.type.037 = converted %a.ref, %.loc36.1
// CHECK:STDOUT:   %A.ref.loc45: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc44.1: %A = as_compatible %.loc36.2
// CHECK:STDOUT:   %.loc44.2: %A = converted %.loc36.2, %.loc44.1
// CHECK:STDOUT:   %A.ref.loc32: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a_value: %A = value_binding a_value, %.loc44.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
