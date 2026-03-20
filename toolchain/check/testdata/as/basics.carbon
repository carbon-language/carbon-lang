// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/basics.carbon

// --- simple_as.carbon

library "[[@TEST_NAME]]";

fn Main() -> {.x: (), .y: ()} {
//@dump-sem-ir-begin
  return {.y = (), .x = ()} as {.x: (), .y: ()};
//@dump-sem-ir-end
}

// --- as_type.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let t: type = ({}, {}) as type;
//@dump-sem-ir-end

// --- as_tuple.carbon

library "[[@TEST_NAME]]";

class X {
  // ...
}

fn Make() -> X;

fn Let() {
  // This should create value bindings for both tuple elements.
  //@dump-sem-ir-begin
  let unused a: (X, X) = (Make(), Make()) as (X, X);
  //@dump-sem-ir-end
}

fn Var() {
  // This should initialize both tuple elements in place.
  //@dump-sem-ir-begin
  var unused b: (X, X) = (Make(), Make()) as (X, X);
  //@dump-sem-ir-end
}

// --- identity.carbon

library "[[@TEST_NAME]]";

class X {
  // ...
}

fn Value(n: X) {
  //@dump-sem-ir-begin
  let unused m: X = n as X;
  //@dump-sem-ir-end
}

fn Reference(p: X*) {
  //@dump-sem-ir-begin
  let unused q: X* = &(*p as X);
  //@dump-sem-ir-end
}

fn Make() -> X;

fn Initializing() {
  //@dump-sem-ir-begin
  var unused x: X = (Make() as X);
  //@dump-sem-ir-end
}

// --- overloaded.carbon

class X {
  var x: ();
}

class Y {
  var y: ();
}

impl Y as Core.As(X) {
  fn Convert[self: Y]() -> X { return {.x = self.y}; }
}

impl X as Core.As(Y) {
  fn Convert[self: X]() -> Y { return {.y = self.x}; }
}

//@dump-sem-ir-begin
let n: Y = ({.x = ()} as X) as Y;
//@dump-sem-ir-end

// --- fail_no_conversion.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+7]]:13: error: cannot convert expression of type `Core.IntLiteral` to `{}` with `as` [ConversionFailure]
// CHECK:STDERR: let a: {} = 1 as {};
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+4]]:13: note: type `Core.IntLiteral` does not implement interface `Core.As({})` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: let a: {} = 1 as {};
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
let a: {} = 1 as {};

// --- fail_not_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_not_type.carbon:[[@LINE+7]]:32: error: cannot implicitly convert non-type value of type `{.x: ()}` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: let n: {.x: ()} = {.x = ()} as {.x = ()};
// CHECK:STDERR:                                ^~~~~~~~~
// CHECK:STDERR: fail_not_type.carbon:[[@LINE+4]]:32: note: type `{.x: ()}` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: let n: {.x: ()} = {.x = ()} as {.x = ()};
// CHECK:STDERR:                                ^~~~~~~~~
// CHECK:STDERR:
let n: {.x: ()} = {.x = ()} as {.x = ()};

// CHECK:STDOUT: --- simple_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: %empty_tuple.type, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct_type.y.x: type = struct_type {.y: %empty_tuple.type, .x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct.f76: %struct_type.y.x = struct_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %struct.005: %struct_type.x.y = struct_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> out %return.param: %struct_type.x.y {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc29: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc35: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36.1: %struct_type.y.x = struct_literal (%.loc29, %.loc35) [concrete = constants.%struct.f76]
// CHECK:STDOUT:   %.loc43.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc43.2: type = converted %.loc43.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc49.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc49.2: type = converted %.loc49.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.x.y.loc50: type = struct_type {.x: %empty_tuple.type, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y]
// CHECK:STDOUT:   %empty_tuple.loc35: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36.2: %empty_tuple.type = converted %.loc35, %empty_tuple.loc35 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc29: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36.3: %empty_tuple.type = converted %.loc29, %empty_tuple.loc29 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%.loc36.2, %.loc36.3) [concrete = constants.%struct.005]
// CHECK:STDOUT:   %.loc37.1: %struct_type.x.y = converted %.loc36.1, %struct [concrete = constants.%struct.005]
// CHECK:STDOUT:   %.loc37.2: %empty_tuple.type = struct_access %.loc37.1, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc37.3: ref %empty_tuple.type = struct_access %return.param, element0
// CHECK:STDOUT:   %.loc37.4: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc37.5: init %empty_tuple.type = converted %.loc37.2, %.loc37.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc37.6: %empty_tuple.type = struct_access %.loc37.1, element1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc37.7: ref %empty_tuple.type = struct_access %return.param, element1
// CHECK:STDOUT:   %.loc37.8: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc37.9: init %empty_tuple.type = converted %.loc37.6, %.loc37.8 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc37.10: init %struct_type.x.y to %return.param = struct_init (%.loc37.5, %.loc37.9) [concrete = constants.%struct.005]
// CHECK:STDOUT:   %.loc51: init %struct_type.x.y = converted %.loc37.1, %.loc37.10 [concrete = constants.%struct.005]
// CHECK:STDOUT:   return %.loc51 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.b6b = tuple_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type = value_binding_pattern t [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %t: type = value_binding t, @__global_init.%.loc16.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15: %tuple.type.b6b = tuple_literal (%.loc11, %.loc14) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc17: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc16.1: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc16.2: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc16.3: type = converted %.loc15, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- as_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%X, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.2de: type = tuple_type (%X, %X) [concrete]
// CHECK:STDOUT:   %pattern_type.e9d: type = pattern_type %tuple.type.2de [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc37 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc52 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.e9d = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.ref.loc31: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc33.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc33: init %X to %.loc33.1 = call %Make.ref.loc31()
// CHECK:STDOUT:   %Make.ref.loc35: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc37.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc37: init %X to %.loc37.1 = call %Make.ref.loc35()
// CHECK:STDOUT:   %.loc38.1: %tuple.type.2de = tuple_literal (%Make.call.loc33, %Make.call.loc37)
// CHECK:STDOUT:   %X.ref.loc41: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc44.1: %tuple.type.24b = tuple_literal (%X.ref.loc41, %X.ref.loc43) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc44.2: type = converted %.loc44.1, constants.%tuple.type.2de [concrete = constants.%tuple.type.2de]
// CHECK:STDOUT:   %.loc28.1: type = splice_block %.loc28.3 [concrete = constants.%tuple.type.2de] {
// CHECK:STDOUT:     %X.ref.loc25: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc27: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc28.2: %tuple.type.24b = tuple_literal (%X.ref.loc25, %X.ref.loc27) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc28.3: type = converted %.loc28.2, constants.%tuple.type.2de [concrete = constants.%tuple.type.2de]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc33.2: ref %X = temporary %.loc33.1, %Make.call.loc33
// CHECK:STDOUT:   %.loc33.3: %X = acquire_value %.loc33.2
// CHECK:STDOUT:   %.loc37.2: ref %X = temporary %.loc37.1, %Make.call.loc37
// CHECK:STDOUT:   %.loc37.3: %X = acquire_value %.loc37.2
// CHECK:STDOUT:   %tuple: %tuple.type.2de = tuple_value (%.loc33.3, %.loc37.3)
// CHECK:STDOUT:   %.loc38.2: %tuple.type.2de = converted %.loc38.1, %tuple
// CHECK:STDOUT:   %a: %tuple.type.2de = value_binding a, %.loc38.2
// CHECK:STDOUT:   %Destroy.Op.bound.loc37: <bound method> = bound_method %.loc37.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc37: init %empty_tuple.type = call %Destroy.Op.bound.loc37(%.loc37.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc33: <bound method> = bound_method %.loc33.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc33: init %empty_tuple.type = call %Destroy.Op.bound.loc33(%.loc33.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc37(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.e9d = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.e9d = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.2de = var %b.var_patt
// CHECK:STDOUT:   %Make.ref.loc63: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem0: ref %X = tuple_access %b.var, element0
// CHECK:STDOUT:   %Make.call.loc65: init %X to %tuple.elem0 = call %Make.ref.loc63()
// CHECK:STDOUT:   %Make.ref.loc67: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem1: ref %X = tuple_access %b.var, element1
// CHECK:STDOUT:   %Make.call.loc69: init %X to %tuple.elem1 = call %Make.ref.loc67()
// CHECK:STDOUT:   %.loc70.1: %tuple.type.2de = tuple_literal (%Make.call.loc65, %Make.call.loc69)
// CHECK:STDOUT:   %X.ref.loc73: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc75: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc76.1: %tuple.type.24b = tuple_literal (%X.ref.loc73, %X.ref.loc75) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc76.2: type = converted %.loc76.1, constants.%tuple.type.2de [concrete = constants.%tuple.type.2de]
// CHECK:STDOUT:   %.loc70.2: init %tuple.type.2de to %b.var = tuple_init (%Make.call.loc65, %Make.call.loc69)
// CHECK:STDOUT:   %.loc52: init %tuple.type.2de = converted %.loc70.1, %.loc70.2
// CHECK:STDOUT:   assign %b.var, %.loc52
// CHECK:STDOUT:   %.loc60.1: type = splice_block %.loc60.3 [concrete = constants.%tuple.type.2de] {
// CHECK:STDOUT:     %X.ref.loc57: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc59: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc60.2: %tuple.type.24b = tuple_literal (%X.ref.loc57, %X.ref.loc59) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc60.3: type = converted %.loc60.2, constants.%tuple.type.2de [concrete = constants.%tuple.type.2de]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.2de = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %b.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%b.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc52(%self.param: ref %tuple.type.2de) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %pattern_type.05f: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.2a9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.37f: type = pattern_type %ptr.2a9 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Value(%n.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %pattern_type.05f = value_binding_pattern m [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.ref: %X = name_ref n, %n
// CHECK:STDOUT:   %X.ref.loc24: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc20: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %m: %X = value_binding m, %n.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Reference(%p.param: %ptr.2a9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %pattern_type.37f = value_binding_pattern q [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.ref: %ptr.2a9 = name_ref p, %p
// CHECK:STDOUT:   %.loc45: ref %X = deref %p.ref
// CHECK:STDOUT:   %X.ref.loc48: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %addr: %ptr.2a9 = addr_of %.loc45
// CHECK:STDOUT:   %.loc41: type = splice_block %ptr.loc41 [concrete = constants.%ptr.2a9] {
// CHECK:STDOUT:     %X.ref.loc40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc41: type = ptr_type %X.ref.loc40 [concrete = constants.%ptr.2a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: %ptr.2a9 = value_binding q, %addr
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Initializing() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.05f = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.05f = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %X = var %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc64: ref %X = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %X to %.loc64 = call %Make.ref()
// CHECK:STDOUT:   %X.ref.loc75: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %X.ref.loc68: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %x: ref %X = ref_binding x, %x.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %x.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %pattern_type.77c: type = pattern_type %Y [concrete]
// CHECK:STDOUT:   %struct.948: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %As.type.d1e: type = facet_type <@As, @As(%Y)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.e92: <witness> = impl_witness @X.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %X.as.As.impl.Convert.type: type = fn_type @X.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %X.as.As.impl.Convert: %X.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.c03: %As.type.d1e = facet_value %X, (%As.impl_witness.e92) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.edc: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Y, %As.facet.c03) [concrete]
// CHECK:STDOUT:   %.84a: type = fn_type_with_self_type %As.WithSelf.Convert.type.edc, %As.facet.c03 [concrete]
// CHECK:STDOUT:   %X.as.As.impl.Convert.bound: <bound method> = bound_method %X.val, %X.as.As.impl.Convert [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.77c = value_binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, %Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc105.1: ref %Y = temporary @__global_init.%.loc105.1, @__global_init.%.loc105.2
// CHECK:STDOUT:   %.loc105.2: %Y = acquire_value %.loc105.1
// CHECK:STDOUT:   %n: %Y = value_binding n, %.loc105.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc100.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc101.1: %struct_type.x = struct_literal (%.loc100.1) [concrete = constants.%struct.948]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc101.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc101.3: ref %empty_tuple.type = class_element_access %.loc101.2, element0
// CHECK:STDOUT:   %.loc100.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc101.4: init %empty_tuple.type = converted %.loc100.1, %.loc100.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc101.5: init %X to %.loc101.2 = class_init (%.loc101.4) [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc102.1: init %X = converted %.loc101.1, %.loc101.5 [concrete = constants.%X.val]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %impl.elem0: %.84a = impl_witness_access constants.%As.impl_witness.e92, element0 [concrete = constants.%X.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc102.1, %impl.elem0 [concrete = constants.%X.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %.loc105.1: ref %Y = temporary_storage
// CHECK:STDOUT:   %.loc102.2: ref %X = temporary %.loc101.2, %.loc102.1
// CHECK:STDOUT:   %.loc102.3: %X = acquire_value %.loc102.2
// CHECK:STDOUT:   %X.as.As.impl.Convert.call: init %Y to %.loc105.1 = call %bound_method(%.loc102.3)
// CHECK:STDOUT:   %.loc105.2: init %Y = converted %.loc102.1, %X.as.As.impl.Convert.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
