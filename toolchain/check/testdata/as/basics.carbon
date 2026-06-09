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
// CHECK:STDOUT:   %.loc6_17: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_26: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_27.1: %struct_type.y.x = struct_literal (%.loc6_17, %.loc6_26) [concrete = constants.%struct.f76]
// CHECK:STDOUT:   %.loc6_38.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_38.2: type = converted %.loc6_38.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc6_46.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_46.2: type = converted %.loc6_46.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.x.y.loc6: type = struct_type {.x: %empty_tuple.type, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y]
// CHECK:STDOUT:   %empty_tuple.loc6_26: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_27.2: %empty_tuple.type = converted %.loc6_26, %empty_tuple.loc6_26 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc6_17: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_27.3: %empty_tuple.type = converted %.loc6_17, %empty_tuple.loc6_17 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%.loc6_27.2, %.loc6_27.3) [concrete = constants.%struct.005]
// CHECK:STDOUT:   %.loc6_29.1: %struct_type.x.y = converted %.loc6_27.1, %struct [concrete = constants.%struct.005]
// CHECK:STDOUT:   %.loc6_29.2: %empty_tuple.type = struct_access %.loc6_29.1, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.3: ref %empty_tuple.type = struct_access %return.param, element0
// CHECK:STDOUT:   %.loc6_29.4: init %empty_tuple.type to %.loc6_29.3 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.5: init %empty_tuple.type = converted %.loc6_29.2, %.loc6_29.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.6: init %empty_tuple.type to %.loc6_29.3 = in_place_init %.loc6_29.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.7: %empty_tuple.type = struct_access %.loc6_29.1, element1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.8: ref %empty_tuple.type = struct_access %return.param, element1
// CHECK:STDOUT:   %.loc6_29.9: init %empty_tuple.type to %.loc6_29.8 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.10: init %empty_tuple.type = converted %.loc6_29.7, %.loc6_29.9 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.11: init %empty_tuple.type to %.loc6_29.8 = in_place_init %.loc6_29.10 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.12: init %struct_type.x.y to %return.param = struct_init (%.loc6_29.6, %.loc6_29.11) [concrete = constants.%struct.005]
// CHECK:STDOUT:   %.loc6_48: init %struct_type.x.y = converted %.loc6_29.1, %.loc6_29.12 [concrete = constants.%struct.005]
// CHECK:STDOUT:   return %.loc6_48 to %return.param
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
// CHECK:STDOUT:   %.loc5: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %t: type = value_binding t, @__global_init.%.loc5_24.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_17: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_21: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc5_22: %tuple.type.b6b = tuple_literal (%.loc5_17, %.loc5_21) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc5_27: type = type_literal type [concrete = type]
// CHECK:STDOUT:   %.loc5_24.1: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc5_24.2: type = converted constants.%empty_struct, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc5_24.3: type = converted %.loc5_22, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- as_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%X, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.359: type = tuple_type (%X, %X) [concrete]
// CHECK:STDOUT:   %pattern_type.3ee: type = pattern_type %tuple.type.359 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc13_40.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc20 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.3ee = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.ref.loc13_27: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_32.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc13_32: init %X to %.loc13_32.1 = call %Make.ref.loc13_27()
// CHECK:STDOUT:   %Make.ref.loc13_35: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_40.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc13_40: init %X to %.loc13_40.1 = call %Make.ref.loc13_35()
// CHECK:STDOUT:   %.loc13_41.1: %tuple.type.359 = tuple_literal (%Make.call.loc13_32, %Make.call.loc13_40)
// CHECK:STDOUT:   %X.ref.loc13_47: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc13_50: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc13_51.1: %tuple.type.24b = tuple_literal (%X.ref.loc13_47, %X.ref.loc13_50) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc13_51.2: type = converted %.loc13_51.1, constants.%tuple.type.359 [concrete = constants.%tuple.type.359]
// CHECK:STDOUT:   %.loc13_22.1: type = splice_block %.loc13_22.3 [concrete = constants.%tuple.type.359] {
// CHECK:STDOUT:     %X.ref.loc13_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc13_21: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc13_22.2: %tuple.type.24b = tuple_literal (%X.ref.loc13_18, %X.ref.loc13_21) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc13_22.3: type = converted %.loc13_22.2, constants.%tuple.type.359 [concrete = constants.%tuple.type.359]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_32.2: ref %X = temporary %.loc13_32.1, %Make.call.loc13_32
// CHECK:STDOUT:   %.loc13_32.3: %X = acquire_value %.loc13_32.2
// CHECK:STDOUT:   %.loc13_40.2: ref %X = temporary %.loc13_40.1, %Make.call.loc13_40
// CHECK:STDOUT:   %.loc13_40.3: %X = acquire_value %.loc13_40.2
// CHECK:STDOUT:   %tuple: %tuple.type.359 = tuple_value (%.loc13_32.3, %.loc13_40.3)
// CHECK:STDOUT:   %.loc13_41.2: %tuple.type.359 = converted %.loc13_41.1, %tuple
// CHECK:STDOUT:   %a: %tuple.type.359 = value_binding a, %.loc13_41.2
// CHECK:STDOUT:   %Destroy.Op.bound.loc13_40: <bound method> = bound_method %.loc13_40.2, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc13_40: init %empty_tuple.type = call %Destroy.Op.bound.loc13_40(%.loc13_40.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc13_32: <bound method> = bound_method %.loc13_32.2, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc13_32: init %empty_tuple.type = call %Destroy.Op.bound.loc13_32(%.loc13_32.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_40.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_40.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.3ee = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.3ee = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.359 = var %b.var_patt
// CHECK:STDOUT:   %Make.ref.loc20_27: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem0: ref %X = tuple_access %b.var, element0
// CHECK:STDOUT:   %Make.call.loc20_32: init %X to %tuple.elem0 = call %Make.ref.loc20_27()
// CHECK:STDOUT:   %Make.ref.loc20_35: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem1: ref %X = tuple_access %b.var, element1
// CHECK:STDOUT:   %Make.call.loc20_40: init %X to %tuple.elem1 = call %Make.ref.loc20_35()
// CHECK:STDOUT:   %.loc20_41.1: %tuple.type.359 = tuple_literal (%Make.call.loc20_32, %Make.call.loc20_40)
// CHECK:STDOUT:   %X.ref.loc20_47: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc20_50: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc20_51.1: %tuple.type.24b = tuple_literal (%X.ref.loc20_47, %X.ref.loc20_50) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc20_51.2: type = converted %.loc20_51.1, constants.%tuple.type.359 [concrete = constants.%tuple.type.359]
// CHECK:STDOUT:   %.loc20_41.2: init %tuple.type.359 to %b.var = tuple_init (%Make.call.loc20_32, %Make.call.loc20_40)
// CHECK:STDOUT:   %.loc20_3: init %tuple.type.359 = converted %.loc20_41.1, %.loc20_41.2
// CHECK:STDOUT:   assign %b.var, %.loc20_3
// CHECK:STDOUT:   %.loc20_22.1: type = splice_block %.loc20_22.3 [concrete = constants.%tuple.type.359] {
// CHECK:STDOUT:     %X.ref.loc20_18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc20_21: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc20_22.2: %tuple.type.24b = tuple_literal (%X.ref.loc20_18, %X.ref.loc20_21) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc20_22.3: type = converted %.loc20_22.2, constants.%tuple.type.359 [concrete = constants.%tuple.type.359]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.359 = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %b.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%b.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20(%self.param: ref %tuple.type.359) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.db8: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.990: type = pattern_type %ptr.db8 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Value(%n.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %pattern_type.6cb = value_binding_pattern m [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.ref: %X = name_ref n, %n
// CHECK:STDOUT:   %X.ref.loc10_26: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc10_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %m: %X = value_binding m, %n.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Reference(%p.param: %ptr.db8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %pattern_type.990 = value_binding_pattern q [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.ref: %ptr.db8 = name_ref p, %p
// CHECK:STDOUT:   %.loc16_24: ref %X = deref %p.ref
// CHECK:STDOUT:   %X.ref.loc16_30: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %addr: %ptr.db8 = addr_of %.loc16_24
// CHECK:STDOUT:   %.loc16_18: type = splice_block %ptr.loc16 [concrete = constants.%ptr.db8] {
// CHECK:STDOUT:     %X.ref.loc16_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %X.ref.loc16_17 [concrete = constants.%ptr.db8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: %ptr.db8 = value_binding q, %addr
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Initializing() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6cb = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.6cb = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %X = var %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc24: ref %X = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %X to %.loc24 = call %Make.ref()
// CHECK:STDOUT:   %X.ref.loc24_32: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %X.ref.loc24_17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %x: ref %X = ref_binding x, %x.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %x.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_3.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %pattern_type.7ab: type = pattern_type %Y [concrete]
// CHECK:STDOUT:   %struct.948: %struct_type.x = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %As.type.f20: type = facet_type <@As, @As(%Y)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.915: <witness> = impl_witness @X.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %X.as.As.impl.Convert.type: type = fn_type @X.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %X.as.As.impl.Convert: %X.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.9dc: %As.type.f20 = facet_value %X, (%As.impl_witness.915) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f49: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%Y, %As.facet.9dc) [concrete]
// CHECK:STDOUT:   %.db2: type = fn_type_with_self_type %As.WithSelf.Convert.type.f49, %As.facet.9dc [concrete]
// CHECK:STDOUT:   %X.as.As.impl.Convert.bound: <bound method> = bound_method %X.val, %X.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %.650: ref %X = temporary invalid, %X.val [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ab = value_binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, %Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc19_29.1: ref %Y = temporary @__global_init.%.loc19_29.1, @__global_init.%.loc19_29.2
// CHECK:STDOUT:   %.loc19_29.2: %Y = acquire_value %.loc19_29.1
// CHECK:STDOUT:   %n: %Y = value_binding n, %.loc19_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_20.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_21.1: %struct_type.x = struct_literal (%.loc19_20.1) [concrete = constants.%struct.948]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc19_21.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc19_21.3: ref %empty_tuple.type = class_element_access %.loc19_21.2, element0
// CHECK:STDOUT:   %.loc19_20.2: init %empty_tuple.type to %.loc19_21.3 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_21.4: init %empty_tuple.type = converted %.loc19_20.1, %.loc19_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_21.5: init %empty_tuple.type to %.loc19_21.3 = in_place_init %.loc19_21.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_21.6: init %X to %.loc19_21.2 = class_init (%.loc19_21.5) [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc19_23.1: init %X = converted %.loc19_21.1, %.loc19_21.6 [concrete = constants.%X.val]
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %impl.elem0: %.db2 = impl_witness_access constants.%As.impl_witness.915, element0 [concrete = constants.%X.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc19_23.1, %impl.elem0 [concrete = constants.%X.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %.loc19_29.1: ref %Y = temporary_storage
// CHECK:STDOUT:   %.loc19_23.2: ref %X = temporary %.loc19_21.2, %.loc19_23.1 [concrete = constants.%.650]
// CHECK:STDOUT:   %.loc19_23.3: %X = acquire_value %.loc19_23.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %X.as.As.impl.Convert.call: init %Y to %.loc19_29.1 = call %bound_method(%.loc19_23.3)
// CHECK:STDOUT:   %.loc19_29.2: init %Y = converted %.loc19_23.1, %X.as.As.impl.Convert.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
