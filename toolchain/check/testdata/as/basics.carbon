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
  let a: (X, X) = (Make(), Make()) as (X, X);
  //@dump-sem-ir-end
}

fn Var() {
  // This should initialize both tuple elements in place.
  //@dump-sem-ir-begin
  var b: (X, X) = (Make(), Make()) as (X, X);
  //@dump-sem-ir-end
}

// --- identity.carbon

library "[[@TEST_NAME]]";

class X {
  // ...
}

fn Value(n: X) {
  //@dump-sem-ir-begin
  let m: X = n as X;
  //@dump-sem-ir-end
}

fn Reference(p: X*) {
  //@dump-sem-ir-begin
  let q: X* = &(*p as X);
  //@dump-sem-ir-end
}

fn Make() -> X;

fn Initializing() {
  //@dump-sem-ir-begin
  var x: X = (Make() as X);
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
// CHECK:STDERR: fail_no_conversion.carbon:[[@LINE+4]]:13: note: type `Core.IntLiteral` does not implement interface `Core.As({})` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let a: {} = 1 as {};
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
let a: {} = 1 as {};

// --- fail_not_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_not_type.carbon:[[@LINE+7]]:32: error: cannot implicitly convert non-type value of type `{.x: ()}` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: let n: {.x: ()} = {.x = ()} as {.x = ()};
// CHECK:STDERR:                                ^~~~~~~~~
// CHECK:STDERR: fail_not_type.carbon:[[@LINE+4]]:32: note: type `{.x: ()}` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: let n: {.x: ()} = {.x = ()} as {.x = ()};
// CHECK:STDERR:                                ^~~~~~~~~
// CHECK:STDERR:
let n: {.x: ()} = {.x = ()} as {.x = ()};

// CHECK:STDOUT: --- simple_as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: %empty_tuple.type, .y: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct_type.y.x: type = struct_type {.y: %empty_tuple.type, .x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> %return.param: %struct_type.x.y {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_17: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_27.1: %struct_type.y.x = struct_literal (%.loc6_17, %.loc6_26)
// CHECK:STDOUT:   %.loc6_38.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_38.2: type = converted %.loc6_38.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc6_46.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_46.2: type = converted %.loc6_46.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %struct_type.x.y.loc6: type = struct_type {.x: %empty_tuple.type, .y: %empty_tuple.type} [concrete = constants.%struct_type.x.y]
// CHECK:STDOUT:   %empty_tuple.loc6_26: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_27.2: %empty_tuple.type = converted %.loc6_26, %empty_tuple.loc6_26 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc6_17: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_27.3: %empty_tuple.type = converted %.loc6_17, %empty_tuple.loc6_17 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%.loc6_27.2, %.loc6_27.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc6_29.1: %struct_type.x.y = converted %.loc6_27.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc6_29.2: %empty_tuple.type = struct_access %.loc6_29.1, element0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.3: ref %empty_tuple.type = struct_access %return, element0
// CHECK:STDOUT:   %.loc6_29.4: init %empty_tuple.type = tuple_init () to %.loc6_29.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.5: init %empty_tuple.type = converted %.loc6_29.2, %.loc6_29.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.6: %empty_tuple.type = struct_access %.loc6_29.1, element1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.7: ref %empty_tuple.type = struct_access %return, element1
// CHECK:STDOUT:   %.loc6_29.8: init %empty_tuple.type = tuple_init () to %.loc6_29.7 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.9: init %empty_tuple.type = converted %.loc6_29.6, %.loc6_29.8 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_29.10: init %struct_type.x.y = struct_init (%.loc6_29.5, %.loc6_29.9) to %return [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc6_48: init %struct_type.x.y = converted %.loc6_29.1, %.loc6_29.10 [concrete = constants.%struct]
// CHECK:STDOUT:   return %.loc6_48 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %tuple.type.b6b: type = tuple_type (%empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %pattern_type = binding_pattern t [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: type = bind_name t, @__global_init.%.loc5_24.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_17: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_21: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc5_22: %tuple.type.b6b = tuple_literal (%.loc5_17, %.loc5_21)
// CHECK:STDOUT:   %.loc5_24.1: type = converted %.loc5_17, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc5_24.2: type = converted %.loc5_21, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc5_24.3: type = converted %.loc5_22, constants.%tuple.type.b6b [concrete = constants.%tuple.type.b6b]
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
// CHECK:STDOUT:   %tuple.type.b67: type = tuple_type (%X, %X) [concrete]
// CHECK:STDOUT:   %pattern_type.bb7: type = pattern_type %tuple.type.b67 [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.b19: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.dc1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.b19) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.177: %DestroyT.as_type.as.Destroy.impl.Op.type.dc1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %facet_value.4ff: %type_where = facet_value %tuple.type.b67, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.1bb: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.4ff) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.482: %DestroyT.as_type.as.Destroy.impl.Op.type.1bb = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.120: type = ptr_type %tuple.type.b67 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.bb7 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.ref.loc13_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_25.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc13_25: init %X = call %Make.ref.loc13_20() to %.loc13_25.1
// CHECK:STDOUT:   %Make.ref.loc13_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_33.1: ref %X = temporary_storage
// CHECK:STDOUT:   %Make.call.loc13_33: init %X = call %Make.ref.loc13_28() to %.loc13_33.1
// CHECK:STDOUT:   %.loc13_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc13_25, %Make.call.loc13_33)
// CHECK:STDOUT:   %X.ref.loc13_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc13_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc13_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc13_40, %X.ref.loc13_43)
// CHECK:STDOUT:   %.loc13_44.2: type = converted %.loc13_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc13_15.1: type = splice_block %.loc13_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc13_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc13_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc13_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc13_11, %X.ref.loc13_14)
// CHECK:STDOUT:     %.loc13_15.3: type = converted %.loc13_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_25.2: ref %X = temporary %.loc13_25.1, %Make.call.loc13_25
// CHECK:STDOUT:   %.loc13_25.3: %X = bind_value %.loc13_25.2
// CHECK:STDOUT:   %.loc13_33.2: ref %X = temporary %.loc13_33.1, %Make.call.loc13_33
// CHECK:STDOUT:   %.loc13_33.3: %X = bind_value %.loc13_33.2
// CHECK:STDOUT:   %tuple: %tuple.type.b67 = tuple_value (%.loc13_25.3, %.loc13_33.3)
// CHECK:STDOUT:   %.loc13_34.2: %tuple.type.b67 = converted %.loc13_34.1, %tuple
// CHECK:STDOUT:   %a: %tuple.type.b67 = bind_name a, %.loc13_34.2
// CHECK:STDOUT:   %facet_value.loc13_33: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value.b19]
// CHECK:STDOUT:   %.loc13_33.4: %type_where = converted constants.%X, %facet_value.loc13_33 [concrete = constants.%facet_value.b19]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc13_33: <bound method> = bound_method %.loc13_33.2, constants.%DestroyT.as_type.as.Destroy.impl.Op.177
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc13_33: <bound method> = bound_method %.loc13_33.2, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc13_33: %ptr.d17 = addr_of %.loc13_33.2
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc13_33: init %empty_tuple.type = call %bound_method.loc13_33(%addr.loc13_33)
// CHECK:STDOUT:   %facet_value.loc13_25: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value.b19]
// CHECK:STDOUT:   %.loc13_25.4: %type_where = converted constants.%X, %facet_value.loc13_25 [concrete = constants.%facet_value.b19]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc13_25: <bound method> = bound_method %.loc13_25.2, constants.%DestroyT.as_type.as.Destroy.impl.Op.177
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc13_25: <bound method> = bound_method %.loc13_25.2, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc13_25: %ptr.d17 = addr_of %.loc13_25.2
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc13_25: init %empty_tuple.type = call %bound_method.loc13_25(%addr.loc13_25)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.bb7 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.bb7 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.b67 = var %b.var_patt
// CHECK:STDOUT:   %Make.ref.loc20_20: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem0: ref %X = tuple_access %b.var, element0
// CHECK:STDOUT:   %Make.call.loc20_25: init %X = call %Make.ref.loc20_20() to %tuple.elem0
// CHECK:STDOUT:   %Make.ref.loc20_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %tuple.elem1: ref %X = tuple_access %b.var, element1
// CHECK:STDOUT:   %Make.call.loc20_33: init %X = call %Make.ref.loc20_28() to %tuple.elem1
// CHECK:STDOUT:   %.loc20_34.1: %tuple.type.b67 = tuple_literal (%Make.call.loc20_25, %Make.call.loc20_33)
// CHECK:STDOUT:   %X.ref.loc20_40: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc20_43: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc20_44.1: %tuple.type.24b = tuple_literal (%X.ref.loc20_40, %X.ref.loc20_43)
// CHECK:STDOUT:   %.loc20_44.2: type = converted %.loc20_44.1, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   %.loc20_34.2: init %tuple.type.b67 = tuple_init (%Make.call.loc20_25, %Make.call.loc20_33) to %b.var
// CHECK:STDOUT:   %.loc20_3.1: init %tuple.type.b67 = converted %.loc20_34.1, %.loc20_34.2
// CHECK:STDOUT:   assign %b.var, %.loc20_3.1
// CHECK:STDOUT:   %.loc20_15.1: type = splice_block %.loc20_15.3 [concrete = constants.%tuple.type.b67] {
// CHECK:STDOUT:     %X.ref.loc20_11: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %X.ref.loc20_14: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %.loc20_15.2: %tuple.type.24b = tuple_literal (%X.ref.loc20_11, %X.ref.loc20_14)
// CHECK:STDOUT:     %.loc20_15.3: type = converted %.loc20_15.2, constants.%tuple.type.b67 [concrete = constants.%tuple.type.b67]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.b67 = bind_name b, %b.var
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%tuple.type.b67, () [concrete = constants.%facet_value.4ff]
// CHECK:STDOUT:   %.loc20_3.2: %type_where = converted constants.%tuple.type.b67, %facet_value [concrete = constants.%facet_value.4ff]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %b.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.482
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %b.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.120 = addr_of %b.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.1c6: type = pattern_type %ptr.d17 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.dc1: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.177: %DestroyT.as_type.as.Destroy.impl.Op.type.dc1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Value(%n.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %pattern_type.019 = binding_pattern m [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.ref: %X = name_ref n, %n
// CHECK:STDOUT:   %X.ref.loc10_19: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc10_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %m: %X = bind_name m, %n.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Reference(%p.param: %ptr.d17) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %pattern_type.1c6 = binding_pattern q [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.ref: %ptr.d17 = name_ref p, %p
// CHECK:STDOUT:   %.loc16_17: ref %X = deref %p.ref
// CHECK:STDOUT:   %X.ref.loc16_23: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %addr: %ptr.d17 = addr_of %.loc16_17
// CHECK:STDOUT:   %.loc16_11: type = splice_block %ptr.loc16 [concrete = constants.%ptr.d17] {
// CHECK:STDOUT:     %X.ref.loc16_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %X.ref.loc16_10 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: %ptr.d17 = bind_name q, %addr
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Initializing() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.019 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.019 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %X = var %x.var_patt
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc24_3.1: ref %X = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %X = call %Make.ref() to %.loc24_3.1
// CHECK:STDOUT:   %X.ref.loc24_25: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %X.ref.loc24_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %x: ref %X = bind_name x, %x.var
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc24_3.2: %type_where = converted constants.%X, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %x.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.177
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.d17 = addr_of %x.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %pattern_type.c27: type = pattern_type %Y [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %As.type.d60: type = facet_type <@As, @As(%Y)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.d2d: type = fn_type @As.Convert, @As(%Y) [concrete]
// CHECK:STDOUT:   %As.impl_witness.3fa: <witness> = impl_witness file.%As.impl_witness_table.loc14 [concrete]
// CHECK:STDOUT:   %X.as.As.impl.Convert.type: type = fn_type @X.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %X.as.As.impl.Convert: %X.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.2fd: %As.type.d60 = facet_value %X, (%As.impl_witness.3fa) [concrete]
// CHECK:STDOUT:   %.e12: type = fn_type_with_self_type %As.Convert.type.d2d, %As.facet.2fd [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.c27 = binding_pattern n [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, %Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %.loc19_29.1: ref %Y = temporary @__global_init.%.loc19_29.1, @__global_init.%.loc19_29.2
// CHECK:STDOUT:   %.loc19_29.2: %Y = bind_value %.loc19_29.1
// CHECK:STDOUT:   %n: %Y = bind_name n, %.loc19_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_20.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_21.1: %struct_type.x = struct_literal (%.loc19_20.1)
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc19_21.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc19_21.3: ref %empty_tuple.type = class_element_access %.loc19_21.2, element0
// CHECK:STDOUT:   %.loc19_20.2: init %empty_tuple.type = tuple_init () to %.loc19_21.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_21.4: init %empty_tuple.type = converted %.loc19_20.1, %.loc19_20.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_21.5: init %X = class_init (%.loc19_21.4), %.loc19_21.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc19_21.6: ref %X = temporary %.loc19_21.2, %.loc19_21.5
// CHECK:STDOUT:   %.loc19_23.1: ref %X = converted %.loc19_21.1, %.loc19_21.6
// CHECK:STDOUT:   %Y.ref: type = name_ref Y, file.%Y.decl [concrete = constants.%Y]
// CHECK:STDOUT:   %impl.elem0: %.e12 = impl_witness_access constants.%As.impl_witness.3fa, element0 [concrete = constants.%X.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc19_23.1, %impl.elem0
// CHECK:STDOUT:   %.loc19_29.1: ref %Y = temporary_storage
// CHECK:STDOUT:   %.loc19_23.2: %X = bind_value %.loc19_23.1
// CHECK:STDOUT:   %X.as.As.impl.Convert.call: init %Y = call %bound_method(%.loc19_23.2) to %.loc19_29.1
// CHECK:STDOUT:   %.loc19_29.2: init %Y = converted %.loc19_23.1, %X.as.As.impl.Convert.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
