// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/destroy_calls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/destroy_calls.carbon

// --- implicit_return.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}
class C {}

//@dump-sem-ir-begin
fn F() {
  var a: A;
  var b: B;
  var c: C;
}
//@dump-sem-ir-end

// --- nested_scope.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}
class C {}

//@dump-sem-ir-begin
fn F() {
  var a: A;
  var b: B;
  if (true) {
    var c: C;
  }
}
//@dump-sem-ir-end

// --- temp.carbon

library "[[@TEST_NAME]]";

class A { fn Make() -> A; }
class B { fn Make() -> B; }
class C { fn Make() -> C; }

//@dump-sem-ir-begin
fn F() {
  // TODO: The scoping of these destroy calls is incorrect. Maybe we need to
  // establish statement scopes?
  A.Make();
  B.Make();
  C.Make();
}
//@dump-sem-ir-end

// --- generic_class.carbon
library "[[@TEST_NAME]]";

class C {}

class D(template T:! type) {}

//@dump-sem-ir-begin
fn F() {
  var a: D(C);
}
//@dump-sem-ir-end

// --- generic_use_inside_generic.carbon
library "[[@TEST_NAME]]";

class C(template T:! type) {}

//@dump-sem-ir-begin
fn F(template T:! type) {
  var v: C(T);
}
//@dump-sem-ir-end

fn G() { F({}); }

// CHECK:STDOUT: --- implicit_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c10 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.049 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.049 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var %b.var_patt
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc12: %ptr.019 = addr_of %c.var
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr.loc12)
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.bound: <bound method> = bound_method %b.var, constants.%B.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc11: %ptr.e79 = addr_of %b.var
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.call: init %empty_tuple.type = call %B.as.Destroy.impl.Op.bound(%addr.loc11)
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%A.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc10: %ptr.6db = addr_of %a.var
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.call: init %empty_tuple.type = call %A.as.Destroy.impl.Op.bound(%addr.loc10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c10 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.049 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.049 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var %b.var_patt
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = bind_name b, %b.var
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc13: %ptr.019 = addr_of %c.var
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr.loc13)
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.bound: <bound method> = bound_method %b.var, constants.%B.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc11: %ptr.e79 = addr_of %b.var
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.call: init %empty_tuple.type = call %B.as.Destroy.impl.Op.bound(%addr.loc11)
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%A.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc10: %ptr.6db = addr_of %a.var
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.call: init %empty_tuple.type = call %A.as.Destroy.impl.Op.bound(%addr.loc10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- temp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.Make.type: type = fn_type @A.Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.Make: %A.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.Make.type: type = fn_type @B.Make [concrete]
// CHECK:STDOUT:   %B.Make: %B.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.Make.type: type = fn_type @C.Make [concrete]
// CHECK:STDOUT:   %C.Make: %C.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Make.ref.loc12: %A.Make.type = name_ref Make, @A.%A.Make.decl [concrete = constants.%A.Make]
// CHECK:STDOUT:   %.loc12_10.1: ref %A = temporary_storage
// CHECK:STDOUT:   %A.Make.call: init %A = call %Make.ref.loc12() to %.loc12_10.1
// CHECK:STDOUT:   %.loc12_10.2: ref %A = temporary %.loc12_10.1, %A.Make.call
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Make.ref.loc13: %B.Make.type = name_ref Make, @B.%B.Make.decl [concrete = constants.%B.Make]
// CHECK:STDOUT:   %.loc13_10.1: ref %B = temporary_storage
// CHECK:STDOUT:   %B.Make.call: init %B = call %Make.ref.loc13() to %.loc13_10.1
// CHECK:STDOUT:   %.loc13_10.2: ref %B = temporary %.loc13_10.1, %B.Make.call
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Make.ref.loc14: %C.Make.type = name_ref Make, @C.%C.Make.decl [concrete = constants.%C.Make]
// CHECK:STDOUT:   %.loc14_10.1: ref %C = temporary_storage
// CHECK:STDOUT:   %C.Make.call: init %C = call %Make.ref.loc14() to %.loc14_10.1
// CHECK:STDOUT:   %.loc14_10.2: ref %C = temporary %.loc14_10.1, %C.Make.call
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc14_10.2, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc14: %ptr.019 = addr_of %.loc14_10.2
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr.loc14)
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc13_10.2, constants.%B.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc13: %ptr.e79 = addr_of %.loc13_10.2
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.call: init %empty_tuple.type = call %B.as.Destroy.impl.Op.bound(%addr.loc13)
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc12_10.2, constants.%A.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc12: %ptr.6db = addr_of %.loc12_10.2
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.call: init %empty_tuple.type = call %A.as.Destroy.impl.Op.bound(%addr.loc12)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D.type: type = generic_class_type @D [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.113: type = class_type @D, @D(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.c957: type = pattern_type %D.113 [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type.50a: type = fn_type @D.as.Destroy.impl.Op, @D.as.Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.4c5: %D.as.Destroy.impl.Op.type.50a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.22d: type = ptr_type %D.113 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c957 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c957 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %D.113 = var %a.var_patt
// CHECK:STDOUT:   %.loc9: type = splice_block %D [concrete = constants.%D.113] {
// CHECK:STDOUT:     %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D: type = class_type @D, @D(constants.%C) [concrete = constants.%D.113]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %D.113 = bind_name a, %a.var
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%D.as.Destroy.impl.Op.4c5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %D.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.22d = addr_of %a.var
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_use_inside_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.4c2: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [template]
// CHECK:STDOUT:   %ptr.7d2: type = ptr_type %C.f2e [template]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.1da: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [template]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.d87: %C.as.Destroy.impl.Op.type.1da = struct_value () [template]
// CHECK:STDOUT:   %Destroy.facet.2c0: %Destroy.type = facet_value %C.f2e, (%Destroy.impl_witness.4c2) [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.389: <witness> = require_complete_type %C.f2e [template]
// CHECK:STDOUT:   %pattern_type.e5e: type = pattern_type %C.f2e [template]
// CHECK:STDOUT:   %.c7b: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.2c0 [template]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.d58: <specific function> = specific_function %C.as.Destroy.impl.Op.d87, @C.as.Destroy.impl.Op(%T) [template]
// CHECK:STDOUT:   %require_complete.448: <witness> = require_complete_type %ptr.7d2 [template]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.99a: type = pattern_type %C.7a7 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.40b: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6d2: %Destroy.type = facet_value %C.7a7, (%Destroy.impl_witness.40b) [concrete]
// CHECK:STDOUT:   %.75e: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.6d2 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.f30: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.23d: %C.as.Destroy.impl.Op.type.f30 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.308: type = ptr_type %C.7a7 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.ce9: <specific function> = specific_function %C.as.Destroy.impl.Op.23d, @C.as.Destroy.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %complete_type.903: <witness> = complete_type_witness %ptr.308 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc6_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_15.2: type) {
// CHECK:STDOUT:   %T.loc6_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc6_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_13.2: type = class_type @C, @C(%T.loc6_15.1) [template = %C.loc7_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %require_complete.loc7_13: <witness> = require_complete_type %C.loc7_13.2 [template = %require_complete.loc7_13 (constants.%require_complete.389)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc7_13.2 [template = %pattern_type (constants.%pattern_type.e5e)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T.loc6_15.1) [template = %Destroy.impl_witness (constants.%Destroy.impl_witness.4c2)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.loc7_13.2, (%Destroy.impl_witness) [template = %Destroy.facet (constants.%Destroy.facet.2c0)]
// CHECK:STDOUT:   %.loc7_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [template = %.loc7_3 (constants.%.c7b)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T.loc6_15.1) [template = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type.1da)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @F.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type.1da) = struct_value () [template = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.d87)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %C.as.Destroy.impl.Op, @C.as.Destroy.impl.Op(%T.loc6_15.1) [template = %C.as.Destroy.impl.Op.specific_fn (constants.%C.as.Destroy.impl.Op.specific_fn.d58)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C.loc7_13.2 [template = %ptr (constants.%ptr.7d2)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr [template = %require_complete.loc7_3 (constants.%require_complete.448)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.e5e) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.e5e) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%C.loc7_13.2 (%C.f2e) = var %v.var_patt
// CHECK:STDOUT:     %.loc7_13: type = splice_block %C.loc7_13.1 [template = %C.loc7_13.2 (constants.%C.f2e)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_15.2 [template = %T.loc6_15.1 (constants.%T)]
// CHECK:STDOUT:       %C.loc7_13.1: type = class_type @C, @C(constants.%T) [template = %C.loc7_13.2 (constants.%C.f2e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc7_13.2 (%C.f2e) = bind_name v, %v.var
// CHECK:STDOUT:     %impl.elem0: @F.%.loc7_3 (%.c7b) = impl_witness_access constants.%Destroy.impl_witness.4c2, element0 [template = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.d87)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Destroy.impl.Op(constants.%T) [template = %C.as.Destroy.impl.Op.specific_fn (constants.%C.as.Destroy.impl.Op.specific_fn.d58)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_fn
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.7d2) = addr_of %v.var
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc6_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_13.2 => constants.%C.7a7
// CHECK:STDOUT:   %require_complete.loc7_13 => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.99a
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.40b
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.6d2
// CHECK:STDOUT:   %.loc7_3 => constants.%.75e
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.f30
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.23d
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn => constants.%C.as.Destroy.impl.Op.specific_fn.ce9
// CHECK:STDOUT:   %ptr => constants.%ptr.308
// CHECK:STDOUT:   %require_complete.loc7_3 => constants.%complete_type.903
// CHECK:STDOUT: }
// CHECK:STDOUT:
