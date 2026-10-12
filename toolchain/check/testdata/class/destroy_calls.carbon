// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
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
  var unused a: A = {};
  var unused b: B = {};
  var unused c: C = {};
}
//@dump-sem-ir-end

// --- nested_scope.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}
class C {}

//@dump-sem-ir-begin
fn F() {
  var unused a: A = {};
  var unused b: B = {};
  if (true) {
    var unused c: C = {};
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
  A.Make();
  B.Make();
  C.Make();
}
//@dump-sem-ir-end

// --- generic_class.carbon
library "[[@TEST_NAME]]";

class C {}

class D(template T: type) {}

//@dump-sem-ir-begin
fn F() {
  var unused a: D(C) = {};
}
//@dump-sem-ir-end

// --- generic_use_inside_generic.carbon
library "[[@TEST_NAME]]";

class C(T: type) {}

//@dump-sem-ir-begin
fn F(generic T: type) {
  var unused v: C(T) = {};
}
//@dump-sem-ir-end

fn G() { F({}); }

// --- generic_use_inside_template.carbon
library "[[@TEST_NAME]]";

class C(template T: type) {}

//@dump-sem-ir-begin
fn F(template T: type) {
  var unused v: C(T) = {};
}
//@dump-sem-ir-end

fn G() { F({}); }

// CHECK:STDOUT: --- implicit_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.e39 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.e39 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc12_3.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.f6a: %facet_type = facet_value %C, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.5f9: %facet_type = facet_value %B, (%custom_witness.2cb555.3, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.4: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.2ff: %facet_type = facet_value %A, (%custom_witness.2cb555.4, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %A = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc10_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_22.2: init %A to %a.var = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_3.1: init %A = converted %.loc10_22.1, %.loc10_22.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc10_3.1
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc11_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_22.2: init %B to %b.var = class_init () [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc11_3.1: init %B = converted %.loc11_22.1, %.loc11_22.2 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc11_3.1
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.e39 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.e39 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc12_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc12_22.2: init %C to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc12_3.1: init %C = converted %.loc12_22.1, %.loc12_22.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc12_3.1
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc12_3.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc12_3.2: %facet_type = converted constants.%C, %facet_value.loc12_3.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc12_3.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc12_3.3: %facet_type = converted constants.%C, %facet_value.loc12_3.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc12: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%c.var)
// CHECK:STDOUT:   %facet_value.loc11_3.1: %facet_type = facet_value constants.%B, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   %.loc11_3.2: %facet_type = converted constants.%B, %facet_value.loc11_3.1 [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   %facet_value.loc11_3.2: %facet_type = facet_value constants.%B, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   %.loc11_3.3: %facet_type = converted constants.%B, %facet_value.loc11_3.2 [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%b.var)
// CHECK:STDOUT:   %facet_value.loc10_3.1: %facet_type = facet_value constants.%A, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   %.loc10_3.2: %facet_type = converted constants.%A, %facet_value.loc10_3.1 [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   %facet_value.loc10_3.2: %facet_type = facet_value constants.%A, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   %.loc10_3.3: %facet_type = converted constants.%A, %facet_value.loc10_3.2 [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc12_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc12_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12_3.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc11(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.e39 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.e39 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc13_5.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.f6a: %facet_type = facet_value %C, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.5f9: %facet_type = facet_value %B, (%custom_witness.2cb555.3, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.4: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.2ff: %facet_type = facet_value %A, (%custom_witness.2cb555.4, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %A = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc10_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_22.2: init %A to %a.var = class_init () [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc10_3.1: init %A = converted %.loc10_22.1, %.loc10_22.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   assign %a.var, %.loc10_3.1
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %a: ref %A = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.9ef = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.9ef = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc11_22.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_22.2: init %B to %b.var = class_init () [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc11_3.1: init %B = converted %.loc11_22.1, %.loc11_22.2 [concrete = constants.%B.val]
// CHECK:STDOUT:   assign %b.var, %.loc11_3.1
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: ref %B = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.e39 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.e39 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc13_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_24.2: init %C to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_5.1: init %C = converted %.loc13_24.1, %.loc13_24.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc13_5.1
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc13_5.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc13_5.2: %facet_type = converted constants.%C, %facet_value.loc13_5.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc13_5.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc13_5.3: %facet_type = converted constants.%C, %facet_value.loc13_5.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%c.var)
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %facet_value.loc11_3.1: %facet_type = facet_value constants.%B, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   %.loc11_3.2: %facet_type = converted constants.%B, %facet_value.loc11_3.1 [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   %facet_value.loc11_3.2: %facet_type = facet_value constants.%B, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   %.loc11_3.3: %facet_type = converted constants.%B, %facet_value.loc11_3.2 [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%b.var)
// CHECK:STDOUT:   %facet_value.loc10_3.1: %facet_type = facet_value constants.%A, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   %.loc10_3.2: %facet_type = converted constants.%A, %facet_value.loc10_3.1 [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   %facet_value.loc10_3.2: %facet_type = facet_value constants.%A, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   %.loc10_3.3: %facet_type = converted constants.%A, %facet_value.loc10_3.2 [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc13_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc13_5.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_5.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc11(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- temp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.Make.type: type = fn_type @A.Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.Make: %A.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.Make.type: type = fn_type @B.Make [concrete]
// CHECK:STDOUT:   %B.Make: %B.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.Make.type: type = fn_type @C.Make [concrete]
// CHECK:STDOUT:   %C.Make: %C.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc10_10.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_10.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.2ff: %facet_type = facet_value %A, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.5f9: %facet_type = facet_value %B, (%custom_witness.2cb555.3, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc12 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.4: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.f6a: %facet_type = facet_value %C, (%custom_witness.2cb555.4, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Make.ref.loc10: %A.Make.type = name_ref Make, @A.%A.Make.decl [concrete = constants.%A.Make]
// CHECK:STDOUT:   %.loc10_10.1: ref %A = temporary_storage
// CHECK:STDOUT:   %A.Make.call: init %A to %.loc10_10.1 = call %Make.ref.loc10()
// CHECK:STDOUT:   %.loc10_10.2: ref %A = temporary %.loc10_10.1, %A.Make.call
// CHECK:STDOUT:   %facet_value.loc10_10.1: %facet_type = facet_value constants.%A, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   %.loc10_10.3: %facet_type = converted constants.%A, %facet_value.loc10_10.1 [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   %facet_value.loc10_10.2: %facet_type = facet_value constants.%A, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   %.loc10_10.4: %facet_type = converted constants.%A, %facet_value.loc10_10.2 [concrete = constants.%facet_value.2ff]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_10.2)
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %Make.ref.loc11: %B.Make.type = name_ref Make, @B.%B.Make.decl [concrete = constants.%B.Make]
// CHECK:STDOUT:   %.loc11_10.1: ref %B = temporary_storage
// CHECK:STDOUT:   %B.Make.call: init %B to %.loc11_10.1 = call %Make.ref.loc11()
// CHECK:STDOUT:   %.loc11_10.2: ref %B = temporary %.loc11_10.1, %B.Make.call
// CHECK:STDOUT:   %facet_value.loc11_10.1: %facet_type = facet_value constants.%B, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   %.loc11_10.3: %facet_type = converted constants.%B, %facet_value.loc11_10.1 [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   %facet_value.loc11_10.2: %facet_type = facet_value constants.%B, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   %.loc11_10.4: %facet_type = converted constants.%B, %facet_value.loc11_10.2 [concrete = constants.%facet_value.5f9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc11_10.2)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Make.ref.loc12: %C.Make.type = name_ref Make, @C.%C.Make.decl [concrete = constants.%C.Make]
// CHECK:STDOUT:   %.loc12_10.1: ref %C = temporary_storage
// CHECK:STDOUT:   %C.Make.call: init %C to %.loc12_10.1 = call %Make.ref.loc12()
// CHECK:STDOUT:   %.loc12_10.2: ref %C = temporary %.loc12_10.1, %C.Make.call
// CHECK:STDOUT:   %facet_value.loc12_10.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc12_10.3: %facet_type = converted constants.%C, %facet_value.loc12_10.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc12_10.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc12_10.4: %facet_type = converted constants.%C, %facet_value.loc12_10.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc12: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc12_10.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc10_10.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_10.2(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc11(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc12(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %D.type: type = parameterized_class_name_type @D [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D.generic: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.dc6: type = class_type @D, @D(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.c20: type = pattern_type %D.dc6 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.c20 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.c20 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %D.val: %D.dc6 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %D.dc6, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %D.dc6 = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc9_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_25.2: init %D.dc6 to %a.var = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc9_3.1: init %D.dc6 = converted %.loc9_25.1, %.loc9_25.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_20: type = splice_block %D [concrete = constants.%D.dc6] {
// CHECK:STDOUT:     %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D.generic]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D: type = class_type @D, @D(constants.%C) [concrete = constants.%D.dc6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %D.dc6 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c20 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c20 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc9_3.1: %facet_type = facet_value constants.%D.dc6, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc9_3.2: %facet_type = converted constants.%D.dc6, %facet_value.loc9_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc9_3.2: %facet_type = facet_value constants.%D.dc6, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc9_3.3: %facet_type = converted constants.%D.dc6, %facet_value.loc9_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc9_3.2(%self.param: ref %D.dc6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_3.2(%self.param: ref %D.dc6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_use_inside_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.7e5: type = class_type @C, @C(%T.67d) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.466: <witness> = require_complete_type %C.7e5 [symbolic]
// CHECK:STDOUT:   %pattern_type.764: type = pattern_type %C.7e5 [symbolic]
// CHECK:STDOUT:   %v.patt.868: %pattern_type.764 = ref_binding_pattern v [symbolic]
// CHECK:STDOUT:   %v.var_patt.165: %pattern_type.764 = var_pattern %v.patt.868 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val.52e: %C.7e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.b82: <witness> = lookup_impl_witness %C.7e5, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.cc6: <witness> = lookup_impl_witness %C.7e5, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %facet_value.6d3: %facet_type = facet_value %C.7e5, (%DestroyImpl.lookup_impl_witness.b82, %SubobjectDestroy.lookup_impl_witness.cc6) [symbolic]
// CHECK:STDOUT:   %C.b19: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %pattern_type.55f: type = pattern_type %C.b19 [concrete]
// CHECK:STDOUT:   %v.patt.e42: %pattern_type.55f = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.f42: %pattern_type.55f = var_pattern %v.patt.e42 [concrete]
// CHECK:STDOUT:   %C.val.b0f: %C.b19 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.93a: %facet_type = facet_value %C.b19, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.0ba: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.93a) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_15.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_17.1: type = splice_block %.loc6_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_15.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_15.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc6_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_15.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_20.2: type = class_type @C, @C(%T.loc6_15.1) [symbolic = %C.loc7_20.2 (constants.%C.7e5)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %C.loc7_20.2 [symbolic = %require_complete (constants.%require_complete.466)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C.loc7_20.2 [symbolic = %pattern_type (constants.%pattern_type.764)]
// CHECK:STDOUT:   %v.patt.loc7_15.2: @F.%pattern_type (%pattern_type.764) = ref_binding_pattern v [symbolic = %v.patt.loc7_15.2 (constants.%v.patt.868)]
// CHECK:STDOUT:   %v.var_patt.loc7_3.2: @F.%pattern_type (%pattern_type.764) = var_pattern %v.patt.loc7_15.2 [symbolic = %v.var_patt.loc7_3.2 (constants.%v.var_patt.165)]
// CHECK:STDOUT:   %C.val: @F.%C.loc7_20.2 (%C.7e5) = struct_value () [symbolic = %C.val (constants.%C.val.52e)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %C.loc7_20.2, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.b82)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.loc7_20.2, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.cc6)]
// CHECK:STDOUT:   %facet_value.loc7_3.3: %facet_type = facet_value %C.loc7_20.2, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %facet_value.loc7_3.3 (constants.%facet_value.6d3)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%C.loc7_20.2 (%C.7e5) = var_storage %v.var_patt.loc7_3.1
// CHECK:STDOUT:     %.loc7_25.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_25.2: init @F.%C.loc7_20.2 (%C.7e5) to %v.var = class_init () [symbolic = %C.val (constants.%C.val.52e)]
// CHECK:STDOUT:     %.loc7_3.1: init @F.%C.loc7_20.2 (%C.7e5) = converted %.loc7_25.1, %.loc7_25.2 [symbolic = %C.val (constants.%C.val.52e)]
// CHECK:STDOUT:     assign %v.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_20: type = splice_block %C.loc7_20.1 [symbolic = %C.loc7_20.2 (constants.%C.7e5)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_15.2 [symbolic = %T.loc6_15.1 (constants.%T.67d)]
// CHECK:STDOUT:       %C.loc7_20.1: type = class_type @C, @C(constants.%T.67d) [symbolic = %C.loc7_20.2 (constants.%C.7e5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%C.loc7_20.2 (%C.7e5) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc7_15.1: @F.%pattern_type (%pattern_type.764) = ref_binding_pattern v [symbolic = %v.patt.loc7_15.2 (constants.%v.patt.868)]
// CHECK:STDOUT:       %v.var_patt.loc7_3.1: @F.%pattern_type (%pattern_type.764) = var_pattern %v.patt.loc7_15.1 [symbolic = %v.var_patt.loc7_3.2 (constants.%v.var_patt.165)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %facet_value.loc7_3.1: %facet_type = facet_value constants.%C.7e5, (constants.%DestroyImpl.lookup_impl_witness.b82, constants.%SubobjectDestroy.lookup_impl_witness.cc6) [symbolic = %facet_value.loc7_3.3 (constants.%facet_value.6d3)]
// CHECK:STDOUT:     %.loc7_3.2: %facet_type = converted constants.%C.7e5, %facet_value.loc7_3.1 [symbolic = %facet_value.loc7_3.3 (constants.%facet_value.6d3)]
// CHECK:STDOUT:     %facet_value.loc7_3.2: %facet_type = facet_value constants.%C.7e5, (constants.%DestroyImpl.lookup_impl_witness.b82, constants.%SubobjectDestroy.lookup_impl_witness.cc6) [symbolic = %facet_value.loc7_3.3 (constants.%facet_value.6d3)]
// CHECK:STDOUT:     %.loc7_3.3: %facet_type = converted constants.%C.7e5, %facet_value.loc7_3.2 [symbolic = %facet_value.loc7_3.3 (constants.%facet_value.6d3)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%v.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc7_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc7_3.2(%self.param: ref %C.b19) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.2(%self.param: ref %C.b19) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc6_15.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc6_15.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc6_15.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.loc7_20.2 => constants.%C.b19
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.55f
// CHECK:STDOUT:   %v.patt.loc7_15.2 => constants.%v.patt.e42
// CHECK:STDOUT:   %v.var_patt.loc7_3.2 => constants.%v.var_patt.f42
// CHECK:STDOUT:   %C.val => constants.%C.val.b0f
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness => constants.%custom_witness.2cb555.2
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness => constants.%custom_witness.b80ef6.2
// CHECK:STDOUT:   %facet_value.loc7_3.3 => constants.%facet_value.93a
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.0ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_use_inside_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %.7a9: type = type_of_inst @F.%.loc7_20.5 [template]
// CHECK:STDOUT:   %.154: %.7a9 = splice_inst @F.%.loc7_20.5 [template]
// CHECK:STDOUT:   %.537: type = splice_inst @F.%.loc7_20.8 [template]
// CHECK:STDOUT:   %require_complete.cb0: <witness> = require_complete_type %.537 [template]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %.537 [template]
// CHECK:STDOUT:   %v.patt.126: %pattern_type.fcb = ref_binding_pattern v [template]
// CHECK:STDOUT:   %v.var_patt.768: %pattern_type.fcb = var_pattern %v.patt.126 [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.5d6: %.537 = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %.5a2: type = type_of_inst @F.%.loc7_3.6 [template]
// CHECK:STDOUT:   %.13c: %.5a2 = splice_inst @F.%.loc7_3.6 [template]
// CHECK:STDOUT:   %C.b19ef7.1: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %inst.class_type: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %C.b19ef7.2: type = class_type @C, @C(%empty_struct_type) [concrete = %C.b19ef7.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.2b6: type = splice_block %C.b19ef7.1 [concrete = %C.b19ef7.1] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pattern_type.55f: type = pattern_type %C.b19ef7.1 [concrete]
// CHECK:STDOUT:   %v.patt.e42: %pattern_type.55f = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.f42: %pattern_type.55f = var_pattern %v.patt.e42 [concrete]
// CHECK:STDOUT:   %C.val: %C.b19ef7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.763: init %C.b19ef7.1 = splice_block %.b38 [concrete = %C.val] {
// CHECK:STDOUT:       %.69ec85.2: ref %C.b19ef7.1 = specific_inst @F.%v.var, @F(%empty_struct_type)
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%inst.splice_block.426619.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.93aa26.1: %facet_type = facet_value %C.b19ef7.1, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.70c: init %empty_tuple.type = splice_block %SelfDestruct.call {
// CHECK:STDOUT:       %.69ec85.1: ref %C.b19ef7.1 = specific_inst @F.%v.var, @F(%empty_struct_type)
// CHECK:STDOUT:       %facet_value.93aa26.2: %facet_type = facet_value %C.b19ef7.1, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete = %facet_value.93aa26.1]
// CHECK:STDOUT:       %.230b3c.1: %facet_type = converted %C.b19ef7.1, %facet_value.93aa26.2 [concrete = %facet_value.93aa26.1]
// CHECK:STDOUT:       %facet_value.93aa26.3: %facet_type = facet_value %C.b19ef7.1, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete = %facet_value.93aa26.1]
// CHECK:STDOUT:       %.230b3c.2: %facet_type = converted %C.b19ef7.1, %facet_value.93aa26.3 [concrete = %facet_value.93aa26.1]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.0bae62.2(%.69ec85.1)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_16.2: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc6_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc6_16.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc6_16.1: type = symbolic_binding T, 0, template [template = %T.loc6_16.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_20.5: <instruction> = call_action (%C.ref, %T.ref), false [template]
// CHECK:STDOUT:   %.loc7_20.6: type = type_of_inst %.loc7_20.5 [template = %.loc7_20.6 (constants.%.7a9)]
// CHECK:STDOUT:   %.loc7_20.7: @F.%.loc7_20.6 (%.7a9) = splice_inst %.loc7_20.5 [template = %.loc7_20.7 (constants.%.154)]
// CHECK:STDOUT:   %.loc7_20.8: <instruction> = convert_to_value_action %.loc7_20.3, type [template]
// CHECK:STDOUT:   %.loc7_20.9: type = splice_inst %.loc7_20.8 [template = %.loc7_20.9 (constants.%.537)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc7_20.9 [template = %require_complete (constants.%require_complete.cb0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc7_20.9 [template = %pattern_type (constants.%pattern_type.fcb)]
// CHECK:STDOUT:   %v.patt.loc7_15.2: @F.%pattern_type (%pattern_type.fcb) = ref_binding_pattern v [template = %v.patt.loc7_15.2 (constants.%v.patt.126)]
// CHECK:STDOUT:   %v.var_patt.loc7_3.2: @F.%pattern_type (%pattern_type.fcb) = var_pattern %v.patt.loc7_15.2 [template = %v.var_patt.loc7_3.2 (constants.%v.var_patt.768)]
// CHECK:STDOUT:   %.loc7_3.4: %tuple.type = initialize_action %.loc7_25, %.loc7_20.9, %v.var, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc7_3.4, element0 [template]
// CHECK:STDOUT:   %.loc7_3.5: @F.%.loc7_20.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc7_3.5 (constants.%.5d6)]
// CHECK:STDOUT:   %.loc7_3.6: <instruction> = call_action (constants.%SelfDestruct, %v.var), true [template]
// CHECK:STDOUT:   %.loc7_3.7: type = type_of_inst %.loc7_3.6 [template = %.loc7_3.7 (constants.%.5a2)]
// CHECK:STDOUT:   %.loc7_3.8: @F.%.loc7_3.7 (%.5a2) = splice_inst %.loc7_3.6 [template = %.loc7_3.8 (constants.%.13c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%.loc7_20.9 (%.537) = var_storage %v.var_patt.loc7_3.1
// CHECK:STDOUT:     %.loc7_25: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_3.1: @F.%.loc7_20.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc7_3.5 (constants.%.5d6)]
// CHECK:STDOUT:     assign %v.var, %.loc7_3.1
// CHECK:STDOUT:     %.loc7_20.1: type = splice_block %.loc7_20.4 [template = %.loc7_20.9 (constants.%.537)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_16.2 [template = %T.loc6_16.1 (constants.%T.67d)]
// CHECK:STDOUT:       %.loc7_20.2: type = type_of_inst %.loc7_20.5 [template = %.loc7_20.6 (constants.%.7a9)]
// CHECK:STDOUT:       %.loc7_20.3: @F.%.loc7_20.6 (%.7a9) = splice_inst %.loc7_20.5 [template = %.loc7_20.7 (constants.%.154)]
// CHECK:STDOUT:       %.loc7_20.4: type = splice_inst %.loc7_20.8 [template = %.loc7_20.9 (constants.%.537)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%.loc7_20.9 (%.537) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc7_15.1: @F.%pattern_type (%pattern_type.fcb) = ref_binding_pattern v [template = %v.patt.loc7_15.2 (constants.%v.patt.126)]
// CHECK:STDOUT:       %v.var_patt.loc7_3.1: @F.%pattern_type (%pattern_type.fcb) = var_pattern %v.patt.loc7_15.1 [template = %v.var_patt.loc7_3.2 (constants.%v.var_patt.768)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_3.2: type = type_of_inst %.loc7_3.6 [template = %.loc7_3.7 (constants.%.5a2)]
// CHECK:STDOUT:     %.loc7_3.3: @F.%.loc7_3.7 (%.5a2) = splice_inst %.loc7_3.6 [template = %.loc7_3.8 (constants.%.13c)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc7_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc7_3.2(%self.param: ref %C.b19ef7.1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.2(%self.param: ref %C.b19ef7.1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc6_16.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.patt.loc6_16.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc6_16.1 => constants.%empty_struct_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_20.5 => constants.%inst.class_type
// CHECK:STDOUT:   %.loc7_20.6 => type
// CHECK:STDOUT:   %.loc7_20.7 => constants.%C.b19ef7.1
// CHECK:STDOUT:   %.loc7_20.8 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc7_20.9 => constants.%C.b19ef7.1
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.55f
// CHECK:STDOUT:   %v.patt.loc7_15.2 => constants.%v.patt.e42
// CHECK:STDOUT:   %v.var_patt.loc7_3.2 => constants.%v.var_patt.f42
// CHECK:STDOUT:   %.loc7_3.4 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc7_3.5 => constants.%C.val
// CHECK:STDOUT:   %.loc7_3.6 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc7_3.7 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc7_3.8 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
