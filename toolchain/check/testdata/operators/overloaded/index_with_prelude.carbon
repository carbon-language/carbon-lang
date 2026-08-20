// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/index_with_prelude.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/index_with_prelude.carbon

// --- overloaded_index.carbon

library "[[@TEST_NAME]]";

class C {}
class ElementType {}
class SubscriptType {}

impl C as Core.IndexWith(SubscriptType) where .ElementType = ElementType {
  fn At(unused self, unused subscript: SubscriptType) -> ElementType {
    return {};
  }
}

let s: SubscriptType = {};
let c: C = {};
let x: ElementType = c[s];

// --- overloaded_builtin.carbon

library "[[@TEST_NAME]]";

class C {
  adapt ();
}

impl (C, C) as Core.IndexWith(Core.IntLiteral) where .ElementType = C {
  fn At(self, unused subscript: Core.IntLiteral) -> C {
    return self.0;
  }
}

let s: (C, C) = (() as C, () as C);
let e: C = s[0];

// --- fail_invalid_subscript_type.carbon

library "[[@TEST_NAME]]";

class C { }
class ElementType {}
class SubscriptType {}

impl C as Core.IndexWith(SubscriptType) where .ElementType = ElementType {
  fn At(unused self, unused subscript: SubscriptType) -> ElementType {
    return {};
  }
}

let c: C = {};
// CHECK:STDERR: fail_invalid_subscript_type.carbon:[[@LINE+4]]:22: error: cannot access member of interface `Core.IndexWith(Core.IntLiteral)` in type `C` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: let x: ElementType = c[0];
// CHECK:STDERR:                      ^~~~
// CHECK:STDERR:
let x: ElementType = c[0];

// --- fail_index_with_not_implemented.carbon

library "[[@TEST_NAME]]";

class C { }

let c: C = {};
// CHECK:STDERR: fail_index_with_not_implemented.carbon:[[@LINE+4]]:14: error: cannot access member of interface `Core.IndexWith(Core.IntLiteral)` in type `C` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: let x: i32 = c[0];
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let x: i32 = c[0];

// --- indirect_types.carbon

library "[[@TEST_NAME]]";

class Copyable {
  impl as Core.Copy {
    fn Op(unused self) -> Self { return {}; }
  }
}
class NonCopyable {}
class SubscriptType {}

class C {
  var ptr: Copyable*;
}

impl C as Core.IndirectIndexWith(SubscriptType) where .ElementType = Copyable {
  fn Ref(self, unused subscript: SubscriptType) -> ref Copyable {
    return *self.ptr;
  }
}

class CNonCopy {
  var ptr: NonCopyable*;
}

impl CNonCopy as Core.IndirectIndexWith(SubscriptType) where .ElementType = NonCopyable {
  fn Ref(self, unused subscript: SubscriptType) -> ref NonCopyable {
    return *self.ptr;
  }
}

// --- direct_ref_copyable.carbon

library "[[@TEST_NAME]]";

import library "indirect_types";

fn TestDirectRef(c: C, s: SubscriptType) -> ref Copyable {
  return c.(Core.IndirectIndexWith(SubscriptType).Ref)(s);
}

// --- direct_ref_non_copyable.carbon

library "[[@TEST_NAME]]";

import library "indirect_types";

fn TestDirectRefNonCopy(c: CNonCopy, s: SubscriptType) -> ref NonCopyable {
  return c.(Core.IndirectIndexWith(SubscriptType).Ref)(s);
}

// --- indirect_index.carbon

library "[[@TEST_NAME]]";

import library "indirect_types";

fn TestSubscript(c: C, s: SubscriptType) -> Copyable {
  return c[s];
}

// --- fail_todo_non_copy_subscript.carbon

library "[[@TEST_NAME]]";

import library "indirect_types";

fn TestSubscript(c: CNonCopy, s: SubscriptType) {
  // Subscripting should eventually work by using IndirectIndexWith directly
  // without requiring Copy (rather than using IndexWith implemented in terms
  // of IndirectIndexWith).
  // CHECK:STDERR: fail_todo_non_copy_subscript.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.IndexWith(SubscriptType)` in type `CNonCopy` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   c[s];
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  c[s];
}





// CHECK:STDOUT: --- overloaded_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ElementType: type = class_type @ElementType.loc5 [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndexWith.type.df6: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.df6 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.e15: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndexWith.type.d74: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %.Self.frozen.ffd: %IndexWith.type.d74 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndexWith.assoc_type.144: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %assoc0.c96: %IndexWith.assoc_type.144 = assoc_entity element0, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.ca0: type = facet_access_type %.Self.frozen.ffd [symbolic_self]
// CHECK:STDOUT:   %assoc0.a4ed04.2: %IndexWith.assoc_type.e15 = assoc_entity element0, imports.%Core.import_ref.6b6e53.2 [symbolic]
// CHECK:STDOUT:   %IndexWith.lookup_impl_witness.a61: <witness> = lookup_impl_witness %.Self.frozen.ffd, @IndexWith, @IndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e0a: type = impl_witness_access %IndexWith.lookup_impl_witness.a61, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.1dc: %IndexWith.type.d74 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndexWith.lookup_impl_witness.2a2: <witness> = lookup_impl_witness %.Self.1dc, @IndexWith, @IndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc1: type = impl_witness_access %IndexWith.lookup_impl_witness.2a2, element0 [symbolic_self]
// CHECK:STDOUT:   %IndexWith_where.type.7d8: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.f8f) where %impl.elem0.fc1 = %ElementType> [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness.6b3: <witness> = impl_witness @C.as.IndexWith.impl.%IndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b3: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.303: %pattern_type.98b3 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.49a: %pattern_type.98b3 = at_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %pattern_type.ea5: type = pattern_type %SubscriptType.f8f [concrete]
// CHECK:STDOUT:   %subscript.param_patt.5ee: %pattern_type.ea5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %subscript.patt.a6f: %pattern_type.ea5 = at_binding_pattern subscript, %subscript.param_patt.5ee [concrete]
// CHECK:STDOUT:   %IndexWith.facet.aa3: %IndexWith.type.d74 = facet_value %C, (%IndexWith.impl_witness.6b3) [concrete]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.41d: type = fn_type @IndexWith.WithSelf.At, @IndexWith.WithSelf(%SubscriptType.f8f, %IndexWith.facet.aa3) [concrete]
// CHECK:STDOUT:   %.4d0: Core.Form = init_form %ElementType [concrete]
// CHECK:STDOUT:   %pattern_type.67c: type = pattern_type %ElementType [concrete]
// CHECK:STDOUT:   %return.param_patt.41c: %pattern_type.67c = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.66a: %pattern_type.67c = return_slot_pattern %return.param_patt.41c, %ElementType [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.type: type = fn_type @C.as.IndexWith.impl.At [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At: %C.as.IndexWith.impl.At.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ElementType.val: %ElementType = struct_value () [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.ea5 = value_binding_pattern s [concrete]
// CHECK:STDOUT:   %SubscriptType.val: %SubscriptType.f8f = struct_value () [concrete]
// CHECK:STDOUT:   %.b1d: ref %SubscriptType.f8f = temporary invalid, %SubscriptType.val [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b3 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.67c = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %.6e7: type = fn_type_with_self_type %IndexWith.WithSelf.At.type.41d, %IndexWith.facet.aa3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.df6 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.bc6: @IndexWith.WithSelf.%IndexWith.assoc_type (%IndexWith.assoc_type.e15) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.WithSelf.%assoc0 (constants.%assoc0.a4ed04.2)]
// CHECK:STDOUT:   %Core.import_ref.fc2: type = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %Core.import_ref.6b6e53.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %ElementType: type = assoc_const_decl @ElementType.1 [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ElementType = %ElementType.decl
// CHECK:STDOUT:     .SubscriptType = %SubscriptType.decl
// CHECK:STDOUT:     .s = %s
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType.loc5 [concrete = constants.%ElementType] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [concrete = constants.%SubscriptType.f8f] {} {}
// CHECK:STDOUT:   impl_decl @C.as.IndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.df6 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndexWith.type.d74]
// CHECK:STDOUT:     %.Self.frozen: %IndexWith.type.d74 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.ffd]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %IndexWith.type [concrete]
// CHECK:STDOUT:     %.Self.ref: %IndexWith.type.d74 = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.ffd]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type.ca0]
// CHECK:STDOUT:     %.loc8_47.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type.ca0]
// CHECK:STDOUT:     %.loc8_47.2: %IndexWith.assoc_type.144 = specific_constant imports.%Core.import_ref.bc6, @IndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%.Self.frozen.ffd) [concrete = constants.%assoc0.c96]
// CHECK:STDOUT:     %ElementType.ref.loc8_47: %IndexWith.assoc_type.144 = name_ref ElementType, %.loc8_47.2 [concrete = constants.%assoc0.c96]
// CHECK:STDOUT:     %impl.elem0.loc8_47.1: type = impl_witness_access constants.%IndexWith.lookup_impl_witness.a61, element0 [symbolic_self = constants.%impl.elem0.e0a]
// CHECK:STDOUT:     %ElementType.ref.loc8_62: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:     %rewrite.loc8_60.1 = requirement_rewrite %impl.elem0.loc8_47.1, %ElementType.ref.loc8_62 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_47.2: type = impl_witness_access constants.%IndexWith.lookup_impl_witness.2a2, element0 [symbolic_self = constants.%impl.elem0.fc1]
// CHECK:STDOUT:     %rewrite.loc8_60.2 = requirement_rewrite %impl.elem0.loc8_47.2, %ElementType.ref.loc8_62 [concrete]
// CHECK:STDOUT:     %.loc8_41: type = where_expr [concrete = constants.%IndexWith_where.type.7d8] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %IndexWith.type [concrete]
// CHECK:STDOUT:       %rewrite.loc8_60.2 = requirement_rewrite %impl.elem0.loc8_47.2, %ElementType.ref.loc8_62 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_25.1: ref %SubscriptType.f8f = temporary_storage
// CHECK:STDOUT:   %.loc14_25.2: init %SubscriptType.f8f to %.loc14_25.1 = class_init () [concrete = constants.%SubscriptType.val]
// CHECK:STDOUT:   %.loc14_25.3: init %SubscriptType.f8f = converted @__global_init.%.loc14, %.loc14_25.2 [concrete = constants.%SubscriptType.val]
// CHECK:STDOUT:   %.loc14_25.4: ref %SubscriptType.f8f = temporary %.loc14_25.1, %.loc14_25.3 [concrete = constants.%.b1d]
// CHECK:STDOUT:   %.loc14_25.5: %SubscriptType.f8f = acquire_value %.loc14_25.4 [concrete = constants.%SubscriptType.val]
// CHECK:STDOUT:   %SubscriptType.ref: type = name_ref SubscriptType, %SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %s: %SubscriptType.f8f = wrapper_binding s, %.loc14_25.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.ea5 = value_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_13.2: init %C to %.loc15_13.1 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc15_13.3: init %C = converted @__global_init.%.loc15, %.loc15_13.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc15_13.4: ref %C = temporary %.loc15_13.1, %.loc15_13.3 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc15_13.5: %C = acquire_value %.loc15_13.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc15_13.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b3 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_25.1: ref %ElementType = temporary @__global_init.%.loc16, @__global_init.%C.as.IndexWith.impl.At.call
// CHECK:STDOUT:   %.loc16_25.2: %ElementType = acquire_value %.loc16_25.1
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:   %x: %ElementType = wrapper_binding x, %.loc16_25.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.67c = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndexWith.impl: %C.ref as %.loc8_41 {
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.decl: %C.as.IndexWith.impl.At.type = fn_decl @C.as.IndexWith.impl.At [concrete = constants.%C.as.IndexWith.impl.At] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b3 = value_param_pattern [concrete = constants.%self.param_patt.303]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b3 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.49a]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.ea5 = value_param_pattern [concrete = constants.%subscript.param_patt.5ee]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.ea5 = at_binding_pattern subscript, %subscript.param_patt [concrete = constants.%subscript.patt.a6f]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.67c = out_param_pattern [concrete = constants.%return.param_patt.41c]
// CHECK:STDOUT:     %return.patt: %pattern_type.67c = return_slot_pattern %return.param_patt, %ElementType.ref [concrete = constants.%return.patt.66a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IndexWith.facet: %IndexWith.type.d74 = facet_value @C.as.IndexWith.impl.%C.ref, (constants.%IndexWith.impl_witness.6b3) [concrete = constants.%IndexWith.facet.aa3]
// CHECK:STDOUT:     %.loc9_58.1: %IndexWith.type.d74 = converted @C.as.IndexWith.impl.%C.ref, %IndexWith.facet [concrete = constants.%IndexWith.facet.aa3]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%IndexWith.impl_witness.6b3, element0 [concrete = constants.%ElementType]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, %impl.elem0 [concrete = constants.%ElementType]
// CHECK:STDOUT:     %.loc9_58.2: Core.Form = init_form %ElementType.ref [concrete = constants.%.4d0]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.IndexWith.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = wrapper_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.f8f = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %subscript: %SubscriptType.f8f = wrapper_binding subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType = out_param call_param2
// CHECK:STDOUT:     %return: ref %ElementType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C.as.IndexWith.impl.At.decl), @C.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness %IndexWith.impl_witness_table [concrete = constants.%IndexWith.impl_witness.6b3]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ElementType [concrete = constants.%ElementType]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SubscriptType = <poisoned>
// CHECK:STDOUT:   .ElementType = <poisoned>
// CHECK:STDOUT:   .At = %C.as.IndexWith.impl.At.decl
// CHECK:STDOUT:   extend %.loc8_41
// CHECK:STDOUT:   witness = %IndexWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ElementType.loc5 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.f8f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.IndexWith.impl.At(%self.param: %C, %subscript.param: %SubscriptType.f8f) -> out %return.param: %ElementType {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType to %return.param = class_init () [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc10_14: init %ElementType = converted %.loc10_13.1, %.loc10_13.2 [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   return %.loc10_14 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, file.%c
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, file.%s
// CHECK:STDOUT:   %impl.elem1: %.6e7 = impl_witness_access constants.%IndexWith.impl_witness.6b3, element1 [concrete = constants.%C.as.IndexWith.impl.At]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %.loc16: ref %ElementType = temporary_storage
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.call: init %ElementType to %.loc16 = call %bound_method(%c.ref, %s.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overloaded_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple.af4: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.d95: %tuple.type.24b = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %tuple.type.748: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %IndexWith.type.df6: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.df6 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.e15: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType) [symbolic]
// CHECK:STDOUT:   %IndexWith.type.385: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %.Self.frozen.1ee: %IndexWith.type.385 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndexWith.assoc_type.ecd: type = assoc_entity_type @IndexWith, @IndexWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.abe: %IndexWith.assoc_type.ecd = assoc_entity element0, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.420: type = facet_access_type %.Self.frozen.1ee [symbolic_self]
// CHECK:STDOUT:   %assoc0.a4ed04.2: %IndexWith.assoc_type.e15 = assoc_entity element0, imports.%Core.import_ref.6b6e53.2 [symbolic]
// CHECK:STDOUT:   %IndexWith.lookup_impl_witness.5cc: <witness> = lookup_impl_witness %.Self.frozen.1ee, @IndexWith, @IndexWith(Core.IntLiteral) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b24: type = impl_witness_access %IndexWith.lookup_impl_witness.5cc, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.70e: %IndexWith.type.385 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndexWith.lookup_impl_witness.c41: <witness> = lookup_impl_witness %.Self.70e, @IndexWith, @IndexWith(Core.IntLiteral) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7cb: type = impl_witness_access %IndexWith.lookup_impl_witness.c41, element0 [symbolic_self]
// CHECK:STDOUT:   %IndexWith_where.type.f2e: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral) where %impl.elem0.7cb = %C> [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness.10f: <witness> = impl_witness @tuple.type.as.IndexWith.impl.%IndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.8b4: type = pattern_type %tuple.type.748 [concrete]
// CHECK:STDOUT:   %self.param_patt.284: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.dd1: %pattern_type.8b4 = at_binding_pattern self, %self.param_patt.284 [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %subscript.param_patt.d18: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %subscript.patt.f21: %pattern_type.dc0 = at_binding_pattern subscript, %subscript.param_patt.d18 [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b3: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b3 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b3 = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %tuple.type.as.IndexWith.impl.At.type: type = fn_type @tuple.type.as.IndexWith.impl.At [concrete]
// CHECK:STDOUT:   %tuple.type.as.IndexWith.impl.At: %tuple.type.as.IndexWith.impl.At.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet.19e: %IndexWith.type.385 = facet_value %tuple.type.748, (%IndexWith.impl_witness.10f) [concrete]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.797: type = fn_type @IndexWith.WithSelf.At, @IndexWith.WithSelf(Core.IntLiteral, %IndexWith.facet.19e) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %empty_tuple.4db: %C = tuple_value () [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.8b4 = value_binding_pattern s [concrete]
// CHECK:STDOUT:   %tuple.adf: %tuple.type.748 = tuple_value (%empty_tuple.4db, %empty_tuple.4db) [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.98b3 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   %.af9: type = fn_type_with_self_type %IndexWith.WithSelf.At.type.797, %IndexWith.facet.19e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.df6 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.bc6: @IndexWith.WithSelf.%IndexWith.assoc_type (%IndexWith.assoc_type.e15) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.WithSelf.%assoc0 (constants.%assoc0.a4ed04.2)]
// CHECK:STDOUT:   %Core.import_ref.fc2: type = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.6b6e53.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %ElementType: type = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .s = %s
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @tuple.type.as.IndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc8_7: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc8_10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8_11.1: %tuple.type.24b = tuple_literal (%C.ref.loc8_7, %C.ref.loc8_10) [concrete = constants.%tuple.d95]
// CHECK:STDOUT:     %.loc8_11.2: type = converted %.loc8_11.1, constants.%tuple.type.748 [concrete = constants.%tuple.type.748]
// CHECK:STDOUT:     %Core.ref.loc8_16: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.df6 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %Core.ref.loc8_31: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral)> [concrete = constants.%IndexWith.type.385]
// CHECK:STDOUT:     %.Self.frozen: %IndexWith.type.385 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.1ee]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %IndexWith.type [concrete]
// CHECK:STDOUT:     %.Self.ref: %IndexWith.type.385 = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.1ee]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type.420]
// CHECK:STDOUT:     %.loc8_54.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type.420]
// CHECK:STDOUT:     %.loc8_54.2: %IndexWith.assoc_type.ecd = specific_constant imports.%Core.import_ref.bc6, @IndexWith.WithSelf(Core.IntLiteral, constants.%.Self.frozen.1ee) [concrete = constants.%assoc0.abe]
// CHECK:STDOUT:     %ElementType.ref: %IndexWith.assoc_type.ecd = name_ref ElementType, %.loc8_54.2 [concrete = constants.%assoc0.abe]
// CHECK:STDOUT:     %impl.elem0.loc8_54.1: type = impl_witness_access constants.%IndexWith.lookup_impl_witness.5cc, element0 [symbolic_self = constants.%impl.elem0.b24]
// CHECK:STDOUT:     %C.ref.loc8_69: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %rewrite.loc8_67.1 = requirement_rewrite %impl.elem0.loc8_54.1, %C.ref.loc8_69 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_54.2: type = impl_witness_access constants.%IndexWith.lookup_impl_witness.c41, element0 [symbolic_self = constants.%impl.elem0.7cb]
// CHECK:STDOUT:     %rewrite.loc8_67.2 = requirement_rewrite %impl.elem0.loc8_54.2, %C.ref.loc8_69 [concrete]
// CHECK:STDOUT:     %.loc8_48: type = where_expr [concrete = constants.%IndexWith_where.type.f2e] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %IndexWith.type [concrete]
// CHECK:STDOUT:       %rewrite.loc8_67.2 = requirement_rewrite %impl.elem0.loc8_54.2, %C.ref.loc8_69 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.loc14: %tuple.type.748 = tuple_value (@__global_init.%.loc14_21.2, @__global_init.%.loc14_30.2) [concrete = constants.%tuple.adf]
// CHECK:STDOUT:   %.loc14_34: %tuple.type.748 = converted @__global_init.%.loc14_34, %tuple.loc14 [concrete = constants.%tuple.adf]
// CHECK:STDOUT:   %.loc14_13.1: type = splice_block %.loc14_13.3 [concrete = constants.%tuple.type.748] {
// CHECK:STDOUT:     %C.ref.loc14_9: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc14_12: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc14_13.2: %tuple.type.24b = tuple_literal (%C.ref.loc14_9, %C.ref.loc14_12) [concrete = constants.%tuple.d95]
// CHECK:STDOUT:     %.loc14_13.3: type = converted %.loc14_13.2, constants.%tuple.type.748 [concrete = constants.%tuple.type.748]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: %tuple.type.748 = wrapper_binding s, %.loc14_34
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.8b4 = value_binding_pattern s [concrete = constants.%s.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_15.1: init %empty_tuple.type = as_compatible @__global_init.%tuple.type.as.IndexWith.impl.At.call
// CHECK:STDOUT:   %.loc15_15.2: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc15_15.3: ref %empty_tuple.type = temporary %.loc15_15.2, %.loc15_15.1
// CHECK:STDOUT:   %tuple.loc15: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc15_15.4: %C = as_compatible %tuple.loc15 [concrete = constants.%empty_tuple.4db]
// CHECK:STDOUT:   %.loc15_15.5: %C = converted @__global_init.%tuple.type.as.IndexWith.impl.At.call, %.loc15_15.4 [concrete = constants.%empty_tuple.4db]
// CHECK:STDOUT:   %C.ref.loc15: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e: %C = wrapper_binding e, %.loc15_15.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.98b3 = value_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @tuple.type.as.IndexWith.impl: %.loc8_11.2 as %.loc8_48 {
// CHECK:STDOUT:   %tuple.type.as.IndexWith.impl.At.decl: %tuple.type.as.IndexWith.impl.At.type = fn_decl @tuple.type.as.IndexWith.impl.At [concrete = constants.%tuple.type.as.IndexWith.impl.At] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8b4 = value_param_pattern [concrete = constants.%self.param_patt.284]
// CHECK:STDOUT:     %self.patt: %pattern_type.8b4 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.dd1]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.dc0 = value_param_pattern [concrete = constants.%subscript.param_patt.d18]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.dc0 = at_binding_pattern subscript, %subscript.param_patt [concrete = constants.%subscript.patt.f21]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b3 = out_param_pattern [concrete = constants.%return.param_patt.89a]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b3 = return_slot_pattern %return.param_patt, %C.ref [concrete = constants.%return.patt.e4a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc9_53: Core.Form = init_form %C.ref [concrete = constants.%.a44]
// CHECK:STDOUT:     %self.param: %tuple.type.748 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @tuple.type.as.IndexWith.impl.%.loc8_11.2 [concrete = constants.%tuple.type.748]
// CHECK:STDOUT:     %self: %tuple.type.748 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: Core.IntLiteral = value_param call_param1
// CHECK:STDOUT:     %.loc9_37: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %subscript: Core.IntLiteral = wrapper_binding subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %tuple.type.as.IndexWith.impl.At.decl), @tuple.type.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness %IndexWith.impl_witness_table [concrete = constants.%IndexWith.impl_witness.10f]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C [concrete = constants.%C]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .At = %tuple.type.as.IndexWith.impl.At.decl
// CHECK:STDOUT:   extend %.loc8_48
// CHECK:STDOUT:   witness = %IndexWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc5_11: type = converted %.loc5_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   adapt_decl %.loc5_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_tuple.type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @tuple.type.as.IndexWith.impl.At(%self.param: %tuple.type.748, %subscript.param: Core.IntLiteral) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %tuple.type.748 = name_ref self, %self
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: %C = tuple_access %self.ref, element0
// CHECK:STDOUT:   %.loc10_18.1: %empty_tuple.type = as_compatible %tuple.elem0
// CHECK:STDOUT:   %.loc10_18.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc10_18.3: init %C = as_compatible %.loc10_18.2 [concrete = constants.%empty_tuple.4db]
// CHECK:STDOUT:   %.loc10_18.4: init %C = converted %tuple.elem0, %.loc10_18.3 [concrete = constants.%empty_tuple.4db]
// CHECK:STDOUT:   return %.loc10_18.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_19: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %C.ref.loc14_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %empty_tuple.loc14_19: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc14_21.1: %C = as_compatible %empty_tuple.loc14_19 [concrete = constants.%empty_tuple.4db]
// CHECK:STDOUT:   %.loc14_21.2: %C = converted %.loc14_19, %.loc14_21.1 [concrete = constants.%empty_tuple.4db]
// CHECK:STDOUT:   %.loc14_28: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %C.ref.loc14_33: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %empty_tuple.loc14_28: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc14_30.1: %C = as_compatible %empty_tuple.loc14_28 [concrete = constants.%empty_tuple.4db]
// CHECK:STDOUT:   %.loc14_30.2: %C = converted %.loc14_28, %.loc14_30.1 [concrete = constants.%empty_tuple.4db]
// CHECK:STDOUT:   %.loc14_34: %tuple.type.748 = tuple_literal (%.loc14_21.2, %.loc14_30.2) [concrete = constants.%tuple.adf]
// CHECK:STDOUT:   %s.ref: %tuple.type.748 = name_ref s, file.%s
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %impl.elem1: %.af9 = impl_witness_access constants.%IndexWith.impl_witness.10f, element1 [concrete = constants.%tuple.type.as.IndexWith.impl.At]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %s.ref, %impl.elem1
// CHECK:STDOUT:   %tuple.type.as.IndexWith.impl.At.call: init %C = call %bound_method(%s.ref, %int_0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_subscript_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ElementType: type = class_type @ElementType.loc5 [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndexWith.type.df6: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.df6 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.e15: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndexWith.type.d74: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %.Self.frozen.ffd: %IndexWith.type.d74 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndexWith.assoc_type.144: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %assoc0.c96: %IndexWith.assoc_type.144 = assoc_entity element0, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.ca0: type = facet_access_type %.Self.frozen.ffd [symbolic_self]
// CHECK:STDOUT:   %assoc0.a4ed04.2: %IndexWith.assoc_type.e15 = assoc_entity element0, imports.%Core.import_ref.6b6e53.2 [symbolic]
// CHECK:STDOUT:   %IndexWith.lookup_impl_witness.a61: <witness> = lookup_impl_witness %.Self.frozen.ffd, @IndexWith, @IndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e0a: type = impl_witness_access %IndexWith.lookup_impl_witness.a61, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.1dc: %IndexWith.type.d74 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndexWith.lookup_impl_witness.2a2: <witness> = lookup_impl_witness %.Self.1dc, @IndexWith, @IndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc1: type = impl_witness_access %IndexWith.lookup_impl_witness.2a2, element0 [symbolic_self]
// CHECK:STDOUT:   %IndexWith_where.type.7d8: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.f8f) where %impl.elem0.fc1 = %ElementType> [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness.6b3: <witness> = impl_witness @C.as.IndexWith.impl.%IndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b3: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.303: %pattern_type.98b3 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.49a: %pattern_type.98b3 = at_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %pattern_type.ea5: type = pattern_type %SubscriptType.f8f [concrete]
// CHECK:STDOUT:   %subscript.param_patt.5ee: %pattern_type.ea5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %subscript.patt.a6f: %pattern_type.ea5 = at_binding_pattern subscript, %subscript.param_patt.5ee [concrete]
// CHECK:STDOUT:   %IndexWith.facet.aa3: %IndexWith.type.d74 = facet_value %C, (%IndexWith.impl_witness.6b3) [concrete]
// CHECK:STDOUT:   %.4d0: Core.Form = init_form %ElementType [concrete]
// CHECK:STDOUT:   %pattern_type.67c: type = pattern_type %ElementType [concrete]
// CHECK:STDOUT:   %return.param_patt.41c: %pattern_type.67c = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.66a: %pattern_type.67c = return_slot_pattern %return.param_patt.41c, %ElementType [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.type: type = fn_type @C.as.IndexWith.impl.At [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At: %C.as.IndexWith.impl.At.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ElementType.val: %ElementType = struct_value () [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b3 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.67c = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.df6 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.bc6: @IndexWith.WithSelf.%IndexWith.assoc_type (%IndexWith.assoc_type.e15) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.WithSelf.%assoc0 (constants.%assoc0.a4ed04.2)]
// CHECK:STDOUT:   %Core.import_ref.fc2: type = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %Core.import_ref.6b6e53.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %ElementType: type = assoc_const_decl @ElementType.1 [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ElementType = %ElementType.decl
// CHECK:STDOUT:     .SubscriptType = %SubscriptType.decl
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType.loc5 [concrete = constants.%ElementType] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [concrete = constants.%SubscriptType.f8f] {} {}
// CHECK:STDOUT:   impl_decl @C.as.IndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.df6 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndexWith.type.d74]
// CHECK:STDOUT:     %.Self.frozen: %IndexWith.type.d74 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.ffd]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %IndexWith.type [concrete]
// CHECK:STDOUT:     %.Self.ref: %IndexWith.type.d74 = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.ffd]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type.ca0]
// CHECK:STDOUT:     %.loc8_47.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type.ca0]
// CHECK:STDOUT:     %.loc8_47.2: %IndexWith.assoc_type.144 = specific_constant imports.%Core.import_ref.bc6, @IndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%.Self.frozen.ffd) [concrete = constants.%assoc0.c96]
// CHECK:STDOUT:     %ElementType.ref.loc8_47: %IndexWith.assoc_type.144 = name_ref ElementType, %.loc8_47.2 [concrete = constants.%assoc0.c96]
// CHECK:STDOUT:     %impl.elem0.loc8_47.1: type = impl_witness_access constants.%IndexWith.lookup_impl_witness.a61, element0 [symbolic_self = constants.%impl.elem0.e0a]
// CHECK:STDOUT:     %ElementType.ref.loc8_62: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:     %rewrite.loc8_60.1 = requirement_rewrite %impl.elem0.loc8_47.1, %ElementType.ref.loc8_62 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_47.2: type = impl_witness_access constants.%IndexWith.lookup_impl_witness.2a2, element0 [symbolic_self = constants.%impl.elem0.fc1]
// CHECK:STDOUT:     %rewrite.loc8_60.2 = requirement_rewrite %impl.elem0.loc8_47.2, %ElementType.ref.loc8_62 [concrete]
// CHECK:STDOUT:     %.loc8_41: type = where_expr [concrete = constants.%IndexWith_where.type.7d8] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %IndexWith.type [concrete]
// CHECK:STDOUT:       %rewrite.loc8_60.2 = requirement_rewrite %impl.elem0.loc8_47.2, %ElementType.ref.loc8_62 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc14_13.2: init %C to %.loc14_13.1 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc14_13.3: init %C = converted @__global_init.%.loc14, %.loc14_13.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc14_13.4: ref %C = temporary %.loc14_13.1, %.loc14_13.3 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc14_13.5: %C = acquire_value %.loc14_13.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc14_13.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b3 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:   %x: %ElementType = wrapper_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.67c = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndexWith.impl: %C.ref as %.loc8_41 {
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.decl: %C.as.IndexWith.impl.At.type = fn_decl @C.as.IndexWith.impl.At [concrete = constants.%C.as.IndexWith.impl.At] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b3 = value_param_pattern [concrete = constants.%self.param_patt.303]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b3 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.49a]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.ea5 = value_param_pattern [concrete = constants.%subscript.param_patt.5ee]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.ea5 = at_binding_pattern subscript, %subscript.param_patt [concrete = constants.%subscript.patt.a6f]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.67c = out_param_pattern [concrete = constants.%return.param_patt.41c]
// CHECK:STDOUT:     %return.patt: %pattern_type.67c = return_slot_pattern %return.param_patt, %ElementType.ref [concrete = constants.%return.patt.66a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IndexWith.facet: %IndexWith.type.d74 = facet_value @C.as.IndexWith.impl.%C.ref, (constants.%IndexWith.impl_witness.6b3) [concrete = constants.%IndexWith.facet.aa3]
// CHECK:STDOUT:     %.loc9_58.1: %IndexWith.type.d74 = converted @C.as.IndexWith.impl.%C.ref, %IndexWith.facet [concrete = constants.%IndexWith.facet.aa3]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%IndexWith.impl_witness.6b3, element0 [concrete = constants.%ElementType]
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, %impl.elem0 [concrete = constants.%ElementType]
// CHECK:STDOUT:     %.loc9_58.2: Core.Form = init_form %ElementType.ref [concrete = constants.%.4d0]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.IndexWith.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = wrapper_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.f8f = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %subscript: %SubscriptType.f8f = wrapper_binding subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType = out_param call_param2
// CHECK:STDOUT:     %return: ref %ElementType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C.as.IndexWith.impl.At.decl), @C.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness %IndexWith.impl_witness_table [concrete = constants.%IndexWith.impl_witness.6b3]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ElementType [concrete = constants.%ElementType]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SubscriptType = <poisoned>
// CHECK:STDOUT:   .ElementType = <poisoned>
// CHECK:STDOUT:   .At = %C.as.IndexWith.impl.At.decl
// CHECK:STDOUT:   extend %.loc8_41
// CHECK:STDOUT:   witness = %IndexWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ElementType.loc5 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ElementType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.f8f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.IndexWith.impl.At(%self.param: %C, %subscript.param: %SubscriptType.f8f) -> out %return.param: %ElementType {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_13.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_13.2: init %ElementType to %return.param = class_init () [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc10_14: init %ElementType = converted %.loc10_13.1, %.loc10_13.2 [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   return %.loc10_14 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, file.%c
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_index_with_not_implemented.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.98b3: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b3 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %IndexWith.type.df6: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.df6 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.df6 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %.loc6_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc6_13.2: init %C to %.loc6_13.1 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_13.3: init %C = converted @__global_init.%.loc6, %.loc6_13.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_13.4: ref %C = temporary %.loc6_13.1, %.loc6_13.3 [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc6_13.5: %C = acquire_value %.loc6_13.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: %C = wrapper_binding c, %.loc6_13.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b3 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: %i32 = wrapper_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, file.%c
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- indirect_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness: <witness> = impl_witness @Copyable.as.Copy.impl.%Copy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.26e: type = pattern_type %Copyable [concrete]
// CHECK:STDOUT:   %self.param_patt.0ff: %pattern_type.26e = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d83: %pattern_type.26e = at_binding_pattern self, %self.param_patt.0ff [concrete]
// CHECK:STDOUT:   %.d50: Core.Form = init_form %Copyable [concrete]
// CHECK:STDOUT:   %return.param_patt.801: %pattern_type.26e = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5e1: %pattern_type.26e = return_slot_pattern %return.param_patt.801, %Copyable [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.type: type = fn_type @Copyable.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op: %Copyable.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Copyable.val: %Copyable = struct_value () [concrete]
// CHECK:STDOUT:   %NonCopyable: type = class_type @NonCopyable [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.eec: type = ptr_type %Copyable [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %ptr.eec [concrete]
// CHECK:STDOUT:   %struct_type.ptr.650: type = struct_type {.ptr: %ptr.eec} [concrete]
// CHECK:STDOUT:   %complete_type.fc9: <witness> = complete_type_witness %struct_type.ptr.650 [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.841: type = generic_interface_type @IndirectIndexWith [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.generic: %IndirectIndexWith.type.841 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.539: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.620: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %IndirectIndexWith.type.620 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.80e: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %assoc0.c52: %IndirectIndexWith.assoc_type.80e = assoc_entity element0, imports.%Core.import_ref.abc [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %assoc0.ee9a0f.2: %IndirectIndexWith.assoc_type.539 = assoc_entity element0, imports.%Core.import_ref.55c4ab.2 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.lookup_impl_witness.92b: <witness> = lookup_impl_witness %.Self.frozen, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.25e: type = impl_witness_access %IndirectIndexWith.lookup_impl_witness.92b, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %IndirectIndexWith.type.620 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith.lookup_impl_witness.6a3: <witness> = lookup_impl_witness %.Self, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4d4: type = impl_witness_access %IndirectIndexWith.lookup_impl_witness.6a3, element0 [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.e63: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %Copyable> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness.5b4: <witness> = impl_witness @C.as.IndirectIndexWith.impl.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b3: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.303: %pattern_type.98b3 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.49a: %pattern_type.98b3 = at_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %pattern_type.ea5: type = pattern_type %SubscriptType.f8f [concrete]
// CHECK:STDOUT:   %subscript.param_patt.5ee: %pattern_type.ea5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %subscript.patt.a6f: %pattern_type.ea5 = at_binding_pattern subscript, %subscript.param_patt.5ee [concrete]
// CHECK:STDOUT:   %.ce5: Core.Form = ref_form %Copyable [concrete]
// CHECK:STDOUT:   %.a78: %pattern_type.26e = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.eb1: %pattern_type.26e = return_slot_pattern %.a78, %Copyable [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.type: type = fn_type @C.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref: %C.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT:   %CNonCopy: type = class_type @CNonCopy [concrete]
// CHECK:STDOUT:   %ptr.d10: type = ptr_type %NonCopyable [concrete]
// CHECK:STDOUT:   %CNonCopy.elem: type = unbound_element_type %CNonCopy, %ptr.d10 [concrete]
// CHECK:STDOUT:   %struct_type.ptr.ddf: type = struct_type {.ptr: %ptr.d10} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.ptr.ddf [concrete]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.acd: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %NonCopyable> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness.ae4: <witness> = impl_witness @CNonCopy.as.IndirectIndexWith.impl.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.86e: type = pattern_type %CNonCopy [concrete]
// CHECK:STDOUT:   %self.param_patt.b0f: %pattern_type.86e = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.02f: %pattern_type.86e = at_binding_pattern self, %self.param_patt.b0f [concrete]
// CHECK:STDOUT:   %.b90: Core.Form = ref_form %NonCopyable [concrete]
// CHECK:STDOUT:   %pattern_type.1959: type = pattern_type %NonCopyable [concrete]
// CHECK:STDOUT:   %.9c5: %pattern_type.1959 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.d2c: %pattern_type.1959 = return_slot_pattern %.9c5, %NonCopyable [concrete]
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref.type: type = fn_type @CNonCopy.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref: %CNonCopy.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .IndirectIndexWith = %Core.IndirectIndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.IndirectIndexWith: %IndirectIndexWith.type.841 = import_ref Core//prelude/operators/index, IndirectIndexWith, loaded [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.544: @IndirectIndexWith.WithSelf.%IndirectIndexWith.assoc_type (%IndirectIndexWith.assoc_type.539) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%assoc0 (constants.%assoc0.ee9a0f.2)]
// CHECK:STDOUT:   %Core.import_ref.abc: type = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %Core.import_ref.55c4ab.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %ElementType: type = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Copyable = %Copyable.decl
// CHECK:STDOUT:     .NonCopyable = %NonCopyable.decl
// CHECK:STDOUT:     .SubscriptType = %SubscriptType.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CNonCopy = %CNonCopy.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Copyable.decl: type = class_decl @Copyable [concrete = constants.%Copyable] {} {}
// CHECK:STDOUT:   %NonCopyable.decl: type = class_decl @NonCopyable [concrete = constants.%NonCopyable] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [concrete = constants.%SubscriptType.f8f] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.IndirectIndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndirectIndexWith.ref: %IndirectIndexWith.type.841 = name_ref IndirectIndexWith, imports.%Core.IndirectIndexWith [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %IndirectIndexWith.type: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndirectIndexWith.type.620]
// CHECK:STDOUT:     %.Self.frozen: %IndirectIndexWith.type.620 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %IndirectIndexWith.type [concrete]
// CHECK:STDOUT:     %.Self.ref: %IndirectIndexWith.type.620 = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc16_55.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc16_55.2: %IndirectIndexWith.assoc_type.80e = specific_constant imports.%Core.import_ref.544, @IndirectIndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%.Self.frozen) [concrete = constants.%assoc0.c52]
// CHECK:STDOUT:     %ElementType.ref: %IndirectIndexWith.assoc_type.80e = name_ref ElementType, %.loc16_55.2 [concrete = constants.%assoc0.c52]
// CHECK:STDOUT:     %impl.elem0.loc16_55.1: type = impl_witness_access constants.%IndirectIndexWith.lookup_impl_witness.92b, element0 [symbolic_self = constants.%impl.elem0.25e]
// CHECK:STDOUT:     %Copyable.ref: type = name_ref Copyable, file.%Copyable.decl [concrete = constants.%Copyable]
// CHECK:STDOUT:     %rewrite.loc16_68.1 = requirement_rewrite %impl.elem0.loc16_55.1, %Copyable.ref [concrete]
// CHECK:STDOUT:     %impl.elem0.loc16_55.2: type = impl_witness_access constants.%IndirectIndexWith.lookup_impl_witness.6a3, element0 [symbolic_self = constants.%impl.elem0.4d4]
// CHECK:STDOUT:     %rewrite.loc16_68.2 = requirement_rewrite %impl.elem0.loc16_55.2, %Copyable.ref [concrete]
// CHECK:STDOUT:     %.loc16_49: type = where_expr [concrete = constants.%IndirectIndexWith_where.type.e63] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %IndirectIndexWith.type [concrete]
// CHECK:STDOUT:       %rewrite.loc16_68.2 = requirement_rewrite %impl.elem0.loc16_55.2, %Copyable.ref [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CNonCopy.decl: type = class_decl @CNonCopy [concrete = constants.%CNonCopy] {} {}
// CHECK:STDOUT:   impl_decl @CNonCopy.as.IndirectIndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %CNonCopy.ref: type = name_ref CNonCopy, file.%CNonCopy.decl [concrete = constants.%CNonCopy]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndirectIndexWith.ref: %IndirectIndexWith.type.841 = name_ref IndirectIndexWith, imports.%Core.IndirectIndexWith [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %IndirectIndexWith.type: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndirectIndexWith.type.620]
// CHECK:STDOUT:     %.Self.frozen: %IndirectIndexWith.type.620 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %IndirectIndexWith.type [concrete]
// CHECK:STDOUT:     %.Self.ref: %IndirectIndexWith.type.620 = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc26_62.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc26_62.2: %IndirectIndexWith.assoc_type.80e = specific_constant imports.%Core.import_ref.544, @IndirectIndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%.Self.frozen) [concrete = constants.%assoc0.c52]
// CHECK:STDOUT:     %ElementType.ref: %IndirectIndexWith.assoc_type.80e = name_ref ElementType, %.loc26_62.2 [concrete = constants.%assoc0.c52]
// CHECK:STDOUT:     %impl.elem0.loc26_62.1: type = impl_witness_access constants.%IndirectIndexWith.lookup_impl_witness.92b, element0 [symbolic_self = constants.%impl.elem0.25e]
// CHECK:STDOUT:     %NonCopyable.ref: type = name_ref NonCopyable, file.%NonCopyable.decl [concrete = constants.%NonCopyable]
// CHECK:STDOUT:     %rewrite.loc26_75.1 = requirement_rewrite %impl.elem0.loc26_62.1, %NonCopyable.ref [concrete]
// CHECK:STDOUT:     %impl.elem0.loc26_62.2: type = impl_witness_access constants.%IndirectIndexWith.lookup_impl_witness.6a3, element0 [symbolic_self = constants.%impl.elem0.4d4]
// CHECK:STDOUT:     %rewrite.loc26_75.2 = requirement_rewrite %impl.elem0.loc26_62.2, %NonCopyable.ref [concrete]
// CHECK:STDOUT:     %.loc26_56: type = where_expr [concrete = constants.%IndirectIndexWith_where.type.acd] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %IndirectIndexWith.type [concrete]
// CHECK:STDOUT:       %rewrite.loc26_75.2 = requirement_rewrite %impl.elem0.loc26_62.2, %NonCopyable.ref [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Copyable.as.Copy.impl: %Self.ref as %Copy.ref {
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.decl: %Copyable.as.Copy.impl.Op.type = fn_decl @Copyable.as.Copy.impl.Op [concrete = constants.%Copyable.as.Copy.impl.Op] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.26e = value_param_pattern [concrete = constants.%self.param_patt.0ff]
// CHECK:STDOUT:     %self.patt: %pattern_type.26e = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d83]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.26e = out_param_pattern [concrete = constants.%return.param_patt.801]
// CHECK:STDOUT:     %return.patt: %pattern_type.26e = return_slot_pattern %return.param_patt, %Self.ref.loc6_27 [concrete = constants.%return.patt.5e1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc6_27: type = name_ref Self, constants.%Copyable [concrete = constants.%Copyable]
// CHECK:STDOUT:     %.loc6_27: Core.Form = init_form %Self.ref.loc6_27 [concrete = constants.%.d50]
// CHECK:STDOUT:     %self.param: %Copyable = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc6_18: type = name_ref Self, constants.%Copyable [concrete = constants.%Copyable]
// CHECK:STDOUT:     %self: %Copyable = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %Copyable = out_param call_param1
// CHECK:STDOUT:     %return: ref %Copyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy.impl_witness_table = impl_witness_table (%Copyable.as.Copy.impl.Op.decl), @Copyable.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Copy.impl_witness: <witness> = impl_witness %Copy.impl_witness_table [concrete = constants.%Copy.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Copyable.as.Copy.impl.Op.decl
// CHECK:STDOUT:   extend %Copy.ref
// CHECK:STDOUT:   witness = %Copy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndirectIndexWith.impl: %C.ref as %.loc16_49 {
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.decl: %C.as.IndirectIndexWith.impl.Ref.type = fn_decl @C.as.IndirectIndexWith.impl.Ref [concrete = constants.%C.as.IndirectIndexWith.impl.Ref] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b3 = value_param_pattern [concrete = constants.%self.param_patt.303]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b3 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.49a]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.ea5 = value_param_pattern [concrete = constants.%subscript.param_patt.5ee]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.ea5 = at_binding_pattern subscript, %subscript.param_patt [concrete = constants.%subscript.patt.a6f]
// CHECK:STDOUT:     %.loc17_52.1: %pattern_type.26e = ref_return_pattern [concrete = constants.%.a78]
// CHECK:STDOUT:     %return.patt: %pattern_type.26e = return_slot_pattern %.loc17_52.1, %Copyable.ref [concrete = constants.%return.patt.eb1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Copyable.ref: type = name_ref Copyable, file.%Copyable.decl [concrete = constants.%Copyable]
// CHECK:STDOUT:     %.loc17_52.2: type = ref_tag %Copyable.ref
// CHECK:STDOUT:     %.loc17_52.3: Core.Form = ref_form %Copyable.ref [concrete = constants.%.ce5]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.IndirectIndexWith.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = wrapper_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.f8f = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %subscript: %SubscriptType.f8f = wrapper_binding subscript, %subscript.param
// CHECK:STDOUT:     %.loc17_52.4: ref %Copyable = ref_return
// CHECK:STDOUT:     %return: ref %Copyable = return_slot %.loc17_52.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C.as.IndirectIndexWith.impl.Ref.decl), @C.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness %IndirectIndexWith.impl_witness_table [concrete = constants.%IndirectIndexWith.impl_witness.5b4]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%Copyable [concrete = constants.%Copyable]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SubscriptType = <poisoned>
// CHECK:STDOUT:   .Copyable = <poisoned>
// CHECK:STDOUT:   .Ref = %C.as.IndirectIndexWith.impl.Ref.decl
// CHECK:STDOUT:   extend %.loc16_49
// CHECK:STDOUT:   witness = %IndirectIndexWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @CNonCopy.as.IndirectIndexWith.impl: %CNonCopy.ref as %.loc26_56 {
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref.decl: %CNonCopy.as.IndirectIndexWith.impl.Ref.type = fn_decl @CNonCopy.as.IndirectIndexWith.impl.Ref [concrete = constants.%CNonCopy.as.IndirectIndexWith.impl.Ref] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.86e = value_param_pattern [concrete = constants.%self.param_patt.b0f]
// CHECK:STDOUT:     %self.patt: %pattern_type.86e = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.02f]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.ea5 = value_param_pattern [concrete = constants.%subscript.param_patt.5ee]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.ea5 = at_binding_pattern subscript, %subscript.param_patt [concrete = constants.%subscript.patt.a6f]
// CHECK:STDOUT:     %.loc27_52.1: %pattern_type.1959 = ref_return_pattern [concrete = constants.%.9c5]
// CHECK:STDOUT:     %return.patt: %pattern_type.1959 = return_slot_pattern %.loc27_52.1, %NonCopyable.ref [concrete = constants.%return.patt.d2c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NonCopyable.ref: type = name_ref NonCopyable, file.%NonCopyable.decl [concrete = constants.%NonCopyable]
// CHECK:STDOUT:     %.loc27_52.2: type = ref_tag %NonCopyable.ref
// CHECK:STDOUT:     %.loc27_52.3: Core.Form = ref_form %NonCopyable.ref [concrete = constants.%.b90]
// CHECK:STDOUT:     %self.param: %CNonCopy = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @CNonCopy.as.IndirectIndexWith.impl.%CNonCopy.ref [concrete = constants.%CNonCopy]
// CHECK:STDOUT:     %self: %CNonCopy = wrapper_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.f8f = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %subscript: %SubscriptType.f8f = wrapper_binding subscript, %subscript.param
// CHECK:STDOUT:     %.loc27_52.4: ref %NonCopyable = ref_return
// CHECK:STDOUT:     %return: ref %NonCopyable = return_slot %.loc27_52.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %CNonCopy.as.IndirectIndexWith.impl.Ref.decl), @CNonCopy.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness %IndirectIndexWith.impl_witness_table [concrete = constants.%IndirectIndexWith.impl_witness.ae4]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%NonCopyable [concrete = constants.%NonCopyable]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SubscriptType = <poisoned>
// CHECK:STDOUT:   .NonCopyable = <poisoned>
// CHECK:STDOUT:   .Ref = %CNonCopy.as.IndirectIndexWith.impl.Ref.decl
// CHECK:STDOUT:   extend %.loc26_56
// CHECK:STDOUT:   witness = %IndirectIndexWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Copyable {
// CHECK:STDOUT:   impl_decl @Copyable.as.Copy.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Copyable [concrete = constants.%Copyable]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Copyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonCopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NonCopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SubscriptType.f8f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc13: %C.elem = field_decl ptr, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.ptr.650 [concrete = constants.%complete_type.fc9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Copyable = <poisoned>
// CHECK:STDOUT:   .ptr = %.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CNonCopy {
// CHECK:STDOUT:   %.loc23: %CNonCopy.elem = field_decl ptr, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.ptr.ddf [concrete = constants.%complete_type.d5e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CNonCopy
// CHECK:STDOUT:   .NonCopyable = <poisoned>
// CHECK:STDOUT:   .ptr = %.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Copyable.as.Copy.impl.Op(%self.param: %Copyable) -> out %return.param: %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_42.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_42.2: init %Copyable to %return.param = class_init () [concrete = constants.%Copyable.val]
// CHECK:STDOUT:   %.loc6_43: init %Copyable = converted %.loc6_42.1, %.loc6_42.2 [concrete = constants.%Copyable.val]
// CHECK:STDOUT:   return %.loc6_43 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.IndirectIndexWith.impl.Ref(%self.param: %C, %subscript.param: %SubscriptType.f8f) -> ref %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   %ptr.ref: %C.elem = name_ref ptr, @C.%.loc13 [concrete = @C.%.loc13]
// CHECK:STDOUT:   %.loc18_17.1: ref %ptr.eec = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc18_17.2: %ptr.eec = acquire_value %.loc18_17.1
// CHECK:STDOUT:   %.loc18_12: ref %Copyable = deref %.loc18_17.2
// CHECK:STDOUT:   return %.loc18_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CNonCopy.as.IndirectIndexWith.impl.Ref(%self.param: %CNonCopy, %subscript.param: %SubscriptType.f8f) -> ref %NonCopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %CNonCopy = name_ref self, %self
// CHECK:STDOUT:   %ptr.ref: %CNonCopy.elem = name_ref ptr, @CNonCopy.%.loc23 [concrete = @CNonCopy.%.loc23]
// CHECK:STDOUT:   %.loc28_17.1: ref %ptr.d10 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc28_17.2: %ptr.d10 = acquire_value %.loc28_17.1
// CHECK:STDOUT:   %.loc28_12: ref %NonCopyable = deref %.loc28_17.2
// CHECK:STDOUT:   return %.loc28_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- direct_ref_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.eec: type = ptr_type %Copyable [concrete]
// CHECK:STDOUT:   %struct_type.ptr.650: type = struct_type {.ptr: %ptr.eec} [concrete]
// CHECK:STDOUT:   %complete_type.fc9: <witness> = complete_type_witness %struct_type.ptr.650 [concrete]
// CHECK:STDOUT:   %pattern_type.98b3: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.98b3 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b3 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %pattern_type.ea5: type = pattern_type %SubscriptType.f8f [concrete]
// CHECK:STDOUT:   %s.param_patt: %pattern_type.ea5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.ea5 = at_binding_pattern s, %s.param_patt [concrete]
// CHECK:STDOUT:   %.ce5: Core.Form = ref_form %Copyable [concrete]
// CHECK:STDOUT:   %pattern_type.26e: type = pattern_type %Copyable [concrete]
// CHECK:STDOUT:   %.a78: %pattern_type.26e = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.eb1: %pattern_type.26e = return_slot_pattern %.a78, %Copyable [concrete]
// CHECK:STDOUT:   %TestDirectRef.type: type = fn_type @TestDirectRef [concrete]
// CHECK:STDOUT:   %TestDirectRef: %TestDirectRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.841: type = generic_interface_type @IndirectIndexWith [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.generic: %IndirectIndexWith.type.841 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.f6a: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d)> [symbolic]
// CHECK:STDOUT:   %Self.da6: %IndirectIndexWith.type.f6a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.539: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.9a0: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.67d, %Self.da6) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.1d2: %IndirectIndexWith.WithSelf.Ref.type.9a0 = struct_value () [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.620: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.80e: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %assoc1.481: %IndirectIndexWith.assoc_type.80e = assoc_entity element1, imports.%Core.import_ref.2b2 [concrete]
// CHECK:STDOUT:   %assoc1.23c0d9.2: %IndirectIndexWith.assoc_type.539 = assoc_entity element1, imports.%Core.import_ref.079aa7.2 [symbolic]
// CHECK:STDOUT:   %.Self: %IndirectIndexWith.type.620 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith.lookup_impl_witness.6a3: <witness> = lookup_impl_witness %.Self, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4d4: type = impl_witness_access %IndirectIndexWith.lookup_impl_witness.6a3, element0 [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.e63: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %Copyable> [concrete]
// CHECK:STDOUT:   %NonCopyable: type = class_type @NonCopyable [concrete]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.acd: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %NonCopyable> [concrete]
// CHECK:STDOUT:   %CNonCopy: type = class_type @CNonCopy [concrete]
// CHECK:STDOUT:   %ptr.d10: type = ptr_type %NonCopyable [concrete]
// CHECK:STDOUT:   %struct_type.ptr.ddf: type = struct_type {.ptr: %ptr.d10} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.ptr.ddf [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.facet: %IndirectIndexWith.type.620 = facet_value %C, (%IndirectIndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.ac4: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.f8f, %IndirectIndexWith.facet) [concrete]
// CHECK:STDOUT:   %.f83: type = fn_type_with_self_type %IndirectIndexWith.WithSelf.Ref.type.ac4, %IndirectIndexWith.facet [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.type: type = fn_type @C.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref: %C.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Copyable: type = import_ref Main//indirect_types, Copyable, loaded [concrete = constants.%Copyable]
// CHECK:STDOUT:   %Main.NonCopyable = import_ref Main//indirect_types, NonCopyable, unloaded
// CHECK:STDOUT:   %Main.SubscriptType: type = import_ref Main//indirect_types, SubscriptType, loaded [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//indirect_types, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.CNonCopy = import_ref Main//indirect_types, CNonCopy, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndirectIndexWith = %Core.IndirectIndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//indirect_types, loc8_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.cdb = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5c: <witness> = import_ref Main//indirect_types, loc14_1, loaded [concrete = constants.%complete_type.fc9]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.fcf = import_ref Main//indirect_types, loc13_10, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//indirect_types, loc10_22, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.bd5 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.IndirectIndexWith: %IndirectIndexWith.type.841 = import_ref Core//prelude/operators/index, IndirectIndexWith, loaded [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.ed6: @IndirectIndexWith.WithSelf.%IndirectIndexWith.assoc_type (%IndirectIndexWith.assoc_type.539) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%assoc1 (constants.%assoc1.23c0d9.2)]
// CHECK:STDOUT:   %Core.import_ref.2b2: @IndirectIndexWith.WithSelf.%IndirectIndexWith.WithSelf.Ref.type (%IndirectIndexWith.WithSelf.Ref.type.9a0) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%IndirectIndexWith.WithSelf.Ref (constants.%IndirectIndexWith.WithSelf.Ref.1d2)]
// CHECK:STDOUT:   %Core.import_ref.079aa7.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Main.import_ref.ff6: <witness> = import_ref Main//indirect_types, loc16_79, loaded [concrete = constants.%IndirectIndexWith.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.335: type = import_ref Main//indirect_types, loc16_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.0e1: type = import_ref Main//indirect_types, loc16_49, loaded [concrete = constants.%IndirectIndexWith_where.type.e63]
// CHECK:STDOUT:   %Main.import_ref.949 = import_ref Main//indirect_types, loc26_89, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//indirect_types, loc9_20, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.283 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.61c: <witness> = import_ref Main//indirect_types, loc24_1, loaded [concrete = constants.%complete_type.d5e]
// CHECK:STDOUT:   %Main.import_ref.5e5 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8fd = import_ref Main//indirect_types, loc23_10, unloaded
// CHECK:STDOUT:   %Main.import_ref.db0: type = import_ref Main//indirect_types, loc26_6, loaded [concrete = constants.%CNonCopy]
// CHECK:STDOUT:   %Main.import_ref.47a: type = import_ref Main//indirect_types, loc26_56, loaded [concrete = constants.%IndirectIndexWith_where.type.acd]
// CHECK:STDOUT:   %Main.import_ref.bdc = import_ref Main//indirect_types, loc16_79, unloaded
// CHECK:STDOUT:   %Main.import_ref.2b0: %C.as.IndirectIndexWith.impl.Ref.type = import_ref Main//indirect_types, loc17_65, loaded [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.bdc, %Main.import_ref.2b0), @C.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Copyable = imports.%Main.Copyable
// CHECK:STDOUT:     .NonCopyable = imports.%Main.NonCopyable
// CHECK:STDOUT:     .SubscriptType = imports.%Main.SubscriptType
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .CNonCopy = imports.%Main.CNonCopy
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TestDirectRef = %TestDirectRef.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %TestDirectRef.decl: %TestDirectRef.type = fn_decl @TestDirectRef [concrete = constants.%TestDirectRef] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.98b3 = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.98b3 = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.ea5 = value_param_pattern [concrete = constants.%s.param_patt]
// CHECK:STDOUT:     %s.patt: %pattern_type.ea5 = at_binding_pattern s, %s.param_patt [concrete = constants.%s.patt]
// CHECK:STDOUT:     %.loc6_45.1: %pattern_type.26e = ref_return_pattern [concrete = constants.%.a78]
// CHECK:STDOUT:     %return.patt: %pattern_type.26e = return_slot_pattern %.loc6_45.1, %Copyable.ref [concrete = constants.%return.patt.eb1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Copyable.ref: type = name_ref Copyable, imports.%Main.Copyable [concrete = constants.%Copyable]
// CHECK:STDOUT:     %.loc6_45.2: type = ref_tag %Copyable.ref
// CHECK:STDOUT:     %.loc6_45.3: Core.Form = ref_form %Copyable.ref [concrete = constants.%.ce5]
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = wrapper_binding c, %c.param
// CHECK:STDOUT:     %s.param: %SubscriptType.f8f = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref.loc6: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %s: %SubscriptType.f8f = wrapper_binding s, %s.param
// CHECK:STDOUT:     %.loc6_45.4: ref %Copyable = ref_return
// CHECK:STDOUT:     %return: ref %Copyable = return_slot %.loc6_45.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndirectIndexWith.impl: imports.%Main.import_ref.335 as imports.%Main.import_ref.0e1 [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ff6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @CNonCopy.as.IndirectIndexWith.impl: imports.%Main.import_ref.db0 as imports.%Main.import_ref.47a [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.949
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.b5c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT:   .ptr = imports.%Main.import_ref.fcf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Copyable [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.cdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonCopyable [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CNonCopy [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.61c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.5e5
// CHECK:STDOUT:   .ptr = imports.%Main.import_ref.8fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDirectRef(%c.param: %C, %s.param: %SubscriptType.f8f) -> ref %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %IndirectIndexWith.ref: %IndirectIndexWith.type.841 = name_ref IndirectIndexWith, imports.%Core.IndirectIndexWith [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %SubscriptType.ref.loc7: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %IndirectIndexWith.type: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndirectIndexWith.type.620]
// CHECK:STDOUT:   %.loc7: %IndirectIndexWith.assoc_type.80e = specific_constant imports.%Core.import_ref.ed6, @IndirectIndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%Self.da6) [concrete = constants.%assoc1.481]
// CHECK:STDOUT:   %Ref.ref: %IndirectIndexWith.assoc_type.80e = name_ref Ref, %.loc7 [concrete = constants.%assoc1.481]
// CHECK:STDOUT:   %impl.elem1: %.f83 = impl_witness_access constants.%IndirectIndexWith.impl_witness, element1 [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.call: ref %Copyable = call %bound_method(%c.ref, %s.ref)
// CHECK:STDOUT:   return %C.as.IndirectIndexWith.impl.Ref.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.IndirectIndexWith.impl.Ref [from "indirect_types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- direct_ref_non_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CNonCopy: type = class_type @CNonCopy [concrete]
// CHECK:STDOUT:   %NonCopyable: type = class_type @NonCopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.d10: type = ptr_type %NonCopyable [concrete]
// CHECK:STDOUT:   %struct_type.ptr.ddf: type = struct_type {.ptr: %ptr.d10} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.ptr.ddf [concrete]
// CHECK:STDOUT:   %pattern_type.86e: type = pattern_type %CNonCopy [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.86e = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.86e = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %pattern_type.ea5: type = pattern_type %SubscriptType.f8f [concrete]
// CHECK:STDOUT:   %s.param_patt: %pattern_type.ea5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.ea5 = at_binding_pattern s, %s.param_patt [concrete]
// CHECK:STDOUT:   %.b90: Core.Form = ref_form %NonCopyable [concrete]
// CHECK:STDOUT:   %pattern_type.1959: type = pattern_type %NonCopyable [concrete]
// CHECK:STDOUT:   %.9c5: %pattern_type.1959 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.d2c: %pattern_type.1959 = return_slot_pattern %.9c5, %NonCopyable [concrete]
// CHECK:STDOUT:   %TestDirectRefNonCopy.type: type = fn_type @TestDirectRefNonCopy [concrete]
// CHECK:STDOUT:   %TestDirectRefNonCopy: %TestDirectRefNonCopy.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.841: type = generic_interface_type @IndirectIndexWith [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.generic: %IndirectIndexWith.type.841 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.f6a: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d)> [symbolic]
// CHECK:STDOUT:   %Self.da6: %IndirectIndexWith.type.f6a = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.539: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.9a0: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.67d, %Self.da6) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.1d2: %IndirectIndexWith.WithSelf.Ref.type.9a0 = struct_value () [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.620: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.80e: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %assoc1.481: %IndirectIndexWith.assoc_type.80e = assoc_entity element1, imports.%Core.import_ref.2b2 [concrete]
// CHECK:STDOUT:   %assoc1.23c0d9.2: %IndirectIndexWith.assoc_type.539 = assoc_entity element1, imports.%Core.import_ref.079aa7.2 [symbolic]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %.Self: %IndirectIndexWith.type.620 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith.lookup_impl_witness.6a3: <witness> = lookup_impl_witness %.Self, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4d4: type = impl_witness_access %IndirectIndexWith.lookup_impl_witness.6a3, element0 [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.e63: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %Copyable> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.eec: type = ptr_type %Copyable [concrete]
// CHECK:STDOUT:   %struct_type.ptr.650: type = struct_type {.ptr: %ptr.eec} [concrete]
// CHECK:STDOUT:   %complete_type.fc9: <witness> = complete_type_witness %struct_type.ptr.650 [concrete]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.acd: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %NonCopyable> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.facet: %IndirectIndexWith.type.620 = facet_value %CNonCopy, (%IndirectIndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.17a: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.f8f, %IndirectIndexWith.facet) [concrete]
// CHECK:STDOUT:   %.5e7: type = fn_type_with_self_type %IndirectIndexWith.WithSelf.Ref.type.17a, %IndirectIndexWith.facet [concrete]
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref.type: type = fn_type @CNonCopy.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref: %CNonCopy.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Copyable = import_ref Main//indirect_types, Copyable, unloaded
// CHECK:STDOUT:   %Main.NonCopyable: type = import_ref Main//indirect_types, NonCopyable, loaded [concrete = constants.%NonCopyable]
// CHECK:STDOUT:   %Main.SubscriptType: type = import_ref Main//indirect_types, SubscriptType, loaded [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %Main.C = import_ref Main//indirect_types, C, unloaded
// CHECK:STDOUT:   %Main.CNonCopy: type = import_ref Main//indirect_types, CNonCopy, loaded [concrete = constants.%CNonCopy]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndirectIndexWith = %Core.IndirectIndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//indirect_types, loc9_20, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.283 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.61c: <witness> = import_ref Main//indirect_types, loc24_1, loaded [concrete = constants.%complete_type.d5e]
// CHECK:STDOUT:   %Main.import_ref.5e5 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8fd = import_ref Main//indirect_types, loc23_10, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//indirect_types, loc10_22, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.bd5 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.IndirectIndexWith: %IndirectIndexWith.type.841 = import_ref Core//prelude/operators/index, IndirectIndexWith, loaded [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.ed6: @IndirectIndexWith.WithSelf.%IndirectIndexWith.assoc_type (%IndirectIndexWith.assoc_type.539) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%assoc1 (constants.%assoc1.23c0d9.2)]
// CHECK:STDOUT:   %Core.import_ref.2b2: @IndirectIndexWith.WithSelf.%IndirectIndexWith.WithSelf.Ref.type (%IndirectIndexWith.WithSelf.Ref.type.9a0) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%IndirectIndexWith.WithSelf.Ref (constants.%IndirectIndexWith.WithSelf.Ref.1d2)]
// CHECK:STDOUT:   %Core.import_ref.079aa7.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Main.import_ref.ca3 = import_ref Main//indirect_types, loc16_79, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//indirect_types, loc8_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.cdb = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5c: <witness> = import_ref Main//indirect_types, loc14_1, loaded [concrete = constants.%complete_type.fc9]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.fcf = import_ref Main//indirect_types, loc13_10, unloaded
// CHECK:STDOUT:   %Main.import_ref.335: type = import_ref Main//indirect_types, loc16_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.0e1: type = import_ref Main//indirect_types, loc16_49, loaded [concrete = constants.%IndirectIndexWith_where.type.e63]
// CHECK:STDOUT:   %Main.import_ref.6d1: <witness> = import_ref Main//indirect_types, loc26_89, loaded [concrete = constants.%IndirectIndexWith.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.db0: type = import_ref Main//indirect_types, loc26_6, loaded [concrete = constants.%CNonCopy]
// CHECK:STDOUT:   %Main.import_ref.47a: type = import_ref Main//indirect_types, loc26_56, loaded [concrete = constants.%IndirectIndexWith_where.type.acd]
// CHECK:STDOUT:   %Main.import_ref.e33 = import_ref Main//indirect_types, loc26_89, unloaded
// CHECK:STDOUT:   %Main.import_ref.2b0: %CNonCopy.as.IndirectIndexWith.impl.Ref.type = import_ref Main//indirect_types, loc27_68, loaded [concrete = constants.%CNonCopy.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.e33, %Main.import_ref.2b0), @CNonCopy.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Copyable = imports.%Main.Copyable
// CHECK:STDOUT:     .NonCopyable = imports.%Main.NonCopyable
// CHECK:STDOUT:     .SubscriptType = imports.%Main.SubscriptType
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .CNonCopy = imports.%Main.CNonCopy
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TestDirectRefNonCopy = %TestDirectRefNonCopy.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %TestDirectRefNonCopy.decl: %TestDirectRefNonCopy.type = fn_decl @TestDirectRefNonCopy [concrete = constants.%TestDirectRefNonCopy] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.86e = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.86e = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.ea5 = value_param_pattern [concrete = constants.%s.param_patt]
// CHECK:STDOUT:     %s.patt: %pattern_type.ea5 = at_binding_pattern s, %s.param_patt [concrete = constants.%s.patt]
// CHECK:STDOUT:     %.loc6_59.1: %pattern_type.1959 = ref_return_pattern [concrete = constants.%.9c5]
// CHECK:STDOUT:     %return.patt: %pattern_type.1959 = return_slot_pattern %.loc6_59.1, %NonCopyable.ref [concrete = constants.%return.patt.d2c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NonCopyable.ref: type = name_ref NonCopyable, imports.%Main.NonCopyable [concrete = constants.%NonCopyable]
// CHECK:STDOUT:     %.loc6_59.2: type = ref_tag %NonCopyable.ref
// CHECK:STDOUT:     %.loc6_59.3: Core.Form = ref_form %NonCopyable.ref [concrete = constants.%.b90]
// CHECK:STDOUT:     %c.param: %CNonCopy = value_param call_param0
// CHECK:STDOUT:     %CNonCopy.ref: type = name_ref CNonCopy, imports.%Main.CNonCopy [concrete = constants.%CNonCopy]
// CHECK:STDOUT:     %c: %CNonCopy = wrapper_binding c, %c.param
// CHECK:STDOUT:     %s.param: %SubscriptType.f8f = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref.loc6: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %s: %SubscriptType.f8f = wrapper_binding s, %s.param
// CHECK:STDOUT:     %.loc6_59.4: ref %NonCopyable = ref_return
// CHECK:STDOUT:     %return: ref %NonCopyable = return_slot %.loc6_59.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndirectIndexWith.impl: imports.%Main.import_ref.335 as imports.%Main.import_ref.0e1 [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ca3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @CNonCopy.as.IndirectIndexWith.impl: imports.%Main.import_ref.db0 as imports.%Main.import_ref.47a [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.6d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CNonCopy [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.61c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.5e5
// CHECK:STDOUT:   .ptr = imports.%Main.import_ref.8fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonCopyable [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Copyable [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.cdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.b5c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT:   .ptr = imports.%Main.import_ref.fcf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestDirectRefNonCopy(%c.param: %CNonCopy, %s.param: %SubscriptType.f8f) -> ref %NonCopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %CNonCopy = name_ref c, %c
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %IndirectIndexWith.ref: %IndirectIndexWith.type.841 = name_ref IndirectIndexWith, imports.%Core.IndirectIndexWith [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %SubscriptType.ref.loc7: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %IndirectIndexWith.type: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndirectIndexWith.type.620]
// CHECK:STDOUT:   %.loc7: %IndirectIndexWith.assoc_type.80e = specific_constant imports.%Core.import_ref.ed6, @IndirectIndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%Self.da6) [concrete = constants.%assoc1.481]
// CHECK:STDOUT:   %Ref.ref: %IndirectIndexWith.assoc_type.80e = name_ref Ref, %.loc7 [concrete = constants.%assoc1.481]
// CHECK:STDOUT:   %impl.elem1: %.5e7 = impl_witness_access constants.%IndirectIndexWith.impl_witness, element1 [concrete = constants.%CNonCopy.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref.call: ref %NonCopyable = call %bound_method(%c.ref, %s.ref)
// CHECK:STDOUT:   return %CNonCopy.as.IndirectIndexWith.impl.Ref.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CNonCopy.as.IndirectIndexWith.impl.Ref [from "indirect_types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- indirect_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.eec: type = ptr_type %Copyable [concrete]
// CHECK:STDOUT:   %struct_type.ptr.650: type = struct_type {.ptr: %ptr.eec} [concrete]
// CHECK:STDOUT:   %complete_type.fc9: <witness> = complete_type_witness %struct_type.ptr.650 [concrete]
// CHECK:STDOUT:   %pattern_type.98b3: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.98b3 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b3 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %pattern_type.ea5: type = pattern_type %SubscriptType.f8f [concrete]
// CHECK:STDOUT:   %s.param_patt: %pattern_type.ea5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.ea5 = at_binding_pattern s, %s.param_patt [concrete]
// CHECK:STDOUT:   %.d50: Core.Form = init_form %Copyable [concrete]
// CHECK:STDOUT:   %pattern_type.26e: type = pattern_type %Copyable [concrete]
// CHECK:STDOUT:   %return.param_patt.801: %pattern_type.26e = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5e1: %pattern_type.26e = return_slot_pattern %return.param_patt.801, %Copyable [concrete]
// CHECK:STDOUT:   %TestSubscript.type: type = fn_type @TestSubscript [concrete]
// CHECK:STDOUT:   %TestSubscript: %TestSubscript.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.df6: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.df6 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.d74: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.f6a: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d)> [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %.Self.ba6: %IndirectIndexWith.type.f6a = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.lookup_impl_witness.191: <witness> = lookup_impl_witness %.Self.ba6, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.81f: type = impl_witness_access %IndirectIndexWith.lookup_impl_witness.191, element0 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.f6a78c.1: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) where %impl.elem0.81f impls @Copy> [symbolic]
// CHECK:STDOUT:   %T.da6: %IndirectIndexWith_where.type.f6a78c.1 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.type.d25: type = fn_type @T.as_type.as.IndexWith.impl.At, @T.as_type.as.IndexWith.impl(%SubscriptType.67d, %T.da6) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.b0a: %T.as_type.as.IndexWith.impl.At.type.d25 = struct_value () [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.620: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %.Self.c48: %IndirectIndexWith.type.620 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith.lookup_impl_witness.6a3: <witness> = lookup_impl_witness %.Self.c48, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4d4: type = impl_witness_access %IndirectIndexWith.lookup_impl_witness.6a3, element0 [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.e63: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %Copyable> [concrete]
// CHECK:STDOUT:   %NonCopyable: type = class_type @NonCopyable [concrete]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.acd: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %NonCopyable> [concrete]
// CHECK:STDOUT:   %CNonCopy: type = class_type @CNonCopy [concrete]
// CHECK:STDOUT:   %ptr.d10: type = ptr_type %NonCopyable [concrete]
// CHECK:STDOUT:   %struct_type.ptr.ddf: type = struct_type {.ptr: %ptr.d10} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.ptr.ddf [concrete]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.620bc9.1: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 impls @Copy> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.16d: <witness> = impl_witness imports.%Copy.impl_witness_table.7e1 [concrete]
// CHECK:STDOUT:   %facet_value: %IndirectIndexWith_where.type.620bc9.1 = facet_value %C, (%IndirectIndexWith.impl_witness, %Copy.impl_witness.16d) [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness.efa: <witness> = impl_witness imports.%IndexWith.impl_witness_table, @T.as_type.as.IndexWith.impl(%SubscriptType.f8f, %facet_value) [concrete]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.type.f26: type = fn_type @T.as_type.as.IndexWith.impl.At, @T.as_type.as.IndexWith.impl(%SubscriptType.f8f, %facet_value) [concrete]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.ea7: %T.as_type.as.IndexWith.impl.At.type.f26 = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet.213: %IndexWith.type.d74 = facet_value %C, (%IndexWith.impl_witness.efa) [concrete]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.c44: type = fn_type @IndexWith.WithSelf.At, @IndexWith.WithSelf(%SubscriptType.f8f, %IndexWith.facet.213) [concrete]
// CHECK:STDOUT:   %.c04: type = fn_type_with_self_type %IndexWith.WithSelf.At.type.c44, %IndexWith.facet.213 [concrete]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.specific_fn: <specific function> = specific_function %T.as_type.as.IndexWith.impl.At.ea7, @T.as_type.as.IndexWith.impl.At(%SubscriptType.f8f, %facet_value) [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.type: type = fn_type @C.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref: %C.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.type: type = fn_type @Copyable.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op: %Copyable.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Copyable: type = import_ref Main//indirect_types, Copyable, loaded [concrete = constants.%Copyable]
// CHECK:STDOUT:   %Main.NonCopyable = import_ref Main//indirect_types, NonCopyable, unloaded
// CHECK:STDOUT:   %Main.SubscriptType: type = import_ref Main//indirect_types, SubscriptType, loaded [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %Main.C: type = import_ref Main//indirect_types, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.CNonCopy = import_ref Main//indirect_types, CNonCopy, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//indirect_types, loc8_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.cdb = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5c: <witness> = import_ref Main//indirect_types, loc14_1, loaded [concrete = constants.%complete_type.fc9]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.fcf = import_ref Main//indirect_types, loc13_10, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//indirect_types, loc10_22, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.bd5 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.df6 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.4ca = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.409: @T.as_type.as.IndexWith.impl.%T.as_type.as.IndexWith.impl.At.type (%T.as_type.as.IndexWith.impl.At.type.d25) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.IndexWith.impl.%T.as_type.as.IndexWith.impl.At (constants.%T.as_type.as.IndexWith.impl.At.b0a)]
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%Core.import_ref.4ca, %Core.import_ref.409), @T.as_type.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.ff6: <witness> = import_ref Main//indirect_types, loc16_79, loaded [concrete = constants.%IndirectIndexWith.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.335: type = import_ref Main//indirect_types, loc16_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.0e1: type = import_ref Main//indirect_types, loc16_49, loaded [concrete = constants.%IndirectIndexWith_where.type.e63]
// CHECK:STDOUT:   %Main.import_ref.949 = import_ref Main//indirect_types, loc26_89, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//indirect_types, loc9_20, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.283 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.61c: <witness> = import_ref Main//indirect_types, loc24_1, loaded [concrete = constants.%complete_type.d5e]
// CHECK:STDOUT:   %Main.import_ref.5e5 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8fd = import_ref Main//indirect_types, loc23_10, unloaded
// CHECK:STDOUT:   %Main.import_ref.db0: type = import_ref Main//indirect_types, loc26_6, loaded [concrete = constants.%CNonCopy]
// CHECK:STDOUT:   %Main.import_ref.47a: type = import_ref Main//indirect_types, loc26_56, loaded [concrete = constants.%IndirectIndexWith_where.type.acd]
// CHECK:STDOUT:   %Main.import_ref.967: type = import_ref Main//indirect_types, loc16_79, loaded [concrete = constants.%Copyable]
// CHECK:STDOUT:   %Main.import_ref.2b0: %C.as.IndirectIndexWith.impl.Ref.type = import_ref Main//indirect_types, loc17_65, loaded [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.967, %Main.import_ref.2b0), @C.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.dda: <witness> = import_ref Main//indirect_types, loc5_21, loaded [concrete = constants.%Copy.impl_witness.16d]
// CHECK:STDOUT:   %Main.import_ref.300c2c.1: type = import_ref Main//indirect_types, loc5_8, loaded [concrete = constants.%Copyable]
// CHECK:STDOUT:   %Main.import_ref.7f5: type = import_ref Main//indirect_types, loc5_15, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Main.import_ref.b8e: %Copyable.as.Copy.impl.Op.type = import_ref Main//indirect_types, loc6_32, loaded [concrete = constants.%Copyable.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.7e1 = impl_witness_table (%Main.import_ref.b8e), @Copyable.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Copyable = imports.%Main.Copyable
// CHECK:STDOUT:     .NonCopyable = imports.%Main.NonCopyable
// CHECK:STDOUT:     .SubscriptType = imports.%Main.SubscriptType
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .CNonCopy = imports.%Main.CNonCopy
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TestSubscript = %TestSubscript.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %TestSubscript.decl: %TestSubscript.type = fn_decl @TestSubscript [concrete = constants.%TestSubscript] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.98b3 = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.98b3 = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.ea5 = value_param_pattern [concrete = constants.%s.param_patt]
// CHECK:STDOUT:     %s.patt: %pattern_type.ea5 = at_binding_pattern s, %s.param_patt [concrete = constants.%s.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.26e = out_param_pattern [concrete = constants.%return.param_patt.801]
// CHECK:STDOUT:     %return.patt: %pattern_type.26e = return_slot_pattern %return.param_patt, %Copyable.ref [concrete = constants.%return.patt.5e1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Copyable.ref: type = name_ref Copyable, imports.%Main.Copyable [concrete = constants.%Copyable]
// CHECK:STDOUT:     %.loc6_45.2: Core.Form = init_form %Copyable.ref [concrete = constants.%.d50]
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = wrapper_binding c, %c.param
// CHECK:STDOUT:     %s.param: %SubscriptType.f8f = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %s: %SubscriptType.f8f = wrapper_binding s, %s.param
// CHECK:STDOUT:     %return.param: ref %Copyable = out_param call_param2
// CHECK:STDOUT:     %return: ref %Copyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndirectIndexWith.impl: imports.%Main.import_ref.335 as imports.%Main.import_ref.0e1 [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ff6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @CNonCopy.as.IndirectIndexWith.impl: imports.%Main.import_ref.db0 as imports.%Main.import_ref.47a [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.949
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Copyable.as.Copy.impl: imports.%Main.import_ref.300c2c.1 as imports.%Main.import_ref.7f5 [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.dda
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.b5c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT:   .ptr = imports.%Main.import_ref.fcf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Copyable [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.cdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonCopyable [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CNonCopy [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.61c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.5e5
// CHECK:STDOUT:   .ptr = imports.%Main.import_ref.8fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSubscript(%c.param: %C, %s.param: %SubscriptType.f8f) -> out %return.param: %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %impl.elem1: %.c04 = impl_witness_access constants.%IndexWith.impl_witness.efa, element1 [concrete = constants.%T.as_type.as.IndexWith.impl.At.ea7]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @T.as_type.as.IndexWith.impl.At(constants.%SubscriptType.f8f, constants.%facet_value) [concrete = constants.%T.as_type.as.IndexWith.impl.At.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc6_45.1: ref %Copyable = splice_block %return.param {}
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.call: init %Copyable to %.loc6_45.1 = call %bound_method.loc7_13.2(%c.ref, %s.ref)
// CHECK:STDOUT:   return %T.as_type.as.IndexWith.impl.At.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.IndirectIndexWith.impl.Ref [from "indirect_types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Copyable.as.Copy.impl.Op [from "indirect_types.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_non_copy_subscript.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CNonCopy: type = class_type @CNonCopy [concrete]
// CHECK:STDOUT:   %NonCopyable: type = class_type @NonCopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.d10: type = ptr_type %NonCopyable [concrete]
// CHECK:STDOUT:   %struct_type.ptr.ddf: type = struct_type {.ptr: %ptr.d10} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.ptr.ddf [concrete]
// CHECK:STDOUT:   %pattern_type.86e: type = pattern_type %CNonCopy [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.86e = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.86e = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %pattern_type.ea5: type = pattern_type %SubscriptType.f8f [concrete]
// CHECK:STDOUT:   %s.param_patt: %pattern_type.ea5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %s.patt: %pattern_type.ea5 = at_binding_pattern s, %s.param_patt [concrete]
// CHECK:STDOUT:   %TestSubscript.type: type = fn_type @TestSubscript [concrete]
// CHECK:STDOUT:   %TestSubscript: %TestSubscript.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.df6: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.df6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.620: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %.Self.c48: %IndirectIndexWith.type.620 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.lookup_impl_witness.6a3: <witness> = lookup_impl_witness %.Self.c48, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4d4: type = impl_witness_access %IndirectIndexWith.lookup_impl_witness.6a3, element0 [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.e63: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %Copyable> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.eec: type = ptr_type %Copyable [concrete]
// CHECK:STDOUT:   %struct_type.ptr.650: type = struct_type {.ptr: %ptr.eec} [concrete]
// CHECK:STDOUT:   %complete_type.fc9: <witness> = complete_type_witness %struct_type.ptr.650 [concrete]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.acd: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.4d4 = %NonCopyable> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Copyable = import_ref Main//indirect_types, Copyable, unloaded
// CHECK:STDOUT:   %Main.NonCopyable = import_ref Main//indirect_types, NonCopyable, unloaded
// CHECK:STDOUT:   %Main.SubscriptType: type = import_ref Main//indirect_types, SubscriptType, loaded [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %Main.C = import_ref Main//indirect_types, C, unloaded
// CHECK:STDOUT:   %Main.CNonCopy: type = import_ref Main//indirect_types, CNonCopy, loaded [concrete = constants.%CNonCopy]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.8f24d3.1: <witness> = import_ref Main//indirect_types, loc9_20, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.283 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.61c: <witness> = import_ref Main//indirect_types, loc24_1, loaded [concrete = constants.%complete_type.d5e]
// CHECK:STDOUT:   %Main.import_ref.5e5 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.8fd = import_ref Main//indirect_types, loc23_10, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.2: <witness> = import_ref Main//indirect_types, loc10_22, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.bd5 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.df6 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Main.import_ref.ca3 = import_ref Main//indirect_types, loc16_79, unloaded
// CHECK:STDOUT:   %Main.import_ref.8f24d3.3: <witness> = import_ref Main//indirect_types, loc8_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.cdb = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b5c: <witness> = import_ref Main//indirect_types, loc14_1, loaded [concrete = constants.%complete_type.fc9]
// CHECK:STDOUT:   %Main.import_ref.e47 = import_ref Main//indirect_types, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.fcf = import_ref Main//indirect_types, loc13_10, unloaded
// CHECK:STDOUT:   %Main.import_ref.335: type = import_ref Main//indirect_types, loc16_6, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.import_ref.0e1: type = import_ref Main//indirect_types, loc16_49, loaded [concrete = constants.%IndirectIndexWith_where.type.e63]
// CHECK:STDOUT:   %Main.import_ref.6d1: <witness> = import_ref Main//indirect_types, loc26_89, loaded [concrete = constants.%IndirectIndexWith.impl_witness]
// CHECK:STDOUT:   %Main.import_ref.db0: type = import_ref Main//indirect_types, loc26_6, loaded [concrete = constants.%CNonCopy]
// CHECK:STDOUT:   %Main.import_ref.47a: type = import_ref Main//indirect_types, loc26_56, loaded [concrete = constants.%IndirectIndexWith_where.type.acd]
// CHECK:STDOUT:   %Main.import_ref.a0c: type = import_ref Main//indirect_types, loc26_89, loaded [concrete = constants.%NonCopyable]
// CHECK:STDOUT:   %Main.import_ref.282 = import_ref Main//indirect_types, loc27_68, unloaded
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.a0c, %Main.import_ref.282), @CNonCopy.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.4c4 = import_ref Main//indirect_types, loc5_21, unloaded
// CHECK:STDOUT:   %Main.import_ref.300: type = import_ref Main//indirect_types, loc5_8, loaded [concrete = constants.%Copyable]
// CHECK:STDOUT:   %Main.import_ref.7f5: type = import_ref Main//indirect_types, loc5_15, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Copyable = imports.%Main.Copyable
// CHECK:STDOUT:     .NonCopyable = imports.%Main.NonCopyable
// CHECK:STDOUT:     .SubscriptType = imports.%Main.SubscriptType
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .CNonCopy = imports.%Main.CNonCopy
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TestSubscript = %TestSubscript.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %TestSubscript.decl: %TestSubscript.type = fn_decl @TestSubscript [concrete = constants.%TestSubscript] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.86e = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.86e = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.ea5 = value_param_pattern [concrete = constants.%s.param_patt]
// CHECK:STDOUT:     %s.patt: %pattern_type.ea5 = at_binding_pattern s, %s.param_patt [concrete = constants.%s.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %CNonCopy = value_param call_param0
// CHECK:STDOUT:     %CNonCopy.ref: type = name_ref CNonCopy, imports.%Main.CNonCopy [concrete = constants.%CNonCopy]
// CHECK:STDOUT:     %c: %CNonCopy = wrapper_binding c, %c.param
// CHECK:STDOUT:     %s.param: %SubscriptType.f8f = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %s: %SubscriptType.f8f = wrapper_binding s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndirectIndexWith.impl: imports.%Main.import_ref.335 as imports.%Main.import_ref.0e1 [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.ca3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @CNonCopy.as.IndirectIndexWith.impl: imports.%Main.import_ref.db0 as imports.%Main.import_ref.47a [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.6d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Copyable.as.Copy.impl: imports.%Main.import_ref.300 as imports.%Main.import_ref.7f5 [from "indirect_types.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Main.import_ref.4c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CNonCopy [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.61c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.5e5
// CHECK:STDOUT:   .ptr = imports.%Main.import_ref.8fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonCopyable [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SubscriptType [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.bd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Copyable [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.8f24d3.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.cdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "indirect_types.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.b5c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e47
// CHECK:STDOUT:   .ptr = imports.%Main.import_ref.fcf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestSubscript(%c.param: %CNonCopy, %s.param: %SubscriptType.f8f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %CNonCopy = name_ref c, %c
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
