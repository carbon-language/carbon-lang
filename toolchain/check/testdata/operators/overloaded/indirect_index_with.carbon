// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/indirect_index_with.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/indirect_index_with.carbon

// --- types.carbon

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

//@dump-sem-ir-begin
impl C as Core.IndirectIndexWith(SubscriptType) where .ElementType = Copyable {
  fn Ref(self, unused subscript: SubscriptType) -> ref Copyable {
    return *self.ptr;
  }
}
//@dump-sem-ir-end

class CNonCopy {
  var ptr: NonCopyable*;
}

impl CNonCopy as Core.IndirectIndexWith(SubscriptType)
    where .ElementType = NonCopyable {
  fn Ref(self, unused subscript: SubscriptType) -> ref NonCopyable {
    return *self.ptr;
  }
}

// --- ref_copyable.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(c: C, s: SubscriptType, v: Copyable) -> ref Copyable {
  //@dump-sem-ir-begin
  c.(Core.IndirectIndexWith(SubscriptType).Ref)(s) = v;
  return c.(Core.IndirectIndexWith(SubscriptType).Ref)(s);
  //@dump-sem-ir-end
}

// --- ref_non_copyable.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(c: CNonCopy, s: SubscriptType) -> ref NonCopyable {
  //@dump-sem-ir-begin
  return c.(Core.IndirectIndexWith(SubscriptType).Ref)(s);
  //@dump-sem-ir-end
}

// --- at_from_blanket_impl.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(c: C, s: SubscriptType) -> Copyable {
  //@dump-sem-ir-begin
  return c.(Core.IndexWith(SubscriptType).At)(s);
  //@dump-sem-ir-end
}

// --- fail_at_non_copyable.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(c: CNonCopy, s: SubscriptType) {
  // CHECK:STDERR:        ╭─┤ fail_at_non_copyable.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   c.(Core.IndexWith(SubscriptType).At)(s);
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot access member of interface `Core.IndexWith(SubscriptType)`
  // CHECK:STDERR:        ·      ┃        in type `CNonCopy` that does not implement that interface
  // CHECK:STDERR:        ·      ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  c.(Core.IndexWith(SubscriptType).At)(s);
}

// --- subscript_copyable.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(c: C, s: SubscriptType) -> Copyable {
  //@dump-sem-ir-begin
  return c[s];
  //@dump-sem-ir-end
}

// --- subscript_non_copyable.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(c: CNonCopy, s: SubscriptType) -> ref NonCopyable {
  //@dump-sem-ir-begin
  return c[s];
  //@dump-sem-ir-end
}

// --- subscript_assign.carbon

library "[[@TEST_NAME]]";

import library "types";

fn F(c: C, s: SubscriptType, v: Copyable) {
  //@dump-sem-ir-begin
  c[s] = v;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.26e: type = pattern_type %Copyable [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.eec: type = ptr_type %Copyable [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %ptr.eec [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.841: type = generic_interface_type @IndirectIndexWith [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.generic: %IndirectIndexWith.type.841 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.ea2: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.c10: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %IndirectIndexWith.type.c10 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.688: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %assoc0.08b: %IndirectIndexWith.assoc_type.688 = assoc_entity element0, imports.%Core.import_ref.abc [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %assoc0.b3973b.2: %IndirectIndexWith.assoc_type.ea2 = assoc_entity element0, imports.%Core.import_ref.55c4ab.2 [symbolic]
// CHECK:STDOUT:   %.a0b: <witness> = impl_self_witness %.Self.frozen, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e55: type = impl_witness_access %.a0b, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %IndirectIndexWith.type.c10 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.658: <witness> = impl_self_witness %.Self, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.758: type = impl_witness_access %.658, element0 [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.ad3: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.758 = %Copyable> [concrete]
// CHECK:STDOUT:   %.537: <witness> = impl_self_witness %C, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness.d52: <witness> = impl_witness @C.as.IndirectIndexWith.impl.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.303: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.36a: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.303 [concrete]
// CHECK:STDOUT:   %pattern_type.ea5: type = pattern_type %SubscriptType.f8f [concrete]
// CHECK:STDOUT:   %subscript.param_patt.5ee: %pattern_type.ea5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %subscript.patt.32e: %pattern_type.ea5 = wrapper_binding_pattern subscript, %subscript.param_patt.5ee [concrete]
// CHECK:STDOUT:   %.ce5: Core.Form = ref_form %Copyable [concrete]
// CHECK:STDOUT:   %.a78: %pattern_type.26e = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.eb1: %pattern_type.26e = return_slot_pattern %.a78, %Copyable [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.type: type = fn_type @C.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref: %C.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.6e3: @IndirectIndexWith.WithSelf.%IndirectIndexWith.assoc_type (%IndirectIndexWith.assoc_type.ea2) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%assoc0 (constants.%assoc0.b3973b.2)]
// CHECK:STDOUT:   %Core.import_ref.abc: type = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %Core.import_ref.55c4ab.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %ElementType: type = assoc_const_decl @ElementType [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @C.as.IndirectIndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndirectIndexWith.ref: %IndirectIndexWith.type.841 = name_ref IndirectIndexWith, imports.%Core.IndirectIndexWith [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %IndirectIndexWith.type: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndirectIndexWith.type.c10]
// CHECK:STDOUT:     %.Self.frozen: %IndirectIndexWith.type.c10 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %IndirectIndexWith.type [concrete]
// CHECK:STDOUT:     %.Self.ref: %IndirectIndexWith.type.c10 = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc17_55.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc17_55.2: %IndirectIndexWith.assoc_type.688 = specific_constant imports.%Core.import_ref.6e3, @IndirectIndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%.Self.frozen) [concrete = constants.%assoc0.08b]
// CHECK:STDOUT:     %ElementType.ref: %IndirectIndexWith.assoc_type.688 = name_ref ElementType, %.loc17_55.2 [concrete = constants.%assoc0.08b]
// CHECK:STDOUT:     %impl.elem0.loc17_55.1: type = impl_witness_access constants.%.a0b, element0 [symbolic_self = constants.%impl.elem0.e55]
// CHECK:STDOUT:     %Copyable.ref: type = name_ref Copyable, file.%Copyable.decl [concrete = constants.%Copyable]
// CHECK:STDOUT:     %rewrite.loc17_68.1 = requirement_rewrite %impl.elem0.loc17_55.1, %Copyable.ref [concrete]
// CHECK:STDOUT:     %impl.elem0.loc17_55.2: type = impl_witness_access constants.%.658, element0 [symbolic_self = constants.%impl.elem0.758]
// CHECK:STDOUT:     %rewrite.loc17_68.2 = requirement_rewrite %impl.elem0.loc17_55.2, %Copyable.ref [concrete]
// CHECK:STDOUT:     %.loc17_49: type = where_expr [concrete = constants.%IndirectIndexWith_where.type.ad3] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %IndirectIndexWith.type [concrete]
// CHECK:STDOUT:       %rewrite.loc17_68.2 = requirement_rewrite %impl.elem0.loc17_55.2, %Copyable.ref [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = impl_self_witness @C.as.IndirectIndexWith.impl.%C.ref, @IndirectIndexWith, @IndirectIndexWith(constants.%SubscriptType.f8f) [concrete = constants.%.537]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndirectIndexWith.impl: %C.ref as %IndirectIndexWith.type {
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.decl: %C.as.IndirectIndexWith.impl.Ref.type = fn_decl @C.as.IndirectIndexWith.impl.Ref [concrete = constants.%C.as.IndirectIndexWith.impl.Ref] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%self.param_patt.303]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.36a]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.ea5 = value_param_pattern [concrete = constants.%subscript.param_patt.5ee]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.ea5 = wrapper_binding_pattern subscript, %subscript.param_patt [concrete = constants.%subscript.patt.32e]
// CHECK:STDOUT:     %.loc18_52.1: %pattern_type.26e = ref_return_pattern [concrete = constants.%.a78]
// CHECK:STDOUT:     %return.patt: %pattern_type.26e = return_slot_pattern %.loc18_52.1, %.loc18_52.3 [concrete = constants.%return.patt.eb1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Copyable.ref: type = name_ref Copyable, file.%Copyable.decl [concrete = constants.%Copyable]
// CHECK:STDOUT:     %.loc18_52.2: Core.Form = ref_form %Copyable.ref [concrete = constants.%.ce5]
// CHECK:STDOUT:     %.loc18_52.3: type = type_component_of %.loc18_52.2 [concrete = constants.%Copyable]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.IndirectIndexWith.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = wrapper_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.f8f = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:     %subscript: %SubscriptType.f8f = wrapper_binding subscript, %subscript.param
// CHECK:STDOUT:     %.loc18_52.4: ref %Copyable = ref_return
// CHECK:STDOUT:     %return: ref %Copyable = return_slot %.loc18_52.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C.as.IndirectIndexWith.impl.Ref.decl), @C.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness %IndirectIndexWith.impl_witness_table [concrete = constants.%IndirectIndexWith.impl_witness.d52]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%Copyable [concrete = constants.%Copyable]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SubscriptType = <poisoned>
// CHECK:STDOUT:   .Copyable = <poisoned>
// CHECK:STDOUT:   .Ref = %C.as.IndirectIndexWith.impl.Ref.decl
// CHECK:STDOUT:   extend %IndirectIndexWith.type
// CHECK:STDOUT:   witness = %IndirectIndexWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.IndirectIndexWith.impl.Ref(%self.param: %C, %subscript.param: %SubscriptType.f8f) -> ref %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   %ptr.ref: %C.elem = name_ref ptr, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:   %.loc19_17.1: ref %ptr.eec = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc19_17.2: %ptr.eec = acquire_value %.loc19_17.1
// CHECK:STDOUT:   %.loc19_12: ref %Copyable = deref %.loc19_17.2
// CHECK:STDOUT:   return %.loc19_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.841: type = generic_interface_type @IndirectIndexWith [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.generic: %IndirectIndexWith.type.841 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.57f: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d)> [symbolic]
// CHECK:STDOUT:   %Self.2fa: %IndirectIndexWith.type.57f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.ea2: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.923: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.67d, %Self.2fa) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.62a: %IndirectIndexWith.WithSelf.Ref.type.923 = struct_value () [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.c10: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.688: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %assoc1.b1d: %IndirectIndexWith.assoc_type.688 = assoc_entity element1, imports.%Core.import_ref.17a [concrete]
// CHECK:STDOUT:   %assoc1.f633d8.2: %IndirectIndexWith.assoc_type.ea2 = assoc_entity element1, imports.%Core.import_ref.c78b2f.2 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.facet: %IndirectIndexWith.type.c10 = facet_value %C, (%IndirectIndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.7ed: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.f8f, %IndirectIndexWith.facet) [concrete]
// CHECK:STDOUT:   %.edc: type = fn_type_with_self_type %IndirectIndexWith.WithSelf.Ref.type.7ed, %IndirectIndexWith.facet [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.type: type = fn_type @C.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref: %C.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c00: <witness> = impl_witness imports.%Copy.impl_witness_table.a69 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Copyable, (%Copy.impl_witness.c00) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.43f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.43f, %Copy.facet [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.type: type = fn_type @Copyable.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op: %Copyable.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.SubscriptType: type = import_ref Main//types, SubscriptType, loaded [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndirectIndexWith = %Core.IndirectIndexWith
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndirectIndexWith: %IndirectIndexWith.type.841 = import_ref Core//prelude/operators/index, IndirectIndexWith, loaded [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.25c: @IndirectIndexWith.WithSelf.%IndirectIndexWith.assoc_type (%IndirectIndexWith.assoc_type.ea2) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%assoc1 (constants.%assoc1.f633d8.2)]
// CHECK:STDOUT:   %Core.import_ref.17a: @IndirectIndexWith.WithSelf.%IndirectIndexWith.WithSelf.Ref.type (%IndirectIndexWith.WithSelf.Ref.type.923) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%IndirectIndexWith.WithSelf.Ref (constants.%IndirectIndexWith.WithSelf.Ref.62a)]
// CHECK:STDOUT:   %Core.import_ref.c78b2f.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Main.import_ref.bdc = import_ref Main//types, loc17_79, unloaded
// CHECK:STDOUT:   %Main.import_ref.2b0: %C.as.IndirectIndexWith.impl.Ref.type = import_ref Main//types, loc18_65, loaded [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.bdc, %Main.import_ref.2b0), @C.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Main.import_ref.b8e: %Copyable.as.Copy.impl.Op.type = import_ref Main//types, loc6_32, loaded [concrete = constants.%Copyable.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.a69 = impl_witness_table (%Main.import_ref.b8e), @Copyable.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C, %s.param: %SubscriptType.f8f, %v.param: %Copyable) -> ref %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc8: %C = name_ref c, %c
// CHECK:STDOUT:   %Core.ref.loc8: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %IndirectIndexWith.ref.loc8: %IndirectIndexWith.type.841 = name_ref IndirectIndexWith, imports.%Core.IndirectIndexWith [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %SubscriptType.ref.loc8: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %IndirectIndexWith.type.loc8: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndirectIndexWith.type.c10]
// CHECK:STDOUT:   %.loc8_43: %IndirectIndexWith.assoc_type.688 = specific_constant imports.%Core.import_ref.25c, @IndirectIndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%Self.2fa) [concrete = constants.%assoc1.b1d]
// CHECK:STDOUT:   %Ref.ref.loc8: %IndirectIndexWith.assoc_type.688 = name_ref Ref, %.loc8_43 [concrete = constants.%assoc1.b1d]
// CHECK:STDOUT:   %impl.elem1.loc8: %.edc = impl_witness_access constants.%IndirectIndexWith.impl_witness, element1 [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %bound_method.loc8_4: <bound method> = bound_method %c.ref.loc8, %impl.elem1.loc8
// CHECK:STDOUT:   %s.ref.loc8: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.call.loc8: ref %Copyable = call %bound_method.loc8_4(%c.ref.loc8, %s.ref.loc8)
// CHECK:STDOUT:   %v.ref: %Copyable = name_ref v, %v
// CHECK:STDOUT:   %impl.elem0: %.9d4 = impl_witness_access constants.%Copy.impl_witness.c00, element0 [concrete = constants.%Copyable.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_54: <bound method> = bound_method %v.ref, %impl.elem0
// CHECK:STDOUT:   %.loc8_50: ref %Copyable = splice_block %C.as.IndirectIndexWith.impl.Ref.call.loc8 {}
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.call: init %Copyable to %.loc8_50 = call %bound_method.loc8_54(%v.ref)
// CHECK:STDOUT:   assign %C.as.IndirectIndexWith.impl.Ref.call.loc8, %Copyable.as.Copy.impl.Op.call
// CHECK:STDOUT:   %c.ref.loc9: %C = name_ref c, %c
// CHECK:STDOUT:   %Core.ref.loc9: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %IndirectIndexWith.ref.loc9: %IndirectIndexWith.type.841 = name_ref IndirectIndexWith, imports.%Core.IndirectIndexWith [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %SubscriptType.ref.loc9: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %IndirectIndexWith.type.loc9: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndirectIndexWith.type.c10]
// CHECK:STDOUT:   %.loc9: %IndirectIndexWith.assoc_type.688 = specific_constant imports.%Core.import_ref.25c, @IndirectIndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%Self.2fa) [concrete = constants.%assoc1.b1d]
// CHECK:STDOUT:   %Ref.ref.loc9: %IndirectIndexWith.assoc_type.688 = name_ref Ref, %.loc9 [concrete = constants.%assoc1.b1d]
// CHECK:STDOUT:   %impl.elem1.loc9: %.edc = impl_witness_access constants.%IndirectIndexWith.impl_witness, element1 [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %c.ref.loc9, %impl.elem1.loc9
// CHECK:STDOUT:   %s.ref.loc9: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.call.loc9: ref %Copyable = call %bound_method.loc9(%c.ref.loc9, %s.ref.loc9)
// CHECK:STDOUT:   return %C.as.IndirectIndexWith.impl.Ref.call.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_non_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %CNonCopy: type = class_type @CNonCopy [concrete]
// CHECK:STDOUT:   %NonCopyable: type = class_type @NonCopyable [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.841: type = generic_interface_type @IndirectIndexWith [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.generic: %IndirectIndexWith.type.841 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.57f: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d)> [symbolic]
// CHECK:STDOUT:   %Self.2fa: %IndirectIndexWith.type.57f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.ea2: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.923: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.67d, %Self.2fa) [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.62a: %IndirectIndexWith.WithSelf.Ref.type.923 = struct_value () [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.c10: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.assoc_type.688: type = assoc_entity_type @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %assoc1.b1d: %IndirectIndexWith.assoc_type.688 = assoc_entity element1, imports.%Core.import_ref.17a [concrete]
// CHECK:STDOUT:   %assoc1.f633d8.2: %IndirectIndexWith.assoc_type.ea2 = assoc_entity element1, imports.%Core.import_ref.c78b2f.2 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.facet: %IndirectIndexWith.type.c10 = facet_value %CNonCopy, (%IndirectIndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.111: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.f8f, %IndirectIndexWith.facet) [concrete]
// CHECK:STDOUT:   %.e4b: type = fn_type_with_self_type %IndirectIndexWith.WithSelf.Ref.type.111, %IndirectIndexWith.facet [concrete]
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref.type: type = fn_type @CNonCopy.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref: %CNonCopy.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.SubscriptType: type = import_ref Main//types, SubscriptType, loaded [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndirectIndexWith = %Core.IndirectIndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndirectIndexWith: %IndirectIndexWith.type.841 = import_ref Core//prelude/operators/index, IndirectIndexWith, loaded [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.25c: @IndirectIndexWith.WithSelf.%IndirectIndexWith.assoc_type (%IndirectIndexWith.assoc_type.ea2) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%assoc1 (constants.%assoc1.f633d8.2)]
// CHECK:STDOUT:   %Core.import_ref.17a: @IndirectIndexWith.WithSelf.%IndirectIndexWith.WithSelf.Ref.type (%IndirectIndexWith.WithSelf.Ref.type.923) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndirectIndexWith.WithSelf.%IndirectIndexWith.WithSelf.Ref (constants.%IndirectIndexWith.WithSelf.Ref.62a)]
// CHECK:STDOUT:   %Core.import_ref.c78b2f.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Main.import_ref.e33 = import_ref Main//types, loc29_38, unloaded
// CHECK:STDOUT:   %Main.import_ref.2b0: %CNonCopy.as.IndirectIndexWith.impl.Ref.type = import_ref Main//types, loc30_68, loaded [concrete = constants.%CNonCopy.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.e33, %Main.import_ref.2b0), @CNonCopy.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %CNonCopy, %s.param: %SubscriptType.f8f) -> ref %NonCopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %CNonCopy = name_ref c, %c
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %IndirectIndexWith.ref: %IndirectIndexWith.type.841 = name_ref IndirectIndexWith, imports.%Core.IndirectIndexWith [concrete = constants.%IndirectIndexWith.generic]
// CHECK:STDOUT:   %SubscriptType.ref.loc8: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %IndirectIndexWith.type: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndirectIndexWith.type.c10]
// CHECK:STDOUT:   %.loc8: %IndirectIndexWith.assoc_type.688 = specific_constant imports.%Core.import_ref.25c, @IndirectIndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%Self.2fa) [concrete = constants.%assoc1.b1d]
// CHECK:STDOUT:   %Ref.ref: %IndirectIndexWith.assoc_type.688 = name_ref Ref, %.loc8 [concrete = constants.%assoc1.b1d]
// CHECK:STDOUT:   %impl.elem1: %.e4b = impl_witness_access constants.%IndirectIndexWith.impl_witness, element1 [concrete = constants.%CNonCopy.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref.call: ref %NonCopyable = call %bound_method(%c.ref, %s.ref)
// CHECK:STDOUT:   return %CNonCopy.as.IndirectIndexWith.impl.Ref.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- at_from_blanket_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndexWith.type.df6: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.df6 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.type.ea2: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.67d)> [symbolic]
// CHECK:STDOUT:   %Self.ad4: %IndexWith.type.ea2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.783: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.62b: type = fn_type @IndexWith.WithSelf.At, @IndexWith.WithSelf(%SubscriptType.67d, %Self.ad4) [symbolic]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.fa4: %IndexWith.WithSelf.At.type.62b = struct_value () [symbolic]
// CHECK:STDOUT:   %IndexWith.type.a14: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %IndexWith.assoc_type.e71: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.f8f) [concrete]
// CHECK:STDOUT:   %assoc1.0c5: %IndexWith.assoc_type.e71 = assoc_entity element1, imports.%Core.import_ref.bab [concrete]
// CHECK:STDOUT:   %assoc1.487a81.2: %IndexWith.assoc_type.783 = assoc_entity element1, imports.%Core.import_ref.2101b9.2 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.57f: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d)> [symbolic]
// CHECK:STDOUT:   %.Self.16c: %IndirectIndexWith.type.57f = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.lookup_impl_witness.e21: <witness> = lookup_impl_witness %.Self.16c, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.a07: type = impl_witness_access %IndirectIndexWith.lookup_impl_witness.e21, element0 [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.57fb59.1: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.67d) where %impl.elem0.a07 impls @Copy> [symbolic]
// CHECK:STDOUT:   %T.2fa: %IndirectIndexWith_where.type.57fb59.1 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.type.152: type = fn_type @T.as_type.as.IndexWith.impl.At, @T.as_type.as.IndexWith.impl(%SubscriptType.67d, %T.2fa) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.68d: %T.as_type.as.IndexWith.impl.At.type.152 = struct_value () [symbolic]
// CHECK:STDOUT:   %IndirectIndexWith.type.c10: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %.Self.3da: %IndirectIndexWith.type.c10 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith.lookup_impl_witness.37b: <witness> = lookup_impl_witness %.Self.3da, @IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c7e: type = impl_witness_access %IndirectIndexWith.lookup_impl_witness.37b, element0 [symbolic_self]
// CHECK:STDOUT:   %IndirectIndexWith_where.type.c10528.1: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f) where %impl.elem0.c7e impls @Copy> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c00: <witness> = impl_witness imports.%Copy.impl_witness_table.a69 [concrete]
// CHECK:STDOUT:   %facet_value: %IndirectIndexWith_where.type.c10528.1 = facet_value %C, (%IndirectIndexWith.impl_witness, %Copy.impl_witness.c00) [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness.8ae: <witness> = impl_witness imports.%IndexWith.impl_witness_table, @T.as_type.as.IndexWith.impl(%SubscriptType.f8f, %facet_value) [concrete]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.type.a8d: type = fn_type @T.as_type.as.IndexWith.impl.At, @T.as_type.as.IndexWith.impl(%SubscriptType.f8f, %facet_value) [concrete]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.a13: %T.as_type.as.IndexWith.impl.At.type.a8d = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet.fe4: %IndexWith.type.a14 = facet_value %C, (%IndexWith.impl_witness.8ae) [concrete]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.54e: type = fn_type @IndexWith.WithSelf.At, @IndexWith.WithSelf(%SubscriptType.f8f, %IndexWith.facet.fe4) [concrete]
// CHECK:STDOUT:   %.646: type = fn_type_with_self_type %IndexWith.WithSelf.At.type.54e, %IndexWith.facet.fe4 [concrete]
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.specific_fn: <specific function> = specific_function %T.as_type.as.IndexWith.impl.At.a13, @T.as_type.as.IndexWith.impl.At(%SubscriptType.f8f, %facet_value) [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.type: type = fn_type @C.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref: %C.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.type: type = fn_type @Copyable.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op: %Copyable.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.SubscriptType: type = import_ref Main//types, SubscriptType, loaded [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.df6 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.211: @IndexWith.WithSelf.%IndexWith.assoc_type (%IndexWith.assoc_type.783) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.WithSelf.%assoc1 (constants.%assoc1.487a81.2)]
// CHECK:STDOUT:   %Core.import_ref.bab: @IndexWith.WithSelf.%IndexWith.WithSelf.At.type (%IndexWith.WithSelf.At.type.62b) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @IndexWith.WithSelf.%IndexWith.WithSelf.At (constants.%IndexWith.WithSelf.At.fa4)]
// CHECK:STDOUT:   %Core.import_ref.2101b9.2 = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.44f = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.630: @T.as_type.as.IndexWith.impl.%T.as_type.as.IndexWith.impl.At.type (%T.as_type.as.IndexWith.impl.At.type.152) = import_ref Core//prelude/operators/index, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.IndexWith.impl.%T.as_type.as.IndexWith.impl.At (constants.%T.as_type.as.IndexWith.impl.At.68d)]
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%Core.import_ref.44f, %Core.import_ref.630), @T.as_type.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.967: type = import_ref Main//types, loc17_79, loaded [concrete = constants.%Copyable]
// CHECK:STDOUT:   %Main.import_ref.2b0: %C.as.IndirectIndexWith.impl.Ref.type = import_ref Main//types, loc18_65, loaded [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.967, %Main.import_ref.2b0), @C.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.b8e: %Copyable.as.Copy.impl.Op.type = import_ref Main//types, loc6_32, loaded [concrete = constants.%Copyable.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.a69 = impl_witness_table (%Main.import_ref.b8e), @Copyable.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C, %s.param: %SubscriptType.f8f) -> out %return.param: %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:   %IndexWith.ref: %IndexWith.type.df6 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %SubscriptType.ref.loc8: type = name_ref SubscriptType, imports.%Main.SubscriptType [concrete = constants.%SubscriptType.f8f]
// CHECK:STDOUT:   %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.f8f)> [concrete = constants.%IndexWith.type.a14]
// CHECK:STDOUT:   %.loc8: %IndexWith.assoc_type.e71 = specific_constant imports.%Core.import_ref.211, @IndexWith.WithSelf(constants.%SubscriptType.f8f, constants.%Self.ad4) [concrete = constants.%assoc1.0c5]
// CHECK:STDOUT:   %At.ref: %IndexWith.assoc_type.e71 = name_ref At, %.loc8 [concrete = constants.%assoc1.0c5]
// CHECK:STDOUT:   %impl.elem1: %.646 = impl_witness_access constants.%IndexWith.impl_witness.8ae, element1 [concrete = constants.%T.as_type.as.IndexWith.impl.At.a13]
// CHECK:STDOUT:   %bound_method.loc8_11: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem1, @T.as_type.as.IndexWith.impl.At(constants.%SubscriptType.f8f, constants.%facet_value) [concrete = constants.%T.as_type.as.IndexWith.impl.At.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_48: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as_type.as.IndexWith.impl.At.call: init %Copyable to %.loc6_33.1 = call %bound_method.loc8_48(%c.ref, %s.ref)
// CHECK:STDOUT:   return %T.as_type.as.IndexWith.impl.At.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- subscript_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.c10: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.facet: %IndirectIndexWith.type.c10 = facet_value %C, (%IndirectIndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.7ed: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.f8f, %IndirectIndexWith.facet) [concrete]
// CHECK:STDOUT:   %.edc: type = fn_type_with_self_type %IndirectIndexWith.WithSelf.Ref.type.7ed, %IndirectIndexWith.facet [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.type: type = fn_type @C.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref: %C.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c00: <witness> = impl_witness imports.%Copy.impl_witness_table.a69 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Copyable, (%Copy.impl_witness.c00) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.43f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.43f, %Copy.facet [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.type: type = fn_type @Copyable.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op: %Copyable.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.import_ref.bdc = import_ref Main//types, loc17_79, unloaded
// CHECK:STDOUT:   %Main.import_ref.2b0: %C.as.IndirectIndexWith.impl.Ref.type = import_ref Main//types, loc18_65, loaded [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.bdc, %Main.import_ref.2b0), @C.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.b8e: %Copyable.as.Copy.impl.Op.type = import_ref Main//types, loc6_32, loaded [concrete = constants.%Copyable.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.a69 = impl_witness_table (%Main.import_ref.b8e), @Copyable.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C, %s.param: %SubscriptType.f8f) -> out %return.param: %Copyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %impl.elem1: %.edc = impl_witness_access constants.%IndirectIndexWith.impl_witness, element1 [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.call: ref %Copyable = call %bound_method.loc8_13.1(%c.ref, %s.ref)
// CHECK:STDOUT:   %.loc8: %Copyable = acquire_value %C.as.IndirectIndexWith.impl.Ref.call
// CHECK:STDOUT:   %impl.elem0: %.9d4 = impl_witness_access constants.%Copy.impl_witness.c00, element0 [concrete = constants.%Copyable.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %.loc8, %impl.elem0
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.call: init %Copyable to %.loc6_33.1 = call %bound_method.loc8_13.2(%.loc8)
// CHECK:STDOUT:   return %Copyable.as.Copy.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- subscript_non_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %CNonCopy: type = class_type @CNonCopy [concrete]
// CHECK:STDOUT:   %NonCopyable: type = class_type @NonCopyable [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.c10: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.facet: %IndirectIndexWith.type.c10 = facet_value %CNonCopy, (%IndirectIndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.111: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.f8f, %IndirectIndexWith.facet) [concrete]
// CHECK:STDOUT:   %.e4b: type = fn_type_with_self_type %IndirectIndexWith.WithSelf.Ref.type.111, %IndirectIndexWith.facet [concrete]
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref.type: type = fn_type @CNonCopy.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref: %CNonCopy.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.import_ref.e33 = import_ref Main//types, loc29_38, unloaded
// CHECK:STDOUT:   %Main.import_ref.2b0: %CNonCopy.as.IndirectIndexWith.impl.Ref.type = import_ref Main//types, loc30_68, loaded [concrete = constants.%CNonCopy.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.e33, %Main.import_ref.2b0), @CNonCopy.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %CNonCopy, %s.param: %SubscriptType.f8f) -> ref %NonCopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %CNonCopy = name_ref c, %c
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %impl.elem1: %.e4b = impl_witness_access constants.%IndirectIndexWith.impl_witness, element1 [concrete = constants.%CNonCopy.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %CNonCopy.as.IndirectIndexWith.impl.Ref.call: ref %NonCopyable = call %bound_method(%c.ref, %s.ref)
// CHECK:STDOUT:   return %CNonCopy.as.IndirectIndexWith.impl.Ref.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- subscript_assign.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Copyable: type = class_type @Copyable [concrete]
// CHECK:STDOUT:   %SubscriptType.f8f: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.type.c10: type = facet_type <@IndirectIndexWith, @IndirectIndexWith(%SubscriptType.f8f)> [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness: <witness> = impl_witness imports.%IndirectIndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.facet: %IndirectIndexWith.type.c10 = facet_value %C, (%IndirectIndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndirectIndexWith.WithSelf.Ref.type.7ed: type = fn_type @IndirectIndexWith.WithSelf.Ref, @IndirectIndexWith.WithSelf(%SubscriptType.f8f, %IndirectIndexWith.facet) [concrete]
// CHECK:STDOUT:   %.edc: type = fn_type_with_self_type %IndirectIndexWith.WithSelf.Ref.type.7ed, %IndirectIndexWith.facet [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.type: type = fn_type @C.as.IndirectIndexWith.impl.Ref [concrete]
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref: %C.as.IndirectIndexWith.impl.Ref.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c00: <witness> = impl_witness imports.%Copy.impl_witness_table.a69 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %Copyable, (%Copy.impl_witness.c00) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.43f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.43f, %Copy.facet [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.type: type = fn_type @Copyable.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op: %Copyable.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.import_ref.bdc = import_ref Main//types, loc17_79, unloaded
// CHECK:STDOUT:   %Main.import_ref.2b0: %C.as.IndirectIndexWith.impl.Ref.type = import_ref Main//types, loc18_65, loaded [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %IndirectIndexWith.impl_witness_table = impl_witness_table (%Main.import_ref.bdc, %Main.import_ref.2b0), @C.as.IndirectIndexWith.impl [concrete]
// CHECK:STDOUT:   %Main.import_ref.b8e: %Copyable.as.Copy.impl.Op.type = import_ref Main//types, loc6_32, loaded [concrete = constants.%Copyable.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.a69 = impl_witness_table (%Main.import_ref.b8e), @Copyable.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %C, %s.param: %SubscriptType.f8f, %v.param: %Copyable) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %s.ref: %SubscriptType.f8f = name_ref s, %s
// CHECK:STDOUT:   %impl.elem1: %.edc = impl_witness_access constants.%IndirectIndexWith.impl_witness, element1 [concrete = constants.%C.as.IndirectIndexWith.impl.Ref]
// CHECK:STDOUT:   %bound_method.loc8_6: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %C.as.IndirectIndexWith.impl.Ref.call: ref %Copyable = call %bound_method.loc8_6(%c.ref, %s.ref)
// CHECK:STDOUT:   %v.ref: %Copyable = name_ref v, %v
// CHECK:STDOUT:   %impl.elem0: %.9d4 = impl_witness_access constants.%Copy.impl_witness.c00, element0 [concrete = constants.%Copyable.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_10: <bound method> = bound_method %v.ref, %impl.elem0
// CHECK:STDOUT:   %.loc8: ref %Copyable = splice_block %C.as.IndirectIndexWith.impl.Ref.call {}
// CHECK:STDOUT:   %Copyable.as.Copy.impl.Op.call: init %Copyable to %.loc8 = call %bound_method.loc8_10(%v.ref)
// CHECK:STDOUT:   assign %C.as.IndirectIndexWith.impl.Ref.call, %Copyable.as.Copy.impl.Op.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
