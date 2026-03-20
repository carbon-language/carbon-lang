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
  fn At[unused self: Self](unused subscript: SubscriptType) -> ElementType {
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

impl (C, C) as Core.IndexWith(Core.IntLiteral()) where .ElementType = C {
  fn At[self: Self](unused subscript: Core.IntLiteral()) -> C {
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
  fn At[unused self: Self](unused subscript: SubscriptType) -> ElementType {
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


// CHECK:STDOUT: --- overloaded_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ElementType: type = class_type @ElementType.loc10 [concrete]
// CHECK:STDOUT:   %SubscriptType.9ec: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.a96: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndexWith.type.bbe: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.9ec)> [concrete]
// CHECK:STDOUT:   %.Self: %IndexWith.type.bbe = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndexWith.assoc_type.b14: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.9ec) [concrete]
// CHECK:STDOUT:   %assoc0.ab8: %IndexWith.assoc_type.b14 = assoc_entity element0, imports.%Core.import_ref.434 [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %assoc0.f5a: %IndexWith.assoc_type.a96 = assoc_entity element0, imports.%Core.import_ref.d9b [symbolic]
// CHECK:STDOUT:   %IndexWith.lookup_impl_witness.407: <witness> = lookup_impl_witness %.Self, @IndexWith, @IndexWith(%SubscriptType.9ec) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.9d3: type = impl_witness_access %IndexWith.lookup_impl_witness.407, element0 [symbolic_self]
// CHECK:STDOUT:   %IndexWith_where.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.9ec) where %impl.elem0.9d3 = %ElementType> [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness @C.as.IndexWith.impl.%IndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.541: type = pattern_type %SubscriptType.9ec [concrete]
// CHECK:STDOUT:   %.312: Core.Form = init_form %ElementType [concrete]
// CHECK:STDOUT:   %pattern_type.fde: type = pattern_type %ElementType [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.type: type = fn_type @C.as.IndexWith.impl.At [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At: %C.as.IndexWith.impl.At.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.bbe = facet_value %C, (%IndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.cc5: type = fn_type @IndexWith.WithSelf.At, @IndexWith.WithSelf(%SubscriptType.9ec, %IndexWith.facet) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ElementType.val: %ElementType = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.val: %SubscriptType.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %.8f0: type = fn_type_with_self_type %IndexWith.WithSelf.At.type.cc5, %IndexWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.c44: @IndexWith.WithSelf.%IndexWith.assoc_type (%IndexWith.assoc_type.a96) = import_ref Core//prelude/operators/index, loc16, loaded [symbolic = @IndexWith.WithSelf.%assoc0 (constants.%assoc0.f5a)]
// CHECK:STDOUT:   %Core.import_ref.434: type = import_ref Core//prelude/operators/index, loc16, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %Core.import_ref.d9b = import_ref Core//prelude/operators/index, loc16, unloaded
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
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType.loc10 [concrete = constants.%ElementType] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [concrete = constants.%SubscriptType.9ec] {} {}
// CHECK:STDOUT:   impl_decl @C.as.IndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.9ec]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.9ec)> [concrete = constants.%IndexWith.type.bbe]
// CHECK:STDOUT:     %.Self: %IndexWith.type.bbe = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %IndexWith.type.bbe = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc26.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc26.2: %IndexWith.assoc_type.b14 = specific_constant imports.%Core.import_ref.c44, @IndexWith.WithSelf(constants.%SubscriptType.9ec, constants.%.Self) [concrete = constants.%assoc0.ab8]
// CHECK:STDOUT:     %ElementType.ref.loc26: %IndexWith.assoc_type.b14 = name_ref ElementType, %.loc26.2 [concrete = constants.%assoc0.ab8]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%IndexWith.lookup_impl_witness.407, element0 [symbolic_self = constants.%impl.elem0.9d3]
// CHECK:STDOUT:     %ElementType.ref.loc29: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:     %.loc25: type = where_expr %.Self [concrete = constants.%IndexWith_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%IndexWith.type.bbe
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ElementType.ref.loc29
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.541 = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SubscriptType.ref: type = name_ref SubscriptType, %SubscriptType.decl [concrete = constants.%SubscriptType.9ec]
// CHECK:STDOUT:   %.loc60.1: ref %SubscriptType.9ec = temporary_storage
// CHECK:STDOUT:   %.loc60.2: init %SubscriptType.9ec to %.loc60.1 = class_init () [concrete = constants.%SubscriptType.val]
// CHECK:STDOUT:   %.loc60.3: init %SubscriptType.9ec = converted @__global_init.%.loc60, %.loc60.2 [concrete = constants.%SubscriptType.val]
// CHECK:STDOUT:   %.loc60.4: ref %SubscriptType.9ec = temporary %.loc60.1, %.loc60.3
// CHECK:STDOUT:   %.loc60.5: %SubscriptType.9ec = acquire_value %.loc60.4
// CHECK:STDOUT:   %s: %SubscriptType.9ec = value_binding s, %.loc60.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc68.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc68.2: init %C to %.loc68.1 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc68.3: init %C = converted @__global_init.%.loc68, %.loc68.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc68.4: ref %C = temporary %.loc68.1, %.loc68.3
// CHECK:STDOUT:   %.loc68.5: %C = acquire_value %.loc68.4
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc68.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.fde = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:   %.loc78.1: ref %ElementType = temporary @__global_init.%.loc78, @__global_init.%C.as.IndexWith.impl.At.call
// CHECK:STDOUT:   %.loc78.2: %ElementType = acquire_value %.loc78.1
// CHECK:STDOUT:   %x: %ElementType = value_binding x, %.loc78.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndexWith.impl: %C.ref as %.loc25 {
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.decl: %C.as.IndexWith.impl.At.type = fn_decl @C.as.IndexWith.impl.At [concrete = constants.%C.as.IndexWith.impl.At] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7c7 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.541 = value_param_pattern [concrete]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.541 = at_binding_pattern subscript, %subscript.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fde = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.fde = return_slot_pattern %return.param_patt, %ElementType.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:     %.loc46: Core.Form = init_form %ElementType.ref [concrete = constants.%.312]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.IndexWith.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.9ec = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.9ec]
// CHECK:STDOUT:     %subscript: %SubscriptType.9ec = value_binding subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType = out_param call_param2
// CHECK:STDOUT:     %return: ref %ElementType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C.as.IndexWith.impl.At.decl), @C.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness %IndexWith.impl_witness_table [concrete = constants.%IndexWith.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ElementType [concrete = constants.%ElementType]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SubscriptType = <poisoned>
// CHECK:STDOUT:   .ElementType = <poisoned>
// CHECK:STDOUT:   .At = %C.as.IndexWith.impl.At.decl
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
// CHECK:STDOUT: class @ElementType.loc10 {
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
// CHECK:STDOUT:   .Self = constants.%SubscriptType.9ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.IndexWith.impl.At(%self.param: %C, %subscript.param: %SubscriptType.9ec) -> out %return.param: %ElementType {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc50.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc50.2: init %ElementType to %return.param = class_init () [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc51: init %ElementType = converted %.loc50.1, %.loc50.2 [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   return %.loc51 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc60: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc68: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, file.%c
// CHECK:STDOUT:   %s.ref: %SubscriptType.9ec = name_ref s, file.%s
// CHECK:STDOUT:   %impl.elem1: %.8f0 = impl_witness_access constants.%IndexWith.impl_witness, element1 [concrete = constants.%C.as.IndexWith.impl.At]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %c.ref, %impl.elem1
// CHECK:STDOUT:   %.loc78: ref %ElementType = temporary_storage
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.call: init %ElementType to %.loc78 = call %bound_method(%c.ref, %s.ref)
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
// CHECK:STDOUT:   %tuple.6f4: %tuple.type.24b = tuple_value (%C, %C) [concrete]
// CHECK:STDOUT:   %tuple.type.d23: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.a96: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType) [symbolic]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.type.d54: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %.Self: %IndexWith.type.d54 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndexWith.assoc_type.972: type = assoc_entity_type @IndexWith, @IndexWith(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.b00: %IndexWith.assoc_type.972 = assoc_entity element0, imports.%Core.import_ref.434 [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %assoc0.f5a: %IndexWith.assoc_type.a96 = assoc_entity element0, imports.%Core.import_ref.d9b [symbolic]
// CHECK:STDOUT:   %IndexWith.lookup_impl_witness.6f5: <witness> = lookup_impl_witness %.Self, @IndexWith, @IndexWith(Core.IntLiteral) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d7c: type = impl_witness_access %IndexWith.lookup_impl_witness.6f5, element0 [symbolic_self]
// CHECK:STDOUT:   %IndexWith_where.type: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral) where %impl.elem0.d7c = %C> [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness @tuple.type.as.IndexWith.impl.%IndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c3b: type = pattern_type %tuple.type.d23 [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %.a69: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %tuple.type.as.IndexWith.impl.At.type: type = fn_type @tuple.type.as.IndexWith.impl.At [concrete]
// CHECK:STDOUT:   %tuple.type.as.IndexWith.impl.At: %tuple.type.as.IndexWith.impl.At.type = struct_value () [concrete]
// CHECK:STDOUT:   %IndexWith.facet: %IndexWith.type.d54 = facet_value %tuple.type.d23, (%IndexWith.impl_witness) [concrete]
// CHECK:STDOUT:   %IndexWith.WithSelf.At.type.4d2: type = fn_type @IndexWith.WithSelf.At, @IndexWith.WithSelf(Core.IntLiteral, %IndexWith.facet) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %empty_tuple.bfb: %C = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.d21: %tuple.type.d23 = tuple_value (%empty_tuple.bfb, %empty_tuple.bfb) [concrete]
// CHECK:STDOUT:   %.0a8: type = fn_type_with_self_type %IndexWith.WithSelf.At.type.4d2, %IndexWith.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.c44: @IndexWith.WithSelf.%IndexWith.assoc_type (%IndexWith.assoc_type.a96) = import_ref Core//prelude/operators/index, loc16, loaded [symbolic = @IndexWith.WithSelf.%assoc0 (constants.%assoc0.f5a)]
// CHECK:STDOUT:   %Core.import_ref.434: type = import_ref Core//prelude/operators/index, loc16, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.d9b = import_ref Core//prelude/operators/index, loc16, unloaded
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
// CHECK:STDOUT:     %C.ref.loc14: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc17.1: %tuple.type.24b = tuple_literal (%C.ref.loc14, %C.ref.loc16) [concrete = constants.%tuple.6f4]
// CHECK:STDOUT:     %.loc17.2: type = converted %.loc17.1, constants.%tuple.type.d23 [concrete = constants.%tuple.type.d23]
// CHECK:STDOUT:     %Core.ref.loc19: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %Core.ref.loc23: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc28.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc28.2: type = converted %IntLiteral.call, %.loc28.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(Core.IntLiteral)> [concrete = constants.%IndexWith.type.d54]
// CHECK:STDOUT:     %.Self: %IndexWith.type.d54 = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %IndexWith.type.d54 = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc30.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc30.2: %IndexWith.assoc_type.972 = specific_constant imports.%Core.import_ref.c44, @IndexWith.WithSelf(Core.IntLiteral, constants.%.Self) [concrete = constants.%assoc0.b00]
// CHECK:STDOUT:     %ElementType.ref: %IndexWith.assoc_type.972 = name_ref ElementType, %.loc30.2 [concrete = constants.%assoc0.b00]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%IndexWith.lookup_impl_witness.6f5, element0 [symbolic_self = constants.%impl.elem0.d7c]
// CHECK:STDOUT:     %C.ref.loc33: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc29: type = where_expr %.Self [concrete = constants.%IndexWith_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%IndexWith.type.d54
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc33
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.c3b = value_binding_pattern s [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc69.1: type = splice_block %.loc69.3 [concrete = constants.%tuple.type.d23] {
// CHECK:STDOUT:     %C.ref.loc66: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %C.ref.loc68: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc69.2: %tuple.type.24b = tuple_literal (%C.ref.loc66, %C.ref.loc68) [concrete = constants.%tuple.6f4]
// CHECK:STDOUT:     %.loc69.3: type = converted %.loc69.2, constants.%tuple.type.d23 [concrete = constants.%tuple.type.d23]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.loc81: %tuple.type.d23 = tuple_value (@__global_init.%.loc74.2, @__global_init.%.loc79.2) [concrete = constants.%tuple.d21]
// CHECK:STDOUT:   %.loc81: %tuple.type.d23 = converted @__global_init.%.loc81, %tuple.loc81 [concrete = constants.%tuple.d21]
// CHECK:STDOUT:   %s: %tuple.type.d23 = value_binding s, %.loc81
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.7c7 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc86: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc91.1: init %empty_tuple.type = as_compatible @__global_init.%tuple.type.as.IndexWith.impl.At.call
// CHECK:STDOUT:   %.loc91.2: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc91.3: ref %empty_tuple.type = temporary %.loc91.2, %.loc91.1
// CHECK:STDOUT:   %tuple.loc91: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc91.4: %C = as_compatible %tuple.loc91 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %.loc91.5: %C = converted @__global_init.%tuple.type.as.IndexWith.impl.At.call, %.loc91.4 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %e: %C = value_binding e, %.loc91.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @tuple.type.as.IndexWith.impl: %.loc17.2 as %.loc29 {
// CHECK:STDOUT:   %tuple.type.as.IndexWith.impl.At.decl: %tuple.type.as.IndexWith.impl.At.type = fn_decl @tuple.type.as.IndexWith.impl.At [concrete = constants.%tuple.type.as.IndexWith.impl.At] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c3b = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.c3b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.dc0 = at_binding_pattern subscript, %subscript.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7c7 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7c7 = return_slot_pattern %return.param_patt, %C.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc53: Core.Form = init_form %C.ref [concrete = constants.%.a69]
// CHECK:STDOUT:     %self.param: %tuple.type.d23 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @tuple.type.as.IndexWith.impl.%.loc17.2 [concrete = constants.%tuple.type.d23]
// CHECK:STDOUT:     %self: %tuple.type.d23 = value_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: Core.IntLiteral = value_param call_param1
// CHECK:STDOUT:     %.loc50.1: type = splice_block %.loc50.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc50.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc50.3: type = converted %IntLiteral.call, %.loc50.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %subscript: Core.IntLiteral = value_binding subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %tuple.type.as.IndexWith.impl.At.decl), @tuple.type.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness %IndexWith.impl_witness_table [concrete = constants.%IndexWith.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C [concrete = constants.%C]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .At = %tuple.type.as.IndexWith.impl.At.decl
// CHECK:STDOUT:   witness = %IndexWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc9: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc10: type = converted %.loc9, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   adapt_decl %.loc10 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_tuple.type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @tuple.type.as.IndexWith.impl.At(%self.param: %tuple.type.d23, %subscript.param: Core.IntLiteral) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %tuple.type.d23 = name_ref self, %self
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: %C = tuple_access %self.ref, element0
// CHECK:STDOUT:   %.loc59.1: %empty_tuple.type = as_compatible %tuple.elem0
// CHECK:STDOUT:   %.loc59.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc59.3: init %C = as_compatible %.loc59.2 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %.loc59.4: init %C = converted %tuple.elem0, %.loc59.3 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   return %.loc59.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc73: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %C.ref.loc75: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %empty_tuple.loc73: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc74.1: %C = as_compatible %empty_tuple.loc73 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %.loc74.2: %C = converted %.loc73, %.loc74.1 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %.loc78: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %C.ref.loc80: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %empty_tuple.loc78: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple.af4]
// CHECK:STDOUT:   %.loc79.1: %C = as_compatible %empty_tuple.loc78 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %.loc79.2: %C = converted %.loc78, %.loc79.1 [concrete = constants.%empty_tuple.bfb]
// CHECK:STDOUT:   %.loc81: %tuple.type.d23 = tuple_literal (%.loc74.2, %.loc79.2) [concrete = constants.%tuple.d21]
// CHECK:STDOUT:   %s.ref: %tuple.type.d23 = name_ref s, file.%s
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %impl.elem1: %.0a8 = impl_witness_access constants.%IndexWith.impl_witness, element1 [concrete = constants.%tuple.type.as.IndexWith.impl.At]
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
// CHECK:STDOUT:   %ElementType: type = class_type @ElementType.loc10 [concrete]
// CHECK:STDOUT:   %SubscriptType.9ec: type = class_type @SubscriptType [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
// CHECK:STDOUT:   %SubscriptType.67d: type = symbolic_binding SubscriptType, 0 [symbolic]
// CHECK:STDOUT:   %IndexWith.assoc_type.a96: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.67d) [symbolic]
// CHECK:STDOUT:   %IndexWith.type.bbe: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.9ec)> [concrete]
// CHECK:STDOUT:   %.Self: %IndexWith.type.bbe = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IndexWith.assoc_type.b14: type = assoc_entity_type @IndexWith, @IndexWith(%SubscriptType.9ec) [concrete]
// CHECK:STDOUT:   %assoc0.ab8: %IndexWith.assoc_type.b14 = assoc_entity element0, imports.%Core.import_ref.434 [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self [symbolic_self]
// CHECK:STDOUT:   %assoc0.f5a: %IndexWith.assoc_type.a96 = assoc_entity element0, imports.%Core.import_ref.d9b [symbolic]
// CHECK:STDOUT:   %IndexWith.lookup_impl_witness.407: <witness> = lookup_impl_witness %.Self, @IndexWith, @IndexWith(%SubscriptType.9ec) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.9d3: type = impl_witness_access %IndexWith.lookup_impl_witness.407, element0 [symbolic_self]
// CHECK:STDOUT:   %IndexWith_where.type: type = facet_type <@IndexWith, @IndexWith(%SubscriptType.9ec) where %impl.elem0.9d3 = %ElementType> [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness @C.as.IndexWith.impl.%IndexWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.541: type = pattern_type %SubscriptType.9ec [concrete]
// CHECK:STDOUT:   %.312: Core.Form = init_form %ElementType [concrete]
// CHECK:STDOUT:   %pattern_type.fde: type = pattern_type %ElementType [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.type: type = fn_type @C.as.IndexWith.impl.At [concrete]
// CHECK:STDOUT:   %C.as.IndexWith.impl.At: %C.as.IndexWith.impl.At.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ElementType.val: %ElementType = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:   %Core.import_ref.c44: @IndexWith.WithSelf.%IndexWith.assoc_type (%IndexWith.assoc_type.a96) = import_ref Core//prelude/operators/index, loc16, loaded [symbolic = @IndexWith.WithSelf.%assoc0 (constants.%assoc0.f5a)]
// CHECK:STDOUT:   %Core.import_ref.434: type = import_ref Core//prelude/operators/index, loc16, loaded [concrete = %ElementType]
// CHECK:STDOUT:   %Core.import_ref.d9b = import_ref Core//prelude/operators/index, loc16, unloaded
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
// CHECK:STDOUT:   %ElementType.decl: type = class_decl @ElementType.loc10 [concrete = constants.%ElementType] {} {}
// CHECK:STDOUT:   %SubscriptType.decl: type = class_decl @SubscriptType [concrete = constants.%SubscriptType.9ec] {} {}
// CHECK:STDOUT:   impl_decl @C.as.IndexWith.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %IndexWith.ref: %IndexWith.type.504 = name_ref IndexWith, imports.%Core.IndexWith [concrete = constants.%IndexWith.generic]
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.9ec]
// CHECK:STDOUT:     %IndexWith.type: type = facet_type <@IndexWith, @IndexWith(constants.%SubscriptType.9ec)> [concrete = constants.%IndexWith.type.bbe]
// CHECK:STDOUT:     %.Self: %IndexWith.type.bbe = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %IndexWith.type.bbe = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc26.1: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.binding.as_type]
// CHECK:STDOUT:     %.loc26.2: %IndexWith.assoc_type.b14 = specific_constant imports.%Core.import_ref.c44, @IndexWith.WithSelf(constants.%SubscriptType.9ec, constants.%.Self) [concrete = constants.%assoc0.ab8]
// CHECK:STDOUT:     %ElementType.ref.loc26: %IndexWith.assoc_type.b14 = name_ref ElementType, %.loc26.2 [concrete = constants.%assoc0.ab8]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access constants.%IndexWith.lookup_impl_witness.407, element0 [symbolic_self = constants.%impl.elem0.9d3]
// CHECK:STDOUT:     %ElementType.ref.loc29: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:     %.loc25: type = where_expr %.Self [concrete = constants.%IndexWith_where.type] {
// CHECK:STDOUT:       requirement_base_facet_type constants.%IndexWith.type.bbe
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %ElementType.ref.loc29
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc60.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc60.2: init %C to %.loc60.1 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc60.3: init %C = converted @__global_init.%.loc60, %.loc60.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc60.4: ref %C = temporary %.loc60.1, %.loc60.3
// CHECK:STDOUT:   %.loc60.5: %C = acquire_value %.loc60.4
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc60.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.fde = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ElementType.ref: type = name_ref ElementType, %ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:   %x: %ElementType = value_binding x, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.IndexWith.impl: %C.ref as %.loc25 {
// CHECK:STDOUT:   %C.as.IndexWith.impl.At.decl: %C.as.IndexWith.impl.At.type = fn_decl @C.as.IndexWith.impl.At [concrete = constants.%C.as.IndexWith.impl.At] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7c7 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %subscript.param_patt: %pattern_type.541 = value_param_pattern [concrete]
// CHECK:STDOUT:     %subscript.patt: %pattern_type.541 = at_binding_pattern subscript, %subscript.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fde = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.fde = return_slot_pattern %return.param_patt, %ElementType.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ElementType.ref: type = name_ref ElementType, file.%ElementType.decl [concrete = constants.%ElementType]
// CHECK:STDOUT:     %.loc46: Core.Form = init_form %ElementType.ref [concrete = constants.%.312]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.IndexWith.impl.%C.ref [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:     %subscript.param: %SubscriptType.9ec = value_param call_param1
// CHECK:STDOUT:     %SubscriptType.ref: type = name_ref SubscriptType, file.%SubscriptType.decl [concrete = constants.%SubscriptType.9ec]
// CHECK:STDOUT:     %subscript: %SubscriptType.9ec = value_binding subscript, %subscript.param
// CHECK:STDOUT:     %return.param: ref %ElementType = out_param call_param2
// CHECK:STDOUT:     %return: ref %ElementType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IndexWith.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant, %C.as.IndexWith.impl.At.decl), @C.as.IndexWith.impl [concrete]
// CHECK:STDOUT:   %IndexWith.impl_witness: <witness> = impl_witness %IndexWith.impl_witness_table [concrete = constants.%IndexWith.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%ElementType [concrete = constants.%ElementType]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .SubscriptType = <poisoned>
// CHECK:STDOUT:   .ElementType = <poisoned>
// CHECK:STDOUT:   .At = %C.as.IndexWith.impl.At.decl
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
// CHECK:STDOUT: class @ElementType.loc10 {
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
// CHECK:STDOUT:   .Self = constants.%SubscriptType.9ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.IndexWith.impl.At(%self.param: %C, %subscript.param: %SubscriptType.9ec) -> out %return.param: %ElementType {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc50.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc50.2: init %ElementType to %return.param = class_init () [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   %.loc51: init %ElementType = converted %.loc50.1, %.loc50.2 [concrete = constants.%ElementType.val]
// CHECK:STDOUT:   return %.loc51 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc60: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %IndexWith.type.504: type = generic_interface_type @IndexWith [concrete]
// CHECK:STDOUT:   %IndexWith.generic: %IndexWith.type.504 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.IndexWith: %IndexWith.type.504 = import_ref Core//prelude/operators/index, IndexWith, loaded [concrete = constants.%IndexWith.generic]
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
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc14.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc14.2: init %C to %.loc14.1 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc14.3: init %C = converted @__global_init.%.loc14, %.loc14.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc14.4: ref %C = temporary %.loc14.1, %.loc14.3
// CHECK:STDOUT:   %.loc14.5: %C = acquire_value %.loc14.4
// CHECK:STDOUT:   %c: %C = value_binding c, %.loc14.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: %i32 = value_binding x, <error> [concrete = <error>]
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
// CHECK:STDOUT:   %.loc14: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, file.%c
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
