// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/init.carbon

// --- from_struct.carbon

library "[[@TEST_NAME]]";

class Class(T: Core.Destroy) {
  var k: T;
}

//@dump-sem-ir-begin
fn InitFromStructGeneric(generic T: Core.Copy & Core.Destroy, x: T) -> T {
  var v: Class(T) = {.k = x};
  return v.k;
}

fn InitFromStructSpecific(x: i32) -> i32 {
  var v: Class(i32) = {.k = x};
  return v.k;
}
//@dump-sem-ir-end

// --- adapt.carbon

library "[[@TEST_NAME]]";

class Adapt(T: type) {
  adapt T;
}

//@dump-sem-ir-begin
fn InitFromAdaptedGeneric(generic T: Core.Copy, x: T) -> T {
  return (x as Adapt(T)) as T;
}

fn InitFromAdaptedSpecific(x: i32) -> i32 {
  return (x as Adapt(i32)) as i32;
}
//@dump-sem-ir-end


// CHECK:STDOUT: --- from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Class.type: type = parameterized_class_name_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.7f7: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.7f7 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.1a6: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.b4e: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.1a6, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Copy.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.709: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.patt.478: %pattern_type.709 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %T.as_type.74b [symbolic]
// CHECK:STDOUT:   %x.param_patt.e3c: %pattern_type.44a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.ecd: %pattern_type.44a = wrapper_binding_pattern x, %x.param_patt.e3c [symbolic]
// CHECK:STDOUT:   %.2b95: Core.Form = init_form %T.as_type.74b [symbolic]
// CHECK:STDOUT:   %return.param_patt.7e9: %pattern_type.44a = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c4f: %pattern_type.44a = return_slot_pattern %return.param_patt.7e9, %T.as_type.74b [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [concrete]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.44d: <witness> = require_complete_type %T.as_type.74b [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %Class.824: type = class_type @Class, @Class(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %Class.elem.8a5: type = unbound_element_type %Class.824, %T.as_type.74b [symbolic]
// CHECK:STDOUT:   %struct_type.k.aa4: type = struct_type {.k: %T.as_type.74b} [symbolic]
// CHECK:STDOUT:   %require_complete.51f: <witness> = require_complete_type %Class.824 [symbolic]
// CHECK:STDOUT:   %pattern_type.d2e: type = pattern_type %Class.824 [symbolic]
// CHECK:STDOUT:   %v.patt.07c: %pattern_type.d2e = ref_binding_pattern v [symbolic]
// CHECK:STDOUT:   %v.var_patt.b24: %pattern_type.d2e = var_pattern %v.patt.07c [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.4df: <witness> = lookup_impl_witness %T.283, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.3f0: %Copy.type = facet_value %T.as_type.74b, (%Copy.lookup_impl_witness.4df) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9f3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3f0) [symbolic]
// CHECK:STDOUT:   %.257: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9f3, %Copy.facet.3f0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5dc: %.257 = impl_witness_access %Copy.lookup_impl_witness.4df, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.572: <specific function> = specific_impl_function %impl.elem0.5dc, @Copy.WithSelf.Op(%Copy.facet.3f0) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f62: <witness> = lookup_impl_witness %Class.824, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.c4f: %Destroy.type = facet_value %Class.824, (%Destroy.lookup_impl_witness.f62) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.3b3: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.c4f) [symbolic]
// CHECK:STDOUT:   %.6d4: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.3b3, %Destroy.facet.c4f [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.6d4 = impl_witness_access %Destroy.lookup_impl_witness.f62, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2e4: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.c4f) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt.051: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0f7: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt.051 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.0c5: %pattern_type.d94 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.749: %pattern_type.d94 = wrapper_binding_pattern self, %self.param_patt.0c5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc15_19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc15_19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc15_19 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.1, %Destroy.WithSelf.SubobjectDestroy.d01daf.1, %Destroy.WithSelf.SelfDestruct.db3fdb.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.ed0: %Destroy.type = facet_value %i32, (%custom_witness.f8f19d.1) [concrete]
// CHECK:STDOUT:   %Class.9cf: type = class_type @Class, @Class(%Destroy.facet.ed0) [concrete]
// CHECK:STDOUT:   %Class.elem.fab: type = unbound_element_type %Class.9cf, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.k.d7d: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.ff0: type = pattern_type %Class.9cf [concrete]
// CHECK:STDOUT:   %v.patt.7b9: %pattern_type.ff0 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.812: %pattern_type.ff0 = var_pattern %v.patt.7b9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b93: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.e7d: type = pattern_type %struct_type.k.d7d [concrete]
// CHECK:STDOUT:   %self.param_patt.4e0: %pattern_type.e7d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.29c: %pattern_type.e7d = wrapper_binding_pattern self, %self.param_patt.4e0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc15_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc15_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.1cb: %pattern_type.ff0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.55a: %pattern_type.ff0 = wrapper_binding_pattern self, %self.param_patt.1cb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc15_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc15_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc15_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc15_19 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d94 = ref_param_pattern [concrete = constants.%self.param_patt.0c5]
// CHECK:STDOUT:     %self.patt: %pattern_type.d94 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.749]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc15_19 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d94 = ref_param_pattern [concrete = constants.%self.param_patt.0c5]
// CHECK:STDOUT:     %self.patt: %pattern_type.d94 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.749]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc15_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e7d = ref_param_pattern [concrete = constants.%self.param_patt.4e0]
// CHECK:STDOUT:     %self.patt: %pattern_type.e7d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.29c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.k.d7d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.k.d7d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc15_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e7d = ref_param_pattern [concrete = constants.%self.param_patt.4e0]
// CHECK:STDOUT:     %self.patt: %pattern_type.e7d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.29c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.k.d7d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.k.d7d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc15_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ff0 = ref_param_pattern [concrete = constants.%self.param_patt.1cb]
// CHECK:STDOUT:     %self.patt: %pattern_type.ff0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.55a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Class.9cf = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Class.9cf = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc15_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ff0 = ref_param_pattern [concrete = constants.%self.param_patt.1cb]
// CHECK:STDOUT:     %self.patt: %pattern_type.ff0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.55a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Class.9cf = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Class.9cf = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [concrete = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.patt.loc9_35.1: %pattern_type.709 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_35.2 (constants.%T.patt.478)]
// CHECK:STDOUT:     %x.param_patt.loc9_64.1: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = value_param_pattern [symbolic = %x.param_patt.loc9_64.2 (constants.%x.param_patt.e3c)]
// CHECK:STDOUT:     %x.patt.loc9_64.1: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = wrapper_binding_pattern x, %x.param_patt.loc9_64.1 [symbolic = %x.patt.loc9_64.2 (constants.%x.patt.ecd)]
// CHECK:STDOUT:     %return.param_patt.loc9_72.1: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = out_param_pattern [symbolic = %return.param_patt.loc9_72.2 (constants.%return.param_patt.7e9)]
// CHECK:STDOUT:     %return.patt.loc9_69.1: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = return_slot_pattern %return.param_patt.loc9_72.1, %.loc9_72.3 [symbolic = %return.patt.loc9_69.2 (constants.%return.patt.c4f)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_72: %facet_type = name_ref T, %T.loc9_35.2 [symbolic = %T.loc9_35.1 (constants.%T.283)]
// CHECK:STDOUT:     %T.as_type.loc9_72: type = facet_access_type %T.ref.loc9_72 [symbolic = %T.as_type.loc9_66.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:     %.loc9_72.3: type = converted %T.ref.loc9_72, %T.as_type.loc9_72 [symbolic = %T.as_type.loc9_66.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:     %.loc9_72.4: Core.Form = init_form %.loc9_72.3 [symbolic = %.loc9_72.2 (constants.%.2b95)]
// CHECK:STDOUT:     %.loc9_47.1: type = splice_block %.loc9_47.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref.loc9_37: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:       %Core.ref.loc9_49: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc9: %.b4e = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc9: <bound method> = bound_method %Copy.ref, %impl.elem0.loc9 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc9(%Copy.ref, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc9_47.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc9_47.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc9_47.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_35.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc9_35.1 (constants.%T.283)]
// CHECK:STDOUT:     %x.param: @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) = value_param call_param0
// CHECK:STDOUT:     %.loc9_66.1: type = splice_block %.loc9_66.2 [symbolic = %T.as_type.loc9_66.1 (constants.%T.as_type.74b)] {
// CHECK:STDOUT:       %T.ref.loc9_66: %facet_type = name_ref T, %T.loc9_35.2 [symbolic = %T.loc9_35.1 (constants.%T.283)]
// CHECK:STDOUT:       %T.as_type.loc9_66.2: type = facet_access_type %T.ref.loc9_66 [symbolic = %T.as_type.loc9_66.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:       %.loc9_66.2: type = converted %T.ref.loc9_66, %T.as_type.loc9_66.2 [symbolic = %T.as_type.loc9_66.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [concrete = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt.051]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.0f7]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc14_38 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc14_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14: Core.Form = init_form %i32.loc14_38 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc14_30: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc9_35.2: %facet_type) {
// CHECK:STDOUT:   %T.patt.loc9_35.2: %pattern_type.709 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_35.2 (constants.%T.patt.478)]
// CHECK:STDOUT:   %T.loc9_35.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc9_35.1 (constants.%T.283)]
// CHECK:STDOUT:   %T.as_type.loc9_66.1: type = facet_access_type %T.loc9_35.1 [symbolic = %T.as_type.loc9_66.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:   %pattern_type.loc9: type = pattern_type %T.as_type.loc9_66.1 [symbolic = %pattern_type.loc9 (constants.%pattern_type.44a)]
// CHECK:STDOUT:   %x.param_patt.loc9_64.2: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = value_param_pattern [symbolic = %x.param_patt.loc9_64.2 (constants.%x.param_patt.e3c)]
// CHECK:STDOUT:   %x.patt.loc9_64.2: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = wrapper_binding_pattern x, %x.param_patt.loc9_64.2 [symbolic = %x.patt.loc9_64.2 (constants.%x.patt.ecd)]
// CHECK:STDOUT:   %.loc9_72.2: Core.Form = init_form %T.as_type.loc9_66.1 [symbolic = %.loc9_72.2 (constants.%.2b95)]
// CHECK:STDOUT:   %return.param_patt.loc9_72.2: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = out_param_pattern [symbolic = %return.param_patt.loc9_72.2 (constants.%return.param_patt.7e9)]
// CHECK:STDOUT:   %return.patt.loc9_69.2: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = return_slot_pattern %return.param_patt.loc9_72.2, %T.as_type.loc9_66.1 [symbolic = %return.patt.loc9_69.2 (constants.%return.patt.c4f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc9_66.1 [symbolic = %require_complete.loc9 (constants.%require_complete.44d)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc10_17: <witness> = lookup_impl_witness %T.loc9_35.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc10_17 (constants.%Destroy.lookup_impl_witness.28a)]
// CHECK:STDOUT:   %require_witness_defn.loc10_17: require_witness_defn_type = require_witness_defn %Destroy.lookup_impl_witness.loc10_17 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.loc10_17.2: %Destroy.type = facet_value %T.as_type.loc9_66.1, (%Destroy.lookup_impl_witness.loc10_17) [symbolic = %Destroy.facet.loc10_17.2 (constants.%Destroy.facet.16e)]
// CHECK:STDOUT:   %Class.loc10_17.2: type = class_type @Class, @Class(%Destroy.facet.loc10_17.2) [symbolic = %Class.loc10_17.2 (constants.%Class.824)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class.loc10_17.2 [symbolic = %require_complete.loc10 (constants.%require_complete.51f)]
// CHECK:STDOUT:   %pattern_type.loc10: type = pattern_type %Class.loc10_17.2 [symbolic = %pattern_type.loc10 (constants.%pattern_type.d2e)]
// CHECK:STDOUT:   %v.patt.loc10_8.2: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.d2e) = ref_binding_pattern v [symbolic = %v.patt.loc10_8.2 (constants.%v.patt.07c)]
// CHECK:STDOUT:   %v.var_patt.loc10_3.2: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.d2e) = var_pattern %v.patt.loc10_8.2 [symbolic = %v.var_patt.loc10_3.2 (constants.%v.var_patt.b24)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b)} [symbolic = %struct_type.k (constants.%struct_type.k.aa4)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_35.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.4df)]
// CHECK:STDOUT:   %require_witness_defn.loc10_27: require_witness_defn_type = require_witness_defn %Copy.lookup_impl_witness [symbolic]
// CHECK:STDOUT:   %Copy.facet.loc10_27.3: %Copy.type = facet_value %T.as_type.loc9_66.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_27.3 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc10_27.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.9f3)]
// CHECK:STDOUT:   %.loc10_27.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc10_27.3 [symbolic = %.loc10_27.3 (constants.%.257)]
// CHECK:STDOUT:   %impl.elem0.loc10_27.2: @InitFromStructGeneric.%.loc10_27.3 (%.257) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.5dc)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_27.2: <specific function> = specific_impl_function %impl.elem0.loc10_27.2, @Copy.WithSelf.Op(%Copy.facet.loc10_27.3) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.572)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc10_17.2, %T.as_type.loc9_66.1 [symbolic = %Class.elem (constants.%Class.elem.8a5)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc10_3: <witness> = lookup_impl_witness %Class.loc10_17.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc10_3 (constants.%Destroy.lookup_impl_witness.f62)]
// CHECK:STDOUT:   %require_witness_defn.loc10_3: require_witness_defn_type = require_witness_defn %Destroy.lookup_impl_witness.loc10_3 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.3: %Destroy.type = facet_value %Class.loc10_17.2, (%Destroy.lookup_impl_witness.loc10_3) [symbolic = %Destroy.facet.loc10_3.3 (constants.%Destroy.facet.c4f)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.loc10_3.3) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.3b3)]
// CHECK:STDOUT:   %.loc10_3.4: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Destroy.facet.loc10_3.3 [symbolic = %.loc10_3.4 (constants.%.6d4)]
// CHECK:STDOUT:   %impl.elem2.loc10_3.2: @InitFromStructGeneric.%.loc10_3.4 (%.6d4) = impl_witness_access %Destroy.lookup_impl_witness.loc10_3, element2 [symbolic = %impl.elem2.loc10_3.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_3.2: <specific function> = specific_impl_function %impl.elem2.loc10_3.2, @Destroy.WithSelf.SelfDestruct.1(%Destroy.facet.loc10_3.3) [symbolic = %specific_impl_fn.loc10_3.2 (constants.%specific_impl_fn.2e4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b)) -> out %return.param: @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.824) = var_storage %v.var_patt.loc10_3.1
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) = name_ref x, %x
// CHECK:STDOUT:     %.loc10_28.1: @InitFromStructGeneric.%struct_type.k (%struct_type.k.aa4) = struct_literal (%x.ref)
// CHECK:STDOUT:     %impl.elem0.loc10_27.1: @InitFromStructGeneric.%.loc10_27.3 (%.257) = impl_witness_access constants.%Copy.lookup_impl_witness.4df, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.5dc)]
// CHECK:STDOUT:     %bound_method.loc10_27.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_27.1
// CHECK:STDOUT:     %Copy.facet.loc10_27.1: %Copy.type = facet_value constants.%T.as_type.74b, (constants.%Copy.lookup_impl_witness.4df) [symbolic = %Copy.facet.loc10_27.3 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %.loc10_27.1: %Copy.type = converted constants.%T.as_type.74b, %Copy.facet.loc10_27.1 [symbolic = %Copy.facet.loc10_27.3 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %Copy.facet.loc10_27.2: %Copy.type = facet_value constants.%T.as_type.74b, (constants.%Copy.lookup_impl_witness.4df) [symbolic = %Copy.facet.loc10_27.3 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %.loc10_27.2: %Copy.type = converted constants.%T.as_type.74b, %Copy.facet.loc10_27.2 [symbolic = %Copy.facet.loc10_27.3 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_27.1: <specific function> = specific_impl_function %impl.elem0.loc10_27.1, @Copy.WithSelf.Op(constants.%Copy.facet.3f0) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.572)]
// CHECK:STDOUT:     %bound_method.loc10_27.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_27.1
// CHECK:STDOUT:     %.loc10_28.2: ref @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) = class_element_access %v.var, element0
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc10: init @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) to %.loc10_28.2 = call %bound_method.loc10_27.2(%x.ref)
// CHECK:STDOUT:     %.loc10_28.3: init @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) to %.loc10_28.2 = in_place_init %Copy.WithSelf.Op.call.loc10
// CHECK:STDOUT:     %.loc10_28.4: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.824) to %v.var = class_init (%.loc10_28.3)
// CHECK:STDOUT:     %.loc10_3.1: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.824) = converted %.loc10_28.1, %.loc10_28.4
// CHECK:STDOUT:     assign %v.var, %.loc10_3.1
// CHECK:STDOUT:     %.loc10_17.1: type = splice_block %Class.loc10_17.1 [symbolic = %Class.loc10_17.2 (constants.%Class.824)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc10: %facet_type = name_ref T, %T.loc9_35.2 [symbolic = %T.loc9_35.1 (constants.%T.283)]
// CHECK:STDOUT:       %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.as_type.loc9_66.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:       %Destroy.facet.loc10_17.1: %Destroy.type = facet_value %T.as_type.loc10, (constants.%Destroy.lookup_impl_witness.28a) [symbolic = %Destroy.facet.loc10_17.2 (constants.%Destroy.facet.16e)]
// CHECK:STDOUT:       %.loc10_17.2: %Destroy.type = converted %T.ref.loc10, %Destroy.facet.loc10_17.1 [symbolic = %Destroy.facet.loc10_17.2 (constants.%Destroy.facet.16e)]
// CHECK:STDOUT:       %Class.loc10_17.1: type = class_type @Class, @Class(constants.%Destroy.facet.16e) [symbolic = %Class.loc10_17.2 (constants.%Class.824)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.824) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc10_8.1: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.d2e) = ref_binding_pattern v [symbolic = %v.patt.loc10_8.2 (constants.%v.patt.07c)]
// CHECK:STDOUT:       %v.var_patt.loc10_3.1: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.d2e) = var_pattern %v.patt.loc10_8.1 [symbolic = %v.var_patt.loc10_3.2 (constants.%v.var_patt.b24)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.824) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%Class.elem (%Class.elem.8a5) = name_ref k, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:     %.loc11_11.1: ref @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc11_11.2: @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) = acquire_value %.loc11_11.1
// CHECK:STDOUT:     %impl.elem0.loc11: @InitFromStructGeneric.%.loc10_27.3 (%.257) = impl_witness_access constants.%Copy.lookup_impl_witness.4df, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.5dc)]
// CHECK:STDOUT:     %bound_method.loc11_11.1: <bound method> = bound_method %.loc11_11.2, %impl.elem0.loc11
// CHECK:STDOUT:     %Copy.facet.loc11_11.1: %Copy.type = facet_value constants.%T.as_type.74b, (constants.%Copy.lookup_impl_witness.4df) [symbolic = %Copy.facet.loc10_27.3 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %.loc11_11.3: %Copy.type = converted constants.%T.as_type.74b, %Copy.facet.loc11_11.1 [symbolic = %Copy.facet.loc10_27.3 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %Copy.facet.loc11_11.2: %Copy.type = facet_value constants.%T.as_type.74b, (constants.%Copy.lookup_impl_witness.4df) [symbolic = %Copy.facet.loc10_27.3 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %.loc11_11.4: %Copy.type = converted constants.%T.as_type.74b, %Copy.facet.loc11_11.2 [symbolic = %Copy.facet.loc10_27.3 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %specific_impl_fn.loc11: <specific function> = specific_impl_function %impl.elem0.loc11, @Copy.WithSelf.Op(constants.%Copy.facet.3f0) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.572)]
// CHECK:STDOUT:     %bound_method.loc11_11.2: <bound method> = bound_method %.loc11_11.2, %specific_impl_fn.loc11
// CHECK:STDOUT:     %.loc9_72.1: ref @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc11: init @InitFromStructGeneric.%T.as_type.loc9_66.1 (%T.as_type.74b) to %.loc9_72.1 = call %bound_method.loc11_11.2(%.loc11_11.2)
// CHECK:STDOUT:     %impl.elem2.loc10_3.1: @InitFromStructGeneric.%.loc10_3.4 (%.6d4) = impl_witness_access constants.%Destroy.lookup_impl_witness.f62, element2 [symbolic = %impl.elem2.loc10_3.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc10_3.1: <bound method> = bound_method %v.var, %impl.elem2.loc10_3.1
// CHECK:STDOUT:     %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%Class.824, (constants.%Destroy.lookup_impl_witness.f62) [symbolic = %Destroy.facet.loc10_3.3 (constants.%Destroy.facet.c4f)]
// CHECK:STDOUT:     %.loc10_3.2: %Destroy.type = converted constants.%Class.824, %Destroy.facet.loc10_3.1 [symbolic = %Destroy.facet.loc10_3.3 (constants.%Destroy.facet.c4f)]
// CHECK:STDOUT:     %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%Class.824, (constants.%Destroy.lookup_impl_witness.f62) [symbolic = %Destroy.facet.loc10_3.3 (constants.%Destroy.facet.c4f)]
// CHECK:STDOUT:     %.loc10_3.3: %Destroy.type = converted constants.%Class.824, %Destroy.facet.loc10_3.2 [symbolic = %Destroy.facet.loc10_3.3 (constants.%Destroy.facet.c4f)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_3.1: <specific function> = specific_impl_function %impl.elem2.loc10_3.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Destroy.facet.c4f) [symbolic = %specific_impl_fn.loc10_3.2 (constants.%specific_impl_fn.2e4)]
// CHECK:STDOUT:     %bound_method.loc10_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc10_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc10_3.2(%v.var)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call.loc11 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %Class.9cf = var_storage %v.var_patt
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc15_30.1: %struct_type.k.d7d = struct_literal (%x.ref)
// CHECK:STDOUT:   %impl.elem0.loc15: %.2b93 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc15_29.1: <bound method> = bound_method %x.ref, %impl.elem0.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_29.2: <bound method> = bound_method %x.ref, %specific_fn.loc15
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15: init %i32 = call %bound_method.loc15_29.2(%x.ref)
// CHECK:STDOUT:   %.loc15_30.2: ref %i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc15_30.3: init %i32 to %.loc15_30.2 = in_place_init %Int.as.Copy.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_30.4: init %Class.9cf to %v.var = class_init (%.loc15_30.3)
// CHECK:STDOUT:   %.loc15_3: init %Class.9cf = converted %.loc15_30.1, %.loc15_30.4
// CHECK:STDOUT:   assign %v.var, %.loc15_3
// CHECK:STDOUT:   %.loc15_19.1: type = splice_block %Class [concrete = constants.%Class.9cf] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Destroy.facet: %Destroy.type = facet_value %i32.loc15, (constants.%custom_witness.f8f19d.1) [concrete = constants.%Destroy.facet.ed0]
// CHECK:STDOUT:     %.loc15_19.2: %Destroy.type = converted %i32.loc15, %Destroy.facet [concrete = constants.%Destroy.facet.ed0]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%Destroy.facet.ed0) [concrete = constants.%Class.9cf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Class.9cf = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.ff0 = ref_binding_pattern v [concrete = constants.%v.patt.7b9]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.ff0 = var_pattern %v.patt [concrete = constants.%v.var_patt.812]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %Class.9cf = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %Class.elem.fab = name_ref k, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:   %.loc16_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc16_11.2: %i32 = acquire_value %.loc16_11.1
// CHECK:STDOUT:   %impl.elem0.loc16: %.2b93 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc16_11.1: <bound method> = bound_method %.loc16_11.2, %impl.elem0.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_11.2: <bound method> = bound_method %.loc16_11.2, %specific_fn.loc16
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16: init %i32 = call %bound_method.loc16_11.2(%.loc16_11.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %v.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%v.var)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc15_19(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc15_19(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc15_19(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc15_3.1(%self.param: ref %struct_type.k.d7d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc15, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc15)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc15_3.1(%self.param: ref %struct_type.k.d7d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc15_3.1(%self.param: ref %struct_type.k.d7d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc15_3.2(%self.param: ref %Class.9cf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %k.ref: %Class.elem.fab = name_ref k, @Class.%field_decl [concrete = @Class.%field_decl]
// CHECK:STDOUT:   %.loc15: ref %i32 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc15, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc15)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc15_3.2(%self.param: ref %Class.9cf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc15_3.2(%self.param: ref %Class.9cf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T.283) {
// CHECK:STDOUT:   %T.patt.loc9_35.2 => constants.%T.patt.478
// CHECK:STDOUT:   %T.loc9_35.1 => constants.%T.283
// CHECK:STDOUT:   %T.as_type.loc9_66.1 => constants.%T.as_type.74b
// CHECK:STDOUT:   %pattern_type.loc9 => constants.%pattern_type.44a
// CHECK:STDOUT:   %x.param_patt.loc9_64.2 => constants.%x.param_patt.e3c
// CHECK:STDOUT:   %x.patt.loc9_64.2 => constants.%x.patt.ecd
// CHECK:STDOUT:   %.loc9_72.2 => constants.%.2b95
// CHECK:STDOUT:   %return.param_patt.loc9_72.2 => constants.%return.param_patt.7e9
// CHECK:STDOUT:   %return.patt.loc9_69.2 => constants.%return.patt.c4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Adapt.type: type = parameterized_class_name_type @Adapt [concrete]
// CHECK:STDOUT:   %Adapt.generic: %Adapt.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.1f4: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.cec: %pattern_type.2c66b4.2 = wrapper_binding_pattern x, %x.param_patt.1f4 [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %InitFromAdaptedGeneric.type: type = fn_type @InitFromAdaptedGeneric [concrete]
// CHECK:STDOUT:   %InitFromAdaptedGeneric: %InitFromAdaptedGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Adapt.879: type = class_type @Adapt, @Adapt(%T.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.c6e: <witness> = require_complete_type %Adapt.879 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2f9c3.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.8bf: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2f9c3.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb5: %.8bf = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ac: <specific function> = specific_impl_function %impl.elem0.bb5, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt.051: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0f7: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt.051 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific.type: type = fn_type @InitFromAdaptedSpecific [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific: %InitFromAdaptedSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapt.d6d: type = class_type @Adapt, @Adapt(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromAdaptedGeneric.decl: %InitFromAdaptedGeneric.type = fn_decl @InitFromAdaptedGeneric [concrete = constants.%InitFromAdaptedGeneric] {
// CHECK:STDOUT:     %T.patt.loc9_36.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_36.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:     %x.param_patt.loc9_50.1: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc9_50.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:     %x.patt.loc9_50.1: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc9_50.1 [symbolic = %x.patt.loc9_50.2 (constants.%x.patt.cec)]
// CHECK:STDOUT:     %return.param_patt.loc9_58.1: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc9_58.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:     %return.patt.loc9_55.1: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc9_58.1, %.loc9_58.3 [symbolic = %return.patt.loc9_55.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_58: %Copy.type = name_ref T, %T.loc9_36.2 [symbolic = %T.loc9_36.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc9_58: type = facet_access_type %T.ref.loc9_58 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_58.3: type = converted %T.ref.loc9_58, %T.as_type.loc9_58 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_58.4: Core.Form = init_form %.loc9_58.3 [symbolic = %.loc9_58.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %.loc9_42: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_36.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_36.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.param: @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_52.1: type = splice_block %.loc9_52.2 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_52: %Copy.type = name_ref T, %T.loc9_36.2 [symbolic = %T.loc9_36.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc9_52.2: type = facet_access_type %T.ref.loc9_52 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_52.2: type = converted %T.ref.loc9_52, %T.as_type.loc9_52.2 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedSpecific.decl: %InitFromAdaptedSpecific.type = fn_decl @InitFromAdaptedSpecific [concrete = constants.%InitFromAdaptedSpecific] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt.051]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.0f7]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc13_39 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc13_39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13: Core.Form = init_form %i32.loc13_39 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc13_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc9_36.2: %Copy.type) {
// CHECK:STDOUT:   %T.patt.loc9_36.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_36.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   %T.loc9_36.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_36.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc9_52.1: type = facet_access_type %T.loc9_36.1 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_52.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %x.param_patt.loc9_50.2: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc9_50.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:   %x.patt.loc9_50.2: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc9_50.2 [symbolic = %x.patt.loc9_50.2 (constants.%x.patt.cec)]
// CHECK:STDOUT:   %.loc9_58.2: Core.Form = init_form %T.as_type.loc9_52.1 [symbolic = %.loc9_58.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc9_58.2: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc9_58.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc9_55.2: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc9_58.2, %T.as_type.loc9_52.1 [symbolic = %return.patt.loc9_55.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc9_52.1 [symbolic = %require_complete.loc9 (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Adapt.loc10_23.2: type = class_type @Adapt, @Adapt(%T.as_type.loc9_52.1) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.879)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Adapt.loc10_23.2 [symbolic = %require_complete.loc10 (constants.%require_complete.c6e)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc9_36.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2f9c3.2)]
// CHECK:STDOUT:   %.loc10_26.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc9_36.1 [symbolic = %.loc10_26.5 (constants.%.8bf)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_36.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %require_witness_defn: require_witness_defn_type = require_witness_defn %Copy.lookup_impl_witness [symbolic]
// CHECK:STDOUT:   %impl.elem0.loc10_26.2: @InitFromAdaptedGeneric.%.loc10_26.5 (%.8bf) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_26.2: <specific function> = specific_impl_function %impl.elem0.loc10_26.2, @Copy.WithSelf.Op(%T.loc9_36.1) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type)) -> out %return.param: @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:     %T.ref.loc10_22: %Copy.type = name_ref T, %T.loc9_36.2 [symbolic = %T.loc9_36.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc10_23: type = facet_access_type %T.ref.loc10_22 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_23: type = converted %T.ref.loc10_22, %T.as_type.loc10_23 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %Adapt.loc10_23.1: type = class_type @Adapt, @Adapt(constants.%T.as_type) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.879)]
// CHECK:STDOUT:     %.loc10_13.1: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.879) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc10_13.2: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.879) = converted %x.ref, %.loc10_13.1
// CHECK:STDOUT:     %T.ref.loc10_29: %Copy.type = name_ref T, %T.loc9_36.2 [symbolic = %T.loc9_36.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc10_29: type = facet_access_type %T.ref.loc10_29 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_29: type = converted %T.ref.loc10_29, %T.as_type.loc10_29 [symbolic = %T.as_type.loc9_52.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_26.1: @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) = as_compatible %.loc10_13.2
// CHECK:STDOUT:     %.loc10_26.2: @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) = converted %.loc10_13.2, %.loc10_26.1
// CHECK:STDOUT:     %impl.elem0.loc10_26.1: @InitFromAdaptedGeneric.%.loc10_26.5 (%.8bf) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:     %bound_method.loc10_26.1: <bound method> = bound_method %.loc10_26.2, %impl.elem0.loc10_26.1
// CHECK:STDOUT:     %.loc10_26.3: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc9_36.1 (constants.%T.f84)]
// CHECK:STDOUT:     %.loc10_26.4: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc9_36.1 (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_26.1: <specific function> = specific_impl_function %impl.elem0.loc10_26.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:     %bound_method.loc10_26.2: <bound method> = bound_method %.loc10_26.2, %specific_impl_fn.loc10_26.1
// CHECK:STDOUT:     %.loc9_58.1: ref @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @InitFromAdaptedGeneric.%T.as_type.loc9_52.1 (%T.as_type) to %.loc9_58.1 = call %bound_method.loc10_26.2(%.loc10_26.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromAdaptedSpecific(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:   %i32.loc14_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(constants.%i32) [concrete = constants.%Adapt.d6d]
// CHECK:STDOUT:   %.loc14_13.1: %Adapt.d6d = as_compatible %x.ref
// CHECK:STDOUT:   %.loc14_13.2: %Adapt.d6d = converted %x.ref, %.loc14_13.1
// CHECK:STDOUT:   %i32.loc14_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc14_28.1: %i32 = as_compatible %.loc14_13.2
// CHECK:STDOUT:   %.loc14_28.2: %i32 = converted %.loc14_13.2, %.loc14_28.1
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc14_28.1: <bound method> = bound_method %.loc14_28.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_28.2: <bound method> = bound_method %.loc14_28.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_28.2(%.loc14_28.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc9_36.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc9_36.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc9_52.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %x.param_patt.loc9_50.2 => constants.%x.param_patt.1f4
// CHECK:STDOUT:   %x.patt.loc9_50.2 => constants.%x.patt.cec
// CHECK:STDOUT:   %.loc9_58.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %return.param_patt.loc9_58.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc9_55.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
