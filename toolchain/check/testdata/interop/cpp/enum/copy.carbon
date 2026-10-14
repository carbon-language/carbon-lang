// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/copy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/copy.carbon

// --- enum.h

enum Enum : short { a, b, c };

// --- copy_enum.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

//@dump-sem-ir-begin
fn F() {
  var a: Cpp.Enum = Cpp.Enum.a;

  a = Cpp.Enum.b;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- copy_enum.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Enum: type = class_type @Enum [concrete]
// CHECK:STDOUT:   %pattern_type.926: type = pattern_type %Enum [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.926 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.926 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_0: %Enum = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7f5: type = fn_type @Copy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.416: %Copy.WithSelf.Op.type.7f5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bab: <witness> = custom_witness (%Copy.WithSelf.Op.416), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.512: %Copy.type = facet_value %Enum, (%custom_witness.bab) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a9d: type = fn_type @Copy.WithSelf.Op.1, @Copy.WithSelf(%Copy.facet.512) [concrete]
// CHECK:STDOUT:   %.18e: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a9d, %Copy.facet.512 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.bound.996: <bound method> = bound_method %int_0, %Copy.WithSelf.Op.416 [concrete]
// CHECK:STDOUT:   %int_1: %Enum = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.bound.1a9: <bound method> = bound_method %int_1, %Copy.WithSelf.Op.416 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.496: %SubobjectDestroy.type = facet_value %Enum, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e12: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.496) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Enum, (%DestroyImpl.impl_witness.e12, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Enum = %Enum.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Enum.decl: type = class_decl @Enum [concrete = constants.%Enum] {} {}
// CHECK:STDOUT:   %int_0: %Enum = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1: %Enum = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Enum = var_storage %a.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Enum.ref.loc8_24: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   %a.ref.loc8: %Enum = name_ref a, imports.%int_0 [concrete = constants.%int_0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc8: %.18e = impl_witness_access constants.%custom_witness.bab, element0 [concrete = constants.%Copy.WithSelf.Op.416]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %a.ref.loc8, %impl.elem0.loc8 [concrete = constants.%Copy.WithSelf.Op.bound.996]
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc8: init %Enum = call %bound_method.loc8(%a.ref.loc8) [concrete = constants.%int_0]
// CHECK:STDOUT:   assign %a.var, %Copy.WithSelf.Op.call.loc8
// CHECK:STDOUT:   %.loc8_13: type = splice_block %Enum.ref.loc8_13 [concrete = constants.%Enum] {
// CHECK:STDOUT:     %Cpp.ref.loc8_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Enum.ref.loc8_13: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Enum = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.926 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.926 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc10: ref %Enum = name_ref a, %a
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Enum.ref.loc10: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   %b.ref: %Enum = name_ref b, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc10: %.18e = impl_witness_access constants.%custom_witness.bab, element0 [concrete = constants.%Copy.WithSelf.Op.416]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %b.ref, %impl.elem0.loc10 [concrete = constants.%Copy.WithSelf.Op.bound.1a9]
// CHECK:STDOUT:   %Copy.WithSelf.Op.call.loc10: init %Enum = call %bound_method.loc10(%b.ref) [concrete = constants.%int_1]
// CHECK:STDOUT:   assign %a.ref.loc10, %Copy.WithSelf.Op.call.loc10
// CHECK:STDOUT:   %facet_value.loc8_3.1: %facet_type = facet_value constants.%Enum, (constants.%DestroyImpl.impl_witness.e12, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.1: %facet_type = converted constants.%Enum, %facet_value.loc8_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc8_3.2: %facet_type = facet_value constants.%Enum, (constants.%DestroyImpl.impl_witness.e12, constants.%custom_witness.b80) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc8_3.2: %facet_type = converted constants.%Enum, %facet_value.loc8_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %Enum) = "no_op";
// CHECK:STDOUT:
