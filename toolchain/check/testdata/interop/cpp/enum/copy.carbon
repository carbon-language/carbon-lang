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
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Enum: type = class_type @Enum [concrete]
// CHECK:STDOUT:   %pattern_type.926: type = pattern_type %Enum [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.926 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.926 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_0: %Enum = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Enum.Op.type: type = fn_type @Enum.Op [concrete]
// CHECK:STDOUT:   %Enum.Op: %Enum.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0d: <witness> = custom_witness (%Enum.Op), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.200: %Copy.type = facet_value %Enum, (%custom_witness.d0d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a18: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.200) [concrete]
// CHECK:STDOUT:   %.1e6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a18, %Copy.facet.200 [concrete]
// CHECK:STDOUT:   %Enum.Op.bound.ff6: <bound method> = bound_method %int_0, %Enum.Op [concrete]
// CHECK:STDOUT:   %int_1: %Enum = int_value 1 [concrete]
// CHECK:STDOUT:   %Enum.Op.bound.d4f: <bound method> = bound_method %int_1, %Enum.Op [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.type: type = fn_type @Destroy.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy: %Destroy.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Enum.type.facet: %type = facet_value %Enum, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.3d2: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Enum.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.396: %Destroy.WithSelf.SelfDestruct.type.3d2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ce5: <witness> = custom_witness (%Destroy.Op, %Destroy.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.396), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.672: %Destroy.type = facet_value %Enum, (%custom_witness.ce5) [concrete]
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
// CHECK:STDOUT:   %impl.elem0.loc8: %.1e6 = impl_witness_access constants.%custom_witness.d0d, element0 [concrete = constants.%Enum.Op]
// CHECK:STDOUT:   %bound_method.loc8_29: <bound method> = bound_method %a.ref.loc8, %impl.elem0.loc8 [concrete = constants.%Enum.Op.bound.ff6]
// CHECK:STDOUT:   %Enum.Op.call.loc8: init %Enum = call %bound_method.loc8_29(%a.ref.loc8) [concrete = constants.%int_0]
// CHECK:STDOUT:   assign %a.var, %Enum.Op.call.loc8
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
// CHECK:STDOUT:   %impl.elem0.loc10: %.1e6 = impl_witness_access constants.%custom_witness.d0d, element0 [concrete = constants.%Enum.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %b.ref, %impl.elem0.loc10 [concrete = constants.%Enum.Op.bound.d4f]
// CHECK:STDOUT:   %Enum.Op.call.loc10: init %Enum = call %bound_method.loc10(%b.ref) [concrete = constants.%int_1]
// CHECK:STDOUT:   assign %a.ref.loc10, %Enum.Op.call.loc10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.396
// CHECK:STDOUT:   %Destroy.facet.loc8_3.1: %Destroy.type = facet_value constants.%Enum, (constants.%custom_witness.ce5) [concrete = constants.%Destroy.facet.672]
// CHECK:STDOUT:   %.loc8_3.1: %Destroy.type = converted constants.%Enum, %Destroy.facet.loc8_3.1 [concrete = constants.%Destroy.facet.672]
// CHECK:STDOUT:   %Destroy.facet.loc8_3.2: %Destroy.type = facet_value constants.%Enum, (constants.%custom_witness.ce5) [concrete = constants.%Destroy.facet.672]
// CHECK:STDOUT:   %.loc8_3.2: %Destroy.type = converted constants.%Enum, %Destroy.facet.loc8_3.2 [concrete = constants.%Destroy.facet.672]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_3: <bound method> = bound_method %a.var, %Destroy.WithSelf.SelfDestruct.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc8_3(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy(%self.param: ref %Enum) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Enum) = "no_op";
// CHECK:STDOUT:
