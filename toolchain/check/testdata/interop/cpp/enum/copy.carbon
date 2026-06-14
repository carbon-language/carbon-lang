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
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.926 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.926 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Enum = var %a.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc8_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Enum.ref.loc8_24: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   %a.ref.loc8: %Enum = name_ref a, imports.%int_0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc8: %.1e6 = impl_witness_access constants.%custom_witness.d0d, element0 [concrete = constants.%Enum.Op]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %a.ref.loc8, %impl.elem0.loc8 [concrete = constants.%Enum.Op.bound.ff6]
// CHECK:STDOUT:   %Enum.Op.call.loc8: init %Enum = call %bound_method.loc8(%a.ref.loc8) [concrete = constants.%int_0]
// CHECK:STDOUT:   assign %a.var, %Enum.Op.call.loc8
// CHECK:STDOUT:   %.loc8: type = splice_block %Enum.ref.loc8_13 [concrete = constants.%Enum] {
// CHECK:STDOUT:     %Cpp.ref.loc8_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Enum.ref.loc8_13: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Enum = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc10: ref %Enum = name_ref a, %a
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Enum.ref.loc10: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   %b.ref: %Enum = name_ref b, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc10: %.1e6 = impl_witness_access constants.%custom_witness.d0d, element0 [concrete = constants.%Enum.Op]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %b.ref, %impl.elem0.loc10 [concrete = constants.%Enum.Op.bound.d4f]
// CHECK:STDOUT:   %Enum.Op.call.loc10: init %Enum = call %bound_method.loc10(%b.ref) [concrete = constants.%int_1]
// CHECK:STDOUT:   assign %a.ref.loc10, %Enum.Op.call.loc10
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Enum.Op(%self.param: %Enum) -> out %return.param: %Enum = "primitive_copy";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Enum) = "no_op";
// CHECK:STDOUT:
