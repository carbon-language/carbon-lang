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

// --- fail_todo_copy_enum.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

//@dump-sem-ir-begin
fn F() {
  // CHECK:STDERR: fail_todo_copy_enum.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.Destroy` in type `Cpp.Enum` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   var a: Cpp.Enum = Cpp.Enum.a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a: Cpp.Enum = Cpp.Enum.a;

  a = Cpp.Enum.b;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_todo_copy_enum.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Enum: type = class_type @Enum [concrete]
// CHECK:STDOUT:   %pattern_type.ebf: type = pattern_type %Enum [concrete]
// CHECK:STDOUT:   %int_0: %Enum = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Enum.Op.type: type = fn_type @Enum.Op [concrete]
// CHECK:STDOUT:   %Enum.Op: %Enum.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0a8: <witness> = custom_witness (%Enum.Op), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.897: %Copy.type = facet_value %Enum, (%custom_witness.0a8) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.897) [concrete]
// CHECK:STDOUT:   %.76d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67c, %Copy.facet.897 [concrete]
// CHECK:STDOUT:   %Enum.Op.bound.3a0: <bound method> = bound_method %int_0, %Enum.Op [concrete]
// CHECK:STDOUT:   %int_1: %Enum = int_value 1 [concrete]
// CHECK:STDOUT:   %Enum.Op.bound.87b: <bound method> = bound_method %int_1, %Enum.Op [concrete]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.ebf = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ebf = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %Enum = var %a.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc12_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Enum.ref.loc12_24: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   %a.ref.loc12: %Enum = name_ref a, imports.%int_0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc12: %.76d = impl_witness_access constants.%custom_witness.0a8, element0 [concrete = constants.%Enum.Op]
// CHECK:STDOUT:   %bound_method.loc12: <bound method> = bound_method %a.ref.loc12, %impl.elem0.loc12 [concrete = constants.%Enum.Op.bound.3a0]
// CHECK:STDOUT:   %Enum.Op.call.loc12: init %Enum = call %bound_method.loc12(%a.ref.loc12) [concrete = constants.%int_0]
// CHECK:STDOUT:   assign %a.var, %Enum.Op.call.loc12
// CHECK:STDOUT:   %.loc12: type = splice_block %Enum.ref.loc12_13 [concrete = constants.%Enum] {
// CHECK:STDOUT:     %Cpp.ref.loc12_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %Enum.ref.loc12_13: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Enum = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref.loc14: ref %Enum = name_ref a, %a
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Enum.ref.loc14: type = name_ref Enum, imports.%Enum.decl [concrete = constants.%Enum]
// CHECK:STDOUT:   %b.ref: %Enum = name_ref b, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc14: %.76d = impl_witness_access constants.%custom_witness.0a8, element0 [concrete = constants.%Enum.Op]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %b.ref, %impl.elem0.loc14 [concrete = constants.%Enum.Op.bound.87b]
// CHECK:STDOUT:   %Enum.Op.call.loc14: init %Enum = call %bound_method.loc14(%b.ref) [concrete = constants.%int_1]
// CHECK:STDOUT:   assign %a.ref.loc14, %Enum.Op.call.loc14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Enum.Op(%self.param: %Enum) -> out %return.param: %Enum = "primitive_copy";
// CHECK:STDOUT:
