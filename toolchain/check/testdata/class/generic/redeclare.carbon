// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/redeclare.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/redeclare.carbon

// --- valid.carbon

library "[[@TEST_NAME]]";

class Generic(T:! type);

class Generic(T:! type) {
}

// --- fail_mismatch_param_list.carbon

library "[[@TEST_NAME]]";

class A;
// CHECK:STDERR: fail_mismatch_param_list.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: class A(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_param_list.carbon:[[@LINE-4]]:1: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR: class A;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
class A(T:! type) {}

// --- fail_mismatch_implicit_param_list.carbon

library "[[@TEST_NAME]]";

class A {}

class B(N:! A);
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE+7]]:1: error: redeclaration differs because of implicit parameter list [RedeclParamListDiffers]
// CHECK:STDERR: class B[T:! type](N:! T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE-4]]:1: note: previously declared without implicit parameter list [RedeclParamListPrevious]
// CHECK:STDERR: class B(N:! A);
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
class B[T:! type](N:! T) {}

// --- fail_mismatch_param_count.carbon

library "[[@TEST_NAME]]";

class A {}

class C(T:! type);
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter count of 2 [RedeclParamCountDiffers]
// CHECK:STDERR: class C(T:! type, U:! A) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE-4]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: class C(T:! type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class C(T:! type, U:! A) {}

// --- fail_mismatch_param_type.carbon

library "[[@TEST_NAME]]";

class A {}

class D(T:! type);
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE+7]]:9: error: type `<pattern for A>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for type>` [RedeclParamDiffersType]
// CHECK:STDERR: class D(T:! A) {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE-4]]:9: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class D(T:! type);
// CHECK:STDERR:         ^
// CHECK:STDERR:
class D(T:! A) {}

// --- fail_mismatch_param_name.carbon

library "[[@TEST_NAME]]";

class E(T:! type);
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE+7]]:9: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: class E(U:! type) {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE-4]]:9: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class E(T:! type);
// CHECK:STDERR:         ^
// CHECK:STDERR:
class E(U:! type) {}

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Generic.%Destroy.impl_witness_table, @Generic.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.881: type = ptr_type %Generic [symbolic]
// CHECK:STDOUT:   %pattern_type.b64: type = pattern_type %ptr.881 [symbolic]
// CHECK:STDOUT:   %Generic.as.Destroy.impl.Op.type: type = fn_type @Generic.as.Destroy.impl.Op, @Generic.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Generic.as.Destroy.impl.Op: %Generic.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Generic = %Generic.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl.loc4: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.2: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc6: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Generic.as.Destroy.impl(@Generic.%T.loc6: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T) [symbolic = %Generic (constants.%Generic)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Generic.%Destroy.impl_witness_table, @Generic.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.as.Destroy.impl.Op.type: type = fn_type @Generic.as.Destroy.impl.Op, @Generic.as.Destroy.impl(%T) [symbolic = %Generic.as.Destroy.impl.Op.type (constants.%Generic.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Generic.as.Destroy.impl.Op: @Generic.as.Destroy.impl.%Generic.as.Destroy.impl.Op.type (%Generic.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Generic.as.Destroy.impl.Op (constants.%Generic.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Generic.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Generic.as.Destroy.impl.Op.decl: @Generic.as.Destroy.impl.%Generic.as.Destroy.impl.Op.type (%Generic.as.Destroy.impl.Op.type) = fn_decl @Generic.as.Destroy.impl.Op [symbolic = @Generic.as.Destroy.impl.%Generic.as.Destroy.impl.Op (constants.%Generic.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Generic.as.Destroy.impl.Op.%pattern_type (%pattern_type.b64) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Generic.as.Destroy.impl.Op.%pattern_type (%pattern_type.b64) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc6_25.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Generic.as.Destroy.impl.Op.%ptr (%ptr.881) = value_param call_param0
// CHECK:STDOUT:       %.loc6_25.2: type = splice_block %Self.ref [symbolic = %Generic (constants.%Generic)] {
// CHECK:STDOUT:         %.loc6_25.3: type = specific_constant constants.%Generic, @Generic(constants.%T) [symbolic = %Generic (constants.%Generic)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_25.3 [symbolic = %Generic (constants.%Generic)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Generic.as.Destroy.impl.Op.%ptr (%ptr.881) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Generic.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Generic.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Generic [symbolic = @Generic.as.Destroy.impl.%Generic (constants.%Generic)]
// CHECK:STDOUT:     impl_decl @Generic.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Generic.as.Destroy.impl.%Generic.as.Destroy.impl.Op.decl), @Generic.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Generic.as.Destroy.impl(constants.%T) [symbolic = @Generic.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic.as.Destroy.impl.Op(@Generic.%T.loc6: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T) [symbolic = %Generic (constants.%Generic)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Generic [symbolic = %ptr (constants.%ptr.881)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.b64)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Generic.as.Destroy.impl.Op.%ptr (%ptr.881)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Generic => constants.%Generic
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Generic => constants.%Generic
// CHECK:STDOUT:   %ptr => constants.%ptr.881
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.466: type = class_type @A.loc4 [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A.loc12 [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.130: type = class_type @A.loc12, @A.loc12(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @A.loc12.%Destroy.impl_witness_table, @A.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.ca9: type = ptr_type %A.130 [symbolic]
// CHECK:STDOUT:   %pattern_type.09b: type = pattern_type %ptr.ca9 [symbolic]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op, @A.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc4: type = class_decl @A.loc4 [concrete = constants.%A.466] {} {}
// CHECK:STDOUT:   %A.decl.loc12: %A.type = class_decl @A.loc12 [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc12_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @A.as.Destroy.impl(@A.loc12.%T.loc12_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %A: type = class_type @A.loc12, @A.loc12(%T) [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @A.loc12.%Destroy.impl_witness_table, @A.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op, @A.as.Destroy.impl(%T) [symbolic = %A.as.Destroy.impl.Op.type (constants.%A.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: @A.as.Destroy.impl.%A.as.Destroy.impl.Op.type (%A.as.Destroy.impl.Op.type) = struct_value () [symbolic = %A.as.Destroy.impl.Op (constants.%A.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @A.loc12.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %A.as.Destroy.impl.Op.decl: @A.as.Destroy.impl.%A.as.Destroy.impl.Op.type (%A.as.Destroy.impl.Op.type) = fn_decl @A.as.Destroy.impl.Op [symbolic = @A.as.Destroy.impl.%A.as.Destroy.impl.Op (constants.%A.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @A.as.Destroy.impl.Op.%pattern_type (%pattern_type.09b) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @A.as.Destroy.impl.Op.%pattern_type (%pattern_type.09b) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc12_19.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @A.as.Destroy.impl.Op.%ptr (%ptr.ca9) = value_param call_param0
// CHECK:STDOUT:       %.loc12_19.2: type = splice_block %Self.ref [symbolic = %A (constants.%A.130)] {
// CHECK:STDOUT:         %.loc12_19.3: type = specific_constant constants.%A.130, @A.loc12(constants.%T) [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc12_19.3 [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @A.as.Destroy.impl.Op.%ptr (%ptr.ca9) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %A.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @A.loc12.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.loc4;
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A.loc12(%T.loc12_9.2: type) {
// CHECK:STDOUT:   %T.loc12_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A.130 [symbolic = @A.as.Destroy.impl.%A (constants.%A.130)]
// CHECK:STDOUT:     impl_decl @A.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@A.as.Destroy.impl.%A.as.Destroy.impl.Op.decl), @A.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @A.as.Destroy.impl(constants.%T) [symbolic = @A.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.130
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.as.Destroy.impl.Op(@A.loc12.%T.loc12_9.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %A: type = class_type @A.loc12, @A.loc12(%T) [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:   %ptr: type = ptr_type %A [symbolic = %ptr (constants.%ptr.ca9)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.09b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @A.as.Destroy.impl.Op.%ptr (%ptr.ca9)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.loc12(constants.%T) {
// CHECK:STDOUT:   %T.loc12_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %A => constants.%A.130
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %A => constants.%A.130
// CHECK:STDOUT:   %ptr => constants.%ptr.ca9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.09b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_implicit_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.cbd: <witness> = impl_witness @A.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %N.9e6: %A = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %B.type.844c0f.1: type = generic_class_type @B.loc6 [concrete]
// CHECK:STDOUT:   %B.generic.ba299b.1: %B.type.844c0f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %N.f22: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %B.type.844c0f.2: type = generic_class_type @B.loc14 [concrete]
// CHECK:STDOUT:   %B.generic.ba299b.2: %B.type.844c0f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %B.828: type = class_type @B.loc14, @B.loc14(%T, %N.f22) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.291: <witness> = impl_witness @B.loc14.%Destroy.impl_witness_table, @B.as.Destroy.impl(%T, %N.f22) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %ptr.9e8: type = ptr_type %B.828 [symbolic]
// CHECK:STDOUT:   %pattern_type.a6d: type = pattern_type %ptr.9e8 [symbolic]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op, @B.as.Destroy.impl(%T, %N.f22) [symbolic]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: %B.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc6: %B.type.844c0f.1 = class_decl @B.loc6 [concrete = constants.%B.generic.ba299b.1] {
// CHECK:STDOUT:     %N.patt: %pattern_type.c10 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_9.2: %A = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.1 (constants.%N.9e6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc14: %B.type.844c0f.2 = class_decl @B.loc14 [concrete = constants.%B.generic.ba299b.2] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: @B.loc14.%pattern_type (%pattern_type.7dc) = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc14_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc14: type = splice_block %T.ref [symbolic = %T.loc14_9.1 (constants.%T)] {
// CHECK:STDOUT:       %.Self.2: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc14_9.2 [symbolic = %T.loc14_9.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc14_19.2: @B.loc14.%T.loc14_9.1 (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc14_19.1 (constants.%N.f22)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Destroy.impl: @A.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.decl: %A.as.Destroy.impl.Op.type = fn_decl @A.as.Destroy.impl.Op [concrete = constants.%A.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %A.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @A.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @B.as.Destroy.impl(@B.loc14.%T.loc14_9.2: type, @B.loc14.%N.loc14_19.2: @B.loc14.%T.loc14_9.1 (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N: @B.as.Destroy.impl.%T (%T) = bind_symbolic_name N, 1 [symbolic = %N (constants.%N.f22)]
// CHECK:STDOUT:   %B: type = class_type @B.loc14, @B.loc14(%T, %N) [symbolic = %B (constants.%B.828)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @B.loc14.%Destroy.impl_witness_table, @B.as.Destroy.impl(%T, %N) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.291)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.as.Destroy.impl.Op.type: type = fn_type @B.as.Destroy.impl.Op, @B.as.Destroy.impl(%T, %N) [symbolic = %B.as.Destroy.impl.Op.type (constants.%B.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %B.as.Destroy.impl.Op: @B.as.Destroy.impl.%B.as.Destroy.impl.Op.type (%B.as.Destroy.impl.Op.type) = struct_value () [symbolic = %B.as.Destroy.impl.Op (constants.%B.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @B.loc14.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %B.as.Destroy.impl.Op.decl: @B.as.Destroy.impl.%B.as.Destroy.impl.Op.type (%B.as.Destroy.impl.Op.type) = fn_decl @B.as.Destroy.impl.Op [symbolic = @B.as.Destroy.impl.%B.as.Destroy.impl.Op (constants.%B.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @B.as.Destroy.impl.Op.%pattern_type (%pattern_type.a6d) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @B.as.Destroy.impl.Op.%pattern_type (%pattern_type.a6d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc14_26.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @B.as.Destroy.impl.Op.%ptr (%ptr.9e8) = value_param call_param0
// CHECK:STDOUT:       %.loc14_26.2: type = splice_block %Self.ref [symbolic = %B (constants.%B.828)] {
// CHECK:STDOUT:         %.loc14_26.3: type = specific_constant constants.%B.828, @B.loc14(constants.%T, constants.%N.f22) [symbolic = %B (constants.%B.828)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc14_26.3 [symbolic = %B (constants.%B.828)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @B.as.Destroy.impl.Op.%ptr (%ptr.9e8) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %B.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @B.loc14.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:   impl_decl @A.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@A.as.Destroy.impl.%A.as.Destroy.impl.Op.decl), @A.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.cbd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B.loc6(%N.loc6_9.2: %A) {
// CHECK:STDOUT:   %N.loc6_9.1: %A = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.1 (constants.%N.9e6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B.loc14(%T.loc14_9.2: type, %N.loc14_19.2: @B.loc14.%T.loc14_9.1 (%T)) {
// CHECK:STDOUT:   %T.loc14_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc14_19.1: @B.loc14.%T.loc14_9.1 (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc14_19.1 (constants.%N.f22)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc14_9.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc14_9.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B.828 [symbolic = @B.as.Destroy.impl.%B (constants.%B.828)]
// CHECK:STDOUT:     impl_decl @B.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@B.as.Destroy.impl.%B.as.Destroy.impl.Op.decl), @B.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @B.as.Destroy.impl(constants.%T, constants.%N.f22) [symbolic = @B.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.291)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B.828
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Destroy.impl.Op(%self.param: %ptr.6db) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.as.Destroy.impl.Op(@B.loc14.%T.loc14_9.2: type, @B.loc14.%N.loc14_19.2: @B.loc14.%T.loc14_9.1 (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N: @B.as.Destroy.impl.Op.%T (%T) = bind_symbolic_name N, 1 [symbolic = %N (constants.%N.f22)]
// CHECK:STDOUT:   %B: type = class_type @B.loc14, @B.loc14(%T, %N) [symbolic = %B (constants.%B.828)]
// CHECK:STDOUT:   %ptr: type = ptr_type %B [symbolic = %ptr (constants.%ptr.9e8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.a6d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @B.as.Destroy.impl.Op.%ptr (%ptr.9e8)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.loc6(constants.%N.9e6) {
// CHECK:STDOUT:   %N.loc6_9.1 => constants.%N.9e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.loc14(constants.%T, constants.%N.f22) {
// CHECK:STDOUT:   %T.loc14_9.1 => constants.%T
// CHECK:STDOUT:   %N.loc14_19.1 => constants.%N.f22
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.as.Destroy.impl(constants.%T, constants.%N.f22) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N.f22
// CHECK:STDOUT:   %B => constants.%B.828
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.291
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.as.Destroy.impl.Op(constants.%T, constants.%N.f22) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N => constants.%N.f22
// CHECK:STDOUT:   %B => constants.%B.828
// CHECK:STDOUT:   %ptr => constants.%ptr.9e8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.cbd: <witness> = impl_witness @A.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type.e6e560.1: type = generic_class_type @C.loc6 [concrete]
// CHECK:STDOUT:   %C.generic.965b12.1: %C.type.e6e560.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U: %A = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %C.type.e6e560.2: type = generic_class_type @C.loc14 [concrete]
// CHECK:STDOUT:   %C.generic.965b12.2: %C.type.e6e560.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.3d8: type = class_type @C.loc14, @C.loc14(%T, %U) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.36e: <witness> = impl_witness @C.loc14.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %ptr.8f5: type = ptr_type %C.3d8 [symbolic]
// CHECK:STDOUT:   %pattern_type.d5b: type = pattern_type %ptr.8f5 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T, %U) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .C = %C.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %C.decl.loc6: %C.type.e6e560.1 = class_decl @C.loc6 [concrete = constants.%C.generic.965b12.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc14: %C.type.e6e560.2 = class_decl @C.loc14 [concrete = constants.%C.generic.965b12.2] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.c10 = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc14_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc14: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:       %.Self.2: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc14_19.2: %A = bind_symbolic_name U, 1 [symbolic = %U.loc14_19.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Destroy.impl: @A.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.decl: %A.as.Destroy.impl.Op.type = fn_decl @A.as.Destroy.impl.Op [concrete = constants.%A.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %A.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @A.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.loc14.%T.loc14_9.2: type, @C.loc14.%U.loc14_19.2: %A) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U: %A = bind_symbolic_name U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %C: type = class_type @C.loc14, @C.loc14(%T, %U) [symbolic = %C (constants.%C.3d8)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.loc14.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T, %U) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.36e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T, %U) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @C.loc14.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d5b) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d5b) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc14_26.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.8f5) = value_param call_param0
// CHECK:STDOUT:       %.loc14_26.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.3d8)] {
// CHECK:STDOUT:         %.loc14_26.3: type = specific_constant constants.%C.3d8, @C.loc14(constants.%T, constants.%U) [symbolic = %C (constants.%C.3d8)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc14_26.3 [symbolic = %C (constants.%C.3d8)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.8f5) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.loc14.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:   impl_decl @A.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@A.as.Destroy.impl.%A.as.Destroy.impl.Op.decl), @A.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.cbd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C.loc6(%T.loc6_9.2: type) {
// CHECK:STDOUT:   %T.loc6_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C.loc14(%T.loc14_9.2: type, %U.loc14_19.2: %A) {
// CHECK:STDOUT:   %T.loc14_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc14_19.1: %A = bind_symbolic_name U, 1 [symbolic = %U.loc14_19.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C.3d8 [symbolic = @C.as.Destroy.impl.%C (constants.%C.3d8)]
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%T, constants.%U) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.36e)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3d8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Destroy.impl.Op(%self.param: %ptr.6db) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.loc14.%T.loc14_9.2: type, @C.loc14.%U.loc14_19.2: %A) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %U: %A = bind_symbolic_name U, 1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %C: type = class_type @C.loc14, @C.loc14(%T, %U) [symbolic = %C (constants.%C.3d8)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.8f5)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d5b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.8f5)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.loc6(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.loc14(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc14_9.1 => constants.%T
// CHECK:STDOUT:   %U.loc14_19.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %C => constants.%C.3d8
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.36e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %C => constants.%C.3d8
// CHECK:STDOUT:   %ptr => constants.%ptr.8f5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d5b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.cbd: <witness> = impl_witness @A.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.type: type = fn_type @A.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %A.as.Destroy.impl.Op: %A.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %D.type.bbd080.1: type = generic_class_type @D.loc6 [concrete]
// CHECK:STDOUT:   %D.generic.4e2319.1: %D.type.bbd080.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.9e6: %A = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %D.type.bbd080.2: type = generic_class_type @D.loc14 [concrete]
// CHECK:STDOUT:   %D.generic.4e2319.2: %D.type.bbd080.2 = struct_value () [concrete]
// CHECK:STDOUT:   %D.384: type = class_type @D.loc14, @D.loc14(%T.9e6) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.bb0: <witness> = impl_witness @D.loc14.%Destroy.impl_witness_table, @D.as.Destroy.impl(%T.9e6) [symbolic]
// CHECK:STDOUT:   %ptr.988: type = ptr_type %D.384 [symbolic]
// CHECK:STDOUT:   %pattern_type.146: type = pattern_type %ptr.988 [symbolic]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op, @D.as.Destroy.impl(%T.9e6) [symbolic]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .D = %D.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %D.decl.loc6: %D.type.bbd080.1 = class_decl @D.loc6 [concrete = constants.%D.generic.4e2319.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc14: %D.type.bbd080.2 = class_decl @D.loc14 [concrete = constants.%D.generic.4e2319.2] {
// CHECK:STDOUT:     %T.patt: %pattern_type.c10 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_9.2: %A = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.1 (constants.%T.9e6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.Destroy.impl: @A.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %A.as.Destroy.impl.Op.decl: %A.as.Destroy.impl.Op.type = fn_decl @A.as.Destroy.impl.Op [concrete = constants.%A.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %A.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @A.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @D.as.Destroy.impl(@D.loc14.%T.loc14_9.2: %A) {
// CHECK:STDOUT:   %T: %A = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.9e6)]
// CHECK:STDOUT:   %D: type = class_type @D.loc14, @D.loc14(%T) [symbolic = %D (constants.%D.384)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @D.loc14.%Destroy.impl_witness_table, @D.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.bb0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op, @D.as.Destroy.impl(%T) [symbolic = %D.as.Destroy.impl.Op.type (constants.%D.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: @D.as.Destroy.impl.%D.as.Destroy.impl.Op.type (%D.as.Destroy.impl.Op.type) = struct_value () [symbolic = %D.as.Destroy.impl.Op (constants.%D.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @D.loc14.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %D.as.Destroy.impl.Op.decl: @D.as.Destroy.impl.%D.as.Destroy.impl.Op.type (%D.as.Destroy.impl.Op.type) = fn_decl @D.as.Destroy.impl.Op [symbolic = @D.as.Destroy.impl.%D.as.Destroy.impl.Op (constants.%D.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @D.as.Destroy.impl.Op.%pattern_type (%pattern_type.146) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @D.as.Destroy.impl.Op.%pattern_type (%pattern_type.146) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc14_16.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @D.as.Destroy.impl.Op.%ptr (%ptr.988) = value_param call_param0
// CHECK:STDOUT:       %.loc14_16.2: type = splice_block %Self.ref [symbolic = %D (constants.%D.384)] {
// CHECK:STDOUT:         %.loc14_16.3: type = specific_constant constants.%D.384, @D.loc14(constants.%T.9e6) [symbolic = %D (constants.%D.384)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc14_16.3 [symbolic = %D (constants.%D.384)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @D.as.Destroy.impl.Op.%ptr (%ptr.988) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @D.loc14.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:   impl_decl @A.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@A.as.Destroy.impl.%A.as.Destroy.impl.Op.decl), @A.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.cbd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.loc6(%T.loc6_9.2: type) {
// CHECK:STDOUT:   %T.loc6_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @D.loc14(%T.loc14_9.2: %A) {
// CHECK:STDOUT:   %T.loc14_9.1: %A = bind_symbolic_name T, 0 [symbolic = %T.loc14_9.1 (constants.%T.9e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D.384 [symbolic = @D.as.Destroy.impl.%D (constants.%D.384)]
// CHECK:STDOUT:     impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @D.as.Destroy.impl(constants.%T.9e6) [symbolic = @D.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.bb0)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%D.384
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.Destroy.impl.Op(%self.param: %ptr.6db) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @D.as.Destroy.impl.Op(@D.loc14.%T.loc14_9.2: %A) {
// CHECK:STDOUT:   %T: %A = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.9e6)]
// CHECK:STDOUT:   %D: type = class_type @D.loc14, @D.loc14(%T) [symbolic = %D (constants.%D.384)]
// CHECK:STDOUT:   %ptr: type = ptr_type %D [symbolic = %ptr (constants.%ptr.988)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.146)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @D.as.Destroy.impl.Op.%ptr (%ptr.988)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.loc6(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.loc14(constants.%T.9e6) {
// CHECK:STDOUT:   %T.loc14_9.1 => constants.%T.9e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.Destroy.impl(constants.%T.9e6) {
// CHECK:STDOUT:   %T => constants.%T.9e6
// CHECK:STDOUT:   %D => constants.%D.384
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.bb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @D.as.Destroy.impl.Op(constants.%T.9e6) {
// CHECK:STDOUT:   %T => constants.%T.9e6
// CHECK:STDOUT:   %D => constants.%D.384
// CHECK:STDOUT:   %ptr => constants.%ptr.988
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %E.type.b0f8dc.1: type = generic_class_type @E.loc4 [concrete]
// CHECK:STDOUT:   %E.generic.f281ba.1: %E.type.b0f8dc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %E.type.b0f8dc.2: type = generic_class_type @E.loc12 [concrete]
// CHECK:STDOUT:   %E.generic.f281ba.2: %E.type.b0f8dc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %E.ec9c10.2: type = class_type @E.loc12, @E.loc12(%U) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @E.loc12.%Destroy.impl_witness_table, @E.as.Destroy.impl(%U) [symbolic]
// CHECK:STDOUT:   %ptr.878: type = ptr_type %E.ec9c10.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.3d9: type = pattern_type %ptr.878 [symbolic]
// CHECK:STDOUT:   %E.as.Destroy.impl.Op.type: type = fn_type @E.as.Destroy.impl.Op, @E.as.Destroy.impl(%U) [symbolic]
// CHECK:STDOUT:   %E.as.Destroy.impl.Op: %E.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .E = %E.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %E.decl.loc4: %E.type.b0f8dc.1 = class_decl @E.loc4 [concrete = constants.%E.generic.f281ba.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl.loc12: %E.type.b0f8dc.2 = class_decl @E.loc12 [concrete = constants.%E.generic.f281ba.2] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc12_9.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_9.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @E.as.Destroy.impl(@E.loc12.%U.loc12_9.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %E: type = class_type @E.loc12, @E.loc12(%U) [symbolic = %E (constants.%E.ec9c10.2)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @E.loc12.%Destroy.impl_witness_table, @E.as.Destroy.impl(%U) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %E.as.Destroy.impl.Op.type: type = fn_type @E.as.Destroy.impl.Op, @E.as.Destroy.impl(%U) [symbolic = %E.as.Destroy.impl.Op.type (constants.%E.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %E.as.Destroy.impl.Op: @E.as.Destroy.impl.%E.as.Destroy.impl.Op.type (%E.as.Destroy.impl.Op.type) = struct_value () [symbolic = %E.as.Destroy.impl.Op (constants.%E.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @E.loc12.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %E.as.Destroy.impl.Op.decl: @E.as.Destroy.impl.%E.as.Destroy.impl.Op.type (%E.as.Destroy.impl.Op.type) = fn_decl @E.as.Destroy.impl.Op [symbolic = @E.as.Destroy.impl.%E.as.Destroy.impl.Op (constants.%E.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @E.as.Destroy.impl.Op.%pattern_type (%pattern_type.3d9) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @E.as.Destroy.impl.Op.%pattern_type (%pattern_type.3d9) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc12_19.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @E.as.Destroy.impl.Op.%ptr (%ptr.878) = value_param call_param0
// CHECK:STDOUT:       %.loc12_19.2: type = splice_block %Self.ref [symbolic = %E (constants.%E.ec9c10.2)] {
// CHECK:STDOUT:         %.loc12_19.3: type = specific_constant constants.%E.ec9c10.2, @E.loc12(constants.%U) [symbolic = %E (constants.%E.ec9c10.2)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc12_19.3 [symbolic = %E (constants.%E.ec9c10.2)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @E.as.Destroy.impl.Op.%ptr (%ptr.878) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %E.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @E.loc12.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E.loc4(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @E.loc12(%U.loc12_9.2: type) {
// CHECK:STDOUT:   %U.loc12_9.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc12_9.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%E.ec9c10.2 [symbolic = @E.as.Destroy.impl.%E (constants.%E.ec9c10.2)]
// CHECK:STDOUT:     impl_decl @E.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@E.as.Destroy.impl.%E.as.Destroy.impl.Op.decl), @E.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @E.as.Destroy.impl(constants.%U) [symbolic = @E.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%E.ec9c10.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @E.as.Destroy.impl.Op(@E.loc12.%U.loc12_9.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %E: type = class_type @E.loc12, @E.loc12(%U) [symbolic = %E (constants.%E.ec9c10.2)]
// CHECK:STDOUT:   %ptr: type = ptr_type %E [symbolic = %ptr (constants.%ptr.878)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.3d9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @E.as.Destroy.impl.Op.%ptr (%ptr.878)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.loc4(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.loc12(constants.%U) {
// CHECK:STDOUT:   %U.loc12_9.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.as.Destroy.impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %E => constants.%E.ec9c10.2
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @E.as.Destroy.impl.Op(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %E => constants.%E.ec9c10.2
// CHECK:STDOUT:   %ptr => constants.%ptr.878
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
