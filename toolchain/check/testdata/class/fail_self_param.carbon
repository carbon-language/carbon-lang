// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_self_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_self_param.carbon

// CHECK:STDERR: fail_self_param.carbon:[[@LINE+3]]:9: error(SelfParameterNotAllowed): `self` parameter only allowed on functions
// CHECK:STDERR: class C[self:! type](x:! self) {}
// CHECK:STDERR:         ^~~~~~~~~~~
class C[self:! type](x:! self) {}
var v: C(0);

// CHECK:STDOUT: --- fail_self_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %self: type = bind_symbolic_name self, 0 [symbolic]
// CHECK:STDOUT:   %self.patt: type = symbolic_binding_pattern self, 0 [symbolic]
// CHECK:STDOUT:   %x: %self = bind_symbolic_name x, 1 [symbolic]
// CHECK:STDOUT:   %x.patt: %self = symbolic_binding_pattern x, 1 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%self, %x) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(i32, %.4) [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %self.patt.loc14_9.1: type = symbolic_binding_pattern self, 0 [symbolic = %self.patt.loc14_9.2 (constants.%self.patt)]
// CHECK:STDOUT:     %self.param_patt: type = param_pattern %self.patt.loc14_9.1, runtime_param<invalid> [symbolic = %self.patt.loc14_9.2 (constants.%self.patt)]
// CHECK:STDOUT:     %x.patt.loc14_22.1: @C.%self.loc14_9.2 (%self) = symbolic_binding_pattern x, 1 [symbolic = %x.patt.loc14_22.2 (constants.%x.patt)]
// CHECK:STDOUT:     %x.param_patt: @C.%self.loc14_9.2 (%self) = param_pattern %x.patt.loc14_22.1, runtime_param<invalid> [symbolic = %x.patt.loc14_22.2 (constants.%x.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.ref: type = name_ref self, %self.loc14_9.1 [symbolic = %self.loc14_9.2 (constants.%self)]
// CHECK:STDOUT:     %param.loc14_9: type = param runtime_param<invalid>
// CHECK:STDOUT:     %self.loc14_9.1: type = bind_symbolic_name self, 0, %param.loc14_9 [symbolic = %self.loc14_9.2 (constants.%self)]
// CHECK:STDOUT:     %param.loc14_22: @C.%self.loc14_9.2 (%self) = param runtime_param<invalid>
// CHECK:STDOUT:     %x.loc14_22.1: @C.%self.loc14_9.2 (%self) = bind_symbolic_name x, 1, %param.loc14_22 [symbolic = %x.loc14_22.2 (constants.%x)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, %C.decl [template = constants.%C.1]
// CHECK:STDOUT:   %.loc15: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %C: type = class_type @C, @C(i32, constants.%.4) [template = constants.%C.3]
// CHECK:STDOUT:   %v.var: ref %C.3 = var v
// CHECK:STDOUT:   %v: ref %C.3 = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%self.loc14_9.1: type, %x.loc14_22.1: @C.%self.loc14_9.2 (%self)) {
// CHECK:STDOUT:   %self.loc14_9.2: type = bind_symbolic_name self, 0 [symbolic = %self.loc14_9.2 (constants.%self)]
// CHECK:STDOUT:   %self.patt.loc14_9.2: type = symbolic_binding_pattern self, 0 [symbolic = %self.patt.loc14_9.2 (constants.%self.patt)]
// CHECK:STDOUT:   %x.loc14_22.2: %self = bind_symbolic_name x, 1 [symbolic = %x.loc14_22.2 (constants.%x)]
// CHECK:STDOUT:   %x.patt.loc14_22.2: %self = symbolic_binding_pattern x, 1 [symbolic = %x.patt.loc14_22.2 (constants.%x.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc14: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%self, constants.%x) {
// CHECK:STDOUT:   %self.loc14_9.2 => constants.%self
// CHECK:STDOUT:   %self.patt.loc14_9.2 => constants.%self
// CHECK:STDOUT:   %x.loc14_22.2 => constants.%x
// CHECK:STDOUT:   %x.patt.loc14_22.2 => constants.%x
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(i32, constants.%.4) {
// CHECK:STDOUT:   %self.loc14_9.2 => i32
// CHECK:STDOUT:   %self.patt.loc14_9.2 => i32
// CHECK:STDOUT:   %x.loc14_22.2 => constants.%.4
// CHECK:STDOUT:   %x.patt.loc14_22.2 => constants.%.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
