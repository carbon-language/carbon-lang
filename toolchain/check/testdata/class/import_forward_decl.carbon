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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/import_forward_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/import_forward_decl.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class ForwardDecl;

// --- a.impl.carbon

impl library "[[@TEST_NAME]]";

class ForwardDecl {
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardDecl: type = class_type @ForwardDecl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ForwardDecl = %ForwardDecl.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ForwardDecl.decl: type = class_decl @ForwardDecl [concrete = constants.%ForwardDecl] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDecl;
// CHECK:STDOUT:
// CHECK:STDOUT: --- a.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardDecl: type = class_type @ForwardDecl [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @ForwardDecl.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.83c: type = ptr_type %ForwardDecl [concrete]
// CHECK:STDOUT:   %pattern_type.fd7: type = pattern_type %ptr.83c [concrete]
// CHECK:STDOUT:   %ForwardDecl.as.Destroy.impl.Op.type: type = fn_type @ForwardDecl.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %ForwardDecl.as.Destroy.impl.Op: %ForwardDecl.as.Destroy.impl.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .ForwardDecl = %ForwardDecl.decl
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import.loc2_17.1 = import <none>
// CHECK:STDOUT:   %default.import.loc2_17.2 = import <none>
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ForwardDecl.decl: type = class_decl @ForwardDecl [concrete = constants.%ForwardDecl] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ForwardDecl.as.Destroy.impl: @ForwardDecl.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %ForwardDecl.as.Destroy.impl.Op.decl: %ForwardDecl.as.Destroy.impl.Op.type = fn_decl @ForwardDecl.as.Destroy.impl.Op [concrete = constants.%ForwardDecl.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fd7 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fd7 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.83c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%ForwardDecl [concrete = constants.%ForwardDecl]
// CHECK:STDOUT:     %self: %ptr.83c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %ForwardDecl.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @ForwardDecl.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDecl {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%ForwardDecl [concrete = constants.%ForwardDecl]
// CHECK:STDOUT:   impl_decl @ForwardDecl.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@ForwardDecl.as.Destroy.impl.%ForwardDecl.as.Destroy.impl.Op.decl), @ForwardDecl.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%ForwardDecl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForwardDecl.as.Destroy.impl.Op(%self.param: %ptr.83c) = "no_op";
// CHECK:STDOUT:
