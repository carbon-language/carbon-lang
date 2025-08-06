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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_addr_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_addr_self.carbon

class Class {
  fn F[addr self: Class*]();
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+4]]:8: error: `addr` can only be applied to a binding with a pointer type [AddrOnNonPointerType]
  // CHECK:STDERR:   fn G[addr self: Class]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn G[addr self: Class]();
}

fn F(c: Class, p: Class*) {
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+7]]:3: error: `addr self` method cannot be invoked on a value [AddrSelfIsNonRef]
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-12]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c.F();

  c.G();

  // This call is OK.
  (*p).F();

  (*p).G();
}

// CHECK:STDOUT: --- fail_addr_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.796: type = pattern_type %ptr.e71 [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G [concrete]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b40: <witness> = impl_witness @Class.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.761 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.796 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.796 = value_param_pattern %p.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref.loc24_9: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %p.param: %ptr.e71 = value_param call_param1
// CHECK:STDOUT:     %.loc24: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc24_19: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc24_19 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e71 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Class.as.Destroy.impl: constants.%Class as constants.%Destroy.type {
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.decl: %Class.as.Destroy.impl.Op.type = fn_decl @Class.as.Destroy.impl.Op [concrete = constants.%Class.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.796 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.796 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %ptr.e71 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [concrete = constants.%Class.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.796 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.796 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc16_24: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.e71 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [concrete = constants.%Class.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Class = value_param call_param0
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %Class = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.b40]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT:   .F = %Class.F.decl
// CHECK:STDOUT:   .G = %Class.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.F(%self.param: %ptr.e71);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.G(%self.param: %Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.as.Destroy.impl.Op(%self.param: %ptr.e71) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %Class, %p.param: %ptr.e71) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc32: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc32: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound.loc32: <bound method> = bound_method %c.ref.loc32, %F.ref.loc32
// CHECK:STDOUT:   %.loc32: ref %Class = temporary_storage
// CHECK:STDOUT:   %addr.loc32_3.1: %ptr.e71 = addr_of %.loc32
// CHECK:STDOUT:   %Class.F.call.loc32: init %empty_tuple.type = call %Class.F.bound.loc32(%addr.loc32_3.1)
// CHECK:STDOUT:   %c.ref.loc34: %Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc34: %Class.G.type = name_ref G, @Class.%Class.G.decl [concrete = constants.%Class.G]
// CHECK:STDOUT:   %Class.G.bound.loc34: <bound method> = bound_method %c.ref.loc34, %G.ref.loc34
// CHECK:STDOUT:   %Class.G.call.loc34: init %empty_tuple.type = call %Class.G.bound.loc34(%c.ref.loc34)
// CHECK:STDOUT:   %p.ref.loc37: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc37: ref %Class = deref %p.ref.loc37
// CHECK:STDOUT:   %F.ref.loc37: %Class.F.type = name_ref F, @Class.%Class.F.decl [concrete = constants.%Class.F]
// CHECK:STDOUT:   %Class.F.bound.loc37: <bound method> = bound_method %.loc37, %F.ref.loc37
// CHECK:STDOUT:   %addr.loc37: %ptr.e71 = addr_of %.loc37
// CHECK:STDOUT:   %Class.F.call.loc37: init %empty_tuple.type = call %Class.F.bound.loc37(%addr.loc37)
// CHECK:STDOUT:   %p.ref.loc39: %ptr.e71 = name_ref p, %p
// CHECK:STDOUT:   %.loc39_4.1: ref %Class = deref %p.ref.loc39
// CHECK:STDOUT:   %G.ref.loc39: %Class.G.type = name_ref G, @Class.%Class.G.decl [concrete = constants.%Class.G]
// CHECK:STDOUT:   %Class.G.bound.loc39: <bound method> = bound_method %.loc39_4.1, %G.ref.loc39
// CHECK:STDOUT:   %.loc39_4.2: %Class = bind_value %.loc39_4.1
// CHECK:STDOUT:   %Class.G.call.loc39: init %empty_tuple.type = call %Class.G.bound.loc39(%.loc39_4.2)
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc32, constants.%Class.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc32_3.2: %ptr.e71 = addr_of %.loc32
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Class.as.Destroy.impl.Op.bound(%addr.loc32_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
