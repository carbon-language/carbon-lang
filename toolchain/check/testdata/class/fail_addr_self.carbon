// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_addr_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_addr_self.carbon

class Class {
  fn F[addr self: Class*]();
  fn G[addr self: Class]();
}

fn F(c: Class, p: Class*) {
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+7]]:6: error: `addr self` method cannot be invoked on a value
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:      ^
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-8]]:13: note: initializing `addr self` parameter of method declared here
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  c.F();

  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+7]]:6: error: `addr self` method cannot be invoked on a value
  // CHECK:STDERR:   c.G();
  // CHECK:STDERR:      ^
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-16]]:13: note: initializing `addr self` parameter of method declared here
  // CHECK:STDERR:   fn G[addr self: Class]();
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  c.G();

  // This call is OK.
  (*p).F();

  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+9]]:3: error: cannot implicitly convert from `Class*` to `Class`
  // CHECK:STDERR:   (*p).G();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE+6]]:3: note: type `Class*` does not implement interface `ImplicitAs`
  // CHECK:STDERR:   (*p).G();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_addr_self.carbon:[[@LINE-31]]:13: note: initializing `addr self` parameter of method declared here
  // CHECK:STDERR:   fn G[addr self: Class]();
  // CHECK:STDERR:             ^~~~
  (*p).G();
}

// CHECK:STDOUT: --- fail_addr_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type %Class [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, imports.%import_ref.5 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = interface_type @ImplicitAs, @ImplicitAs(%Class) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert, @ImplicitAs(%Class) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, imports.%import_ref.5 [template]
// CHECK:STDOUT:   %.10: %.6 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %ImplicitAs.type.1 = import_ref Core//prelude/operators/as, inst+40, loaded [template = constants.%ImplicitAs]
// CHECK:STDOUT:   %import_ref.2 = import_ref Core//prelude/operators/as, inst+45, unloaded
// CHECK:STDOUT:   %import_ref.3: @ImplicitAs.%.1 (%.6) = import_ref Core//prelude/operators/as, inst+63, loaded [symbolic = @ImplicitAs.%.2 (constants.%.10)]
// CHECK:STDOUT:   %import_ref.4 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//prelude/operators/as, inst+56, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %c.patt: %Class = binding_pattern c
// CHECK:STDOUT:     %p.patt: %.1 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc16_9: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.param: %Class = param c, runtime_param0
// CHECK:STDOUT:     %c: %Class = bind_name c, %c.param
// CHECK:STDOUT:     %Class.ref.loc16_19: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc16: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %p.param: %.1 = param p, runtime_param1
// CHECK:STDOUT:     %p: %.1 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%Dest: type) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.6)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.6) = assoc_entity element0, imports.%import_ref.5 [symbolic = %.2 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.2
// CHECK:STDOUT:     .Convert = imports.%import_ref.3
// CHECK:STDOUT:     witness = (imports.%import_ref.4)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: %.1 = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc12_24: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %self.param: %.1 = param self, runtime_param0
// CHECK:STDOUT:     %self: %.1 = bind_name self, %self.param
// CHECK:STDOUT:     %.loc12_8: %.1 = addr_pattern %self
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt: %Class = binding_pattern self
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.param: %Class = param self, runtime_param0
// CHECK:STDOUT:     %self: %Class = bind_name self, %self.param
// CHECK:STDOUT:     %.loc13: %Class = addr_pattern %self
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self: %.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr %self: %Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%c: %Class, %p: %.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref.loc24: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref.loc24: %F.type.1 = name_ref F, @Class.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc24: <bound method> = bound_method %c.ref.loc24, %F.ref.loc24
// CHECK:STDOUT:   %F.call.loc24: init %.2 = call %.loc24(<invalid>) [template = <error>]
// CHECK:STDOUT:   %c.ref.loc33: %Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref.loc33: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc33: <bound method> = bound_method %c.ref.loc33, %G.ref.loc33
// CHECK:STDOUT:   %G.call.loc33: init %.2 = call %.loc33(<invalid>) [template = <error>]
// CHECK:STDOUT:   %p.ref.loc36: %.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc36_4.1: ref %Class = deref %p.ref.loc36
// CHECK:STDOUT:   %F.ref.loc36: %F.type.1 = name_ref F, @Class.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc36_7: <bound method> = bound_method %.loc36_4.1, %F.ref.loc36
// CHECK:STDOUT:   %.loc36_4.2: %.1 = addr_of %.loc36_4.1
// CHECK:STDOUT:   %F.call.loc36: init %.2 = call %.loc36_7(%.loc36_4.2)
// CHECK:STDOUT:   %p.ref.loc47: %.1 = name_ref p, %p
// CHECK:STDOUT:   %.loc47_4.1: ref %Class = deref %p.ref.loc47
// CHECK:STDOUT:   %G.ref.loc47: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc47_7: <bound method> = bound_method %.loc47_4.1, %G.ref.loc47
// CHECK:STDOUT:   %.loc47_4.2: %.1 = addr_of %.loc47_4.1
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(constants.%Class) [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %.loc47_9.1: %.8 = specific_constant imports.%import_ref.3, @ImplicitAs(constants.%Class) [template = constants.%.9]
// CHECK:STDOUT:   %Convert.ref: %.8 = name_ref Convert, %.loc47_9.1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc47_9.2: %Class = converted %.loc47_4.2, <error> [template = <error>]
// CHECK:STDOUT:   %G.call.loc47: init %.2 = call %.loc47_7(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert(constants.%Dest: type, constants.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic = %Dest (constants.%Dest)]
// CHECK:STDOUT:   %ImplicitAs.type: type = interface_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @Convert.%Self (%Self.2)]() -> @Convert.%Dest (%Dest);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@ImplicitAs.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.%Dest) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert(constants.%Dest, constants.%Self.1) {
// CHECK:STDOUT:   %Dest => constants.%Dest
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%Class) {
// CHECK:STDOUT:   %Dest => constants.%Class
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.8
// CHECK:STDOUT:   %.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
