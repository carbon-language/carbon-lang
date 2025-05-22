// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/redeclaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/redeclaration.carbon

// --- redeclaration.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class;

class Class {
  fn F[self: Self](b: ());
}

fn Class.F[self: Self](b: ()) {}
//@dump-sem-ir-end

// --- redeclaration_introducer.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class A;
class B;
class C;

class A {}
base class B {}
abstract class C {}
//@dump-sem-ir-end

// --- fail_redeclaration_scope.carbon
library "[[@TEST_NAME]]";

class A;

class X {
  // OK, a different A.
  class A { class B; }
  class A.B {}
}

class A { class B; }

class Y {
  // CHECK:STDERR: fail_redeclaration_scope.carbon:[[@LINE+4]]:9: error: name `A` not found [NameNotFound]
  // CHECK:STDERR:   class A.B {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  class A.B {}
}

// --- fail_redefinition.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F();
  fn H();
  fn I() {}
}

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:1: error: redefinition of `class Class` [RedeclRedef]
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-9]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class Class {
  fn G();
  fn H();
  fn I() {}
}

fn Class.F() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+4]]:10: error: out-of-line declaration requires a declaration in scoped entity [QualifiedDeclOutsideScopeEntity]
// CHECK:STDERR: fn Class.G() {}
// CHECK:STDERR:          ^
// CHECK:STDERR:
fn Class.G() {}
fn Class.H() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:1: error: redefinition of `fn I` [RedeclRedef]
// CHECK:STDERR: fn Class.I() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-26]]:3: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR:   fn I() {}
// CHECK:STDERR:   ^~~~~~~~
// CHECK:STDERR:
fn Class.I() {}

// CHECK:STDOUT: --- redeclaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl.loc4: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.decl.loc6: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.cb1 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc10: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc10: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc10: %Class = bind_name self, %self.param.loc10
// CHECK:STDOUT:     %b.param.loc10: %empty_tuple.type = value_param call_param1
// CHECK:STDOUT:     %.loc10_28.1: type = splice_block %.loc10_28.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc10_28.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc10_28.3: type = converted %.loc10_28.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc10: %empty_tuple.type = bind_name b, %b.param.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.761 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.cb1 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc7: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc7: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc7: %Class = bind_name self, %self.param.loc7
// CHECK:STDOUT:     %b.param.loc7: %empty_tuple.type = value_param call_param1
// CHECK:STDOUT:     %.loc7_24.1: type = splice_block %.loc7_24.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc7_24.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc7_24.3: type = converted %.loc7_24.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.loc7: %empty_tuple.type = bind_name b, %b.param.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self.param.loc10: %Class, %b.param.loc10: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redeclaration_introducer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl.loc4: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc5: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl.loc6: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %A.decl.loc8: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc9: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl.loc10: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
