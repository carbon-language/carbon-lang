// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=ignore
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/field.carbon

// --- struct.h

struct Struct {
  int a;
  int b;
  int* _Nonnull p;
};

// --- use_struct_fields.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct.h";

fn F(s: Cpp.Struct) -> (i32, i32, i32) {
  //@dump-sem-ir-begin
  return (s.a, s.b, *s.p);
  //@dump-sem-ir-end
}

// --- union.h

union Union {
  int a;
  int b;
  int* _Nonnull p;
};

// --- use_union_fields.carbon

library "[[@TEST_NAME]]";

import Cpp library "union.h";

fn F(u: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  return u.a;
  //@dump-sem-ir-end
}

fn G(u: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  return u.b;
  //@dump-sem-ir-end
}

fn H(u: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  return *u.p;
  //@dump-sem-ir-end
}

// --- anon_struct_union.h

struct A {
  union {
    struct {
      short a_0;
      short b_2;
      int c_4;
    };
    struct {
      char d_0;
      char e_1;
      int f_4;
      short g_8;
    };
  };
  int h_12;
};

// --- use_anon_struct_union.carbon

library "[[@TEST_NAME]]";

import Cpp library "anon_struct_union.h";

fn GetF(a: Cpp.A) -> i32 { return a.f_4; }

// --- with_bitfields.h

struct Struct {
  int a;
  int b : 2;
};

union Union {
  int c;
  int d : 3;
};

// --- use_non_bitfields_in_type_with_bitfields.carbon

library "[[@TEST_NAME]]";

import Cpp library "with_bitfields.h";

fn F(s: Cpp.Struct) -> i32 {
  //@dump-sem-ir-begin
  return s.a;
  //@dump-sem-ir-end
}

fn G(s: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  return s.c;
  //@dump-sem-ir-end
}

// --- fail_todo_use_bitfields.carbon

library "[[@TEST_NAME]]";

import Cpp library "with_bitfields.h";

fn F(s: Cpp.Struct) -> i32 {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE+11]]:10: error: semantics TODO: `Unsupported: Unhandled kind of field declaration` [SemanticsTodo]
  // CHECK:STDERR:   return s.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE+8]]:10: note: in `Cpp` name lookup for `b` [InCppNameLookup]
  // CHECK:STDERR:   return s.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE+4]]:10: error: member name `b` not found in `Cpp.Struct` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return s.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return s.b;
  //@dump-sem-ir-end
}

fn G(s: Cpp.Union) -> i32 {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE+11]]:10: error: semantics TODO: `Unsupported: Unhandled kind of field declaration` [SemanticsTodo]
  // CHECK:STDERR:   return s.d;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE+8]]:10: note: in `Cpp` name lookup for `d` [InCppNameLookup]
  // CHECK:STDERR:   return s.d;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use_bitfields.carbon:[[@LINE+4]]:10: error: member name `d` not found in `Cpp.Union` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return s.d;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return s.d;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- use_struct_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Struct: type = class_type @Struct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Struct.elem.86b: type = unbound_element_type %Struct, %i32 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Struct.elem.765: type = unbound_element_type %Struct, %ptr.235 [concrete]
// CHECK:STDOUT:   %.550: type = custom_layout_type {size=16, align=8, .a@0: %i32, .b@4: %i32, .p@8: %ptr.235} [concrete]
// CHECK:STDOUT:   %complete_type.8cf: <witness> = complete_type_witness %.550 [concrete]
// CHECK:STDOUT:   %pattern_type.60c: type = pattern_type %Struct [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.189: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.b5a: type = pattern_type %tuple.type.189 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Struct = %Struct.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Struct.decl: type = class_decl @Struct [concrete = constants.%Struct] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "struct.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %s.patt: %pattern_type.60c = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.60c = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.b5a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b5a = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_38.1: %tuple.type.ff9 = tuple_literal (%i32.loc6_25, %i32.loc6_30, %i32.loc6_35)
// CHECK:STDOUT:     %.loc6_38.2: type = converted %.loc6_38.1, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %s.param: %Struct = value_param call_param0
// CHECK:STDOUT:     %.loc6_12: type = splice_block %Struct.ref [concrete = constants.%Struct] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Struct.ref: type = name_ref Struct, imports.%Struct.decl [concrete = constants.%Struct]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %Struct = bind_name s, %s.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.189 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.189 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Struct {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %Struct.elem.86b = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.2: %Struct.elem.86b = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.3: %Struct.elem.765 = field_decl p, element2 [concrete]
// CHECK:STDOUT:   %.4: type = custom_layout_type {size=16, align=8, .a@0: %i32, .b@4: %i32, .p@8: %ptr.235} [concrete = constants.%.550]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.4 [concrete = constants.%complete_type.8cf]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Struct
// CHECK:STDOUT:   .a = %.1
// CHECK:STDOUT:   .b = %.2
// CHECK:STDOUT:   .p = %.3
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %Struct) -> %return.param: %tuple.type.189 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref.loc8_11: %Struct = name_ref s, %s
// CHECK:STDOUT:   %a.ref: %Struct.elem.86b = name_ref a, @Struct.%.1 [concrete = @Struct.%.1]
// CHECK:STDOUT:   %.loc8_12.1: ref %i32 = class_element_access %s.ref.loc8_11, element0
// CHECK:STDOUT:   %.loc8_12.2: %i32 = bind_value %.loc8_12.1
// CHECK:STDOUT:   %s.ref.loc8_16: %Struct = name_ref s, %s
// CHECK:STDOUT:   %b.ref: %Struct.elem.86b = name_ref b, @Struct.%.2 [concrete = @Struct.%.2]
// CHECK:STDOUT:   %.loc8_17.1: ref %i32 = class_element_access %s.ref.loc8_16, element1
// CHECK:STDOUT:   %.loc8_17.2: %i32 = bind_value %.loc8_17.1
// CHECK:STDOUT:   %s.ref.loc8_22: %Struct = name_ref s, %s
// CHECK:STDOUT:   %p.ref: %Struct.elem.765 = name_ref p, @Struct.%.3 [concrete = @Struct.%.3]
// CHECK:STDOUT:   %.loc8_23.1: ref %ptr.235 = class_element_access %s.ref.loc8_22, element2
// CHECK:STDOUT:   %.loc8_23.2: %ptr.235 = bind_value %.loc8_23.1
// CHECK:STDOUT:   %.loc8_21.1: ref %i32 = deref %.loc8_23.2
// CHECK:STDOUT:   %.loc8_25.1: %tuple.type.189 = tuple_literal (%.loc8_12.2, %.loc8_17.2, %.loc8_21.1)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc8_25.2: init %i32 = initialize_from %.loc8_12.2 to %tuple.elem0
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc8_25.3: init %i32 = initialize_from %.loc8_17.2 to %tuple.elem1
// CHECK:STDOUT:   %.loc8_21.2: %i32 = bind_value %.loc8_21.1
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %return, element2
// CHECK:STDOUT:   %.loc8_25.4: init %i32 = initialize_from %.loc8_21.2 to %tuple.elem2
// CHECK:STDOUT:   %.loc8_25.5: init %tuple.type.189 = tuple_init (%.loc8_25.2, %.loc8_25.3, %.loc8_25.4) to %return
// CHECK:STDOUT:   %.loc8_26: init %tuple.type.189 = converted %.loc8_25.1, %.loc8_25.5
// CHECK:STDOUT:   return %.loc8_26 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_union_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Union: type = class_type @Union [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Union.elem.041: type = unbound_element_type %Union, %i32 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Union.elem.92a: type = unbound_element_type %Union, %ptr.235 [concrete]
// CHECK:STDOUT:   %.c84: type = custom_layout_type {size=8, align=8, .a@0: %i32, .b@0: %i32, .p@0: %ptr.235} [concrete]
// CHECK:STDOUT:   %complete_type.1b0: <witness> = complete_type_witness %.c84 [concrete]
// CHECK:STDOUT:   %pattern_type.a90: type = pattern_type %Union [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Union = %Union.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Union.decl: type = class_decl @Union [concrete = constants.%Union] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "union.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %u.patt: %pattern_type.a90 = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.a90 = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %u.param: %Union = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %Union.ref [concrete = constants.%Union] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Union.ref: type = name_ref Union, imports.%Union.decl [concrete = constants.%Union]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %Union = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %u.patt: %pattern_type.a90 = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.a90 = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %u.param: %Union = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %Union.ref [concrete = constants.%Union] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Union.ref: type = name_ref Union, imports.%Union.decl [concrete = constants.%Union]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %Union = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %u.patt: %pattern_type.a90 = binding_pattern u [concrete]
// CHECK:STDOUT:     %u.param_patt: %pattern_type.a90 = value_param_pattern %u.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %u.param: %Union = value_param call_param0
// CHECK:STDOUT:     %.loc18: type = splice_block %Union.ref [concrete = constants.%Union] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Union.ref: type = name_ref Union, imports.%Union.decl [concrete = constants.%Union]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %Union = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Union {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %Union.elem.041 = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.2: %Union.elem.041 = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.3: %Union.elem.92a = field_decl p, element2 [concrete]
// CHECK:STDOUT:   %.4: type = custom_layout_type {size=8, align=8, .a@0: %i32, .b@0: %i32, .p@0: %ptr.235} [concrete = constants.%.c84]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.4 [concrete = constants.%complete_type.1b0]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Union
// CHECK:STDOUT:   .a = %.1
// CHECK:STDOUT:   .b = %.2
// CHECK:STDOUT:   .p = %.3
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%u.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %Union = name_ref u, %u
// CHECK:STDOUT:   %a.ref: %Union.elem.041 = name_ref a, @Union.%.1 [concrete = @Union.%.1]
// CHECK:STDOUT:   %.loc8_11.1: ref %i32 = class_element_access %u.ref, element0
// CHECK:STDOUT:   %.loc8_11.2: %i32 = bind_value %.loc8_11.1
// CHECK:STDOUT:   return %.loc8_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%u.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %Union = name_ref u, %u
// CHECK:STDOUT:   %b.ref: %Union.elem.041 = name_ref b, @Union.%.2 [concrete = @Union.%.2]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %u.ref, element1
// CHECK:STDOUT:   %.loc14_11.2: %i32 = bind_value %.loc14_11.1
// CHECK:STDOUT:   return %.loc14_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%u.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %Union = name_ref u, %u
// CHECK:STDOUT:   %p.ref: %Union.elem.92a = name_ref p, @Union.%.3 [concrete = @Union.%.3]
// CHECK:STDOUT:   %.loc20_12.1: ref %ptr.235 = class_element_access %u.ref, element2
// CHECK:STDOUT:   %.loc20_12.2: %ptr.235 = bind_value %.loc20_12.1
// CHECK:STDOUT:   %.loc20_10.1: ref %i32 = deref %.loc20_12.2
// CHECK:STDOUT:   %.loc20_10.2: %i32 = bind_value %.loc20_10.1
// CHECK:STDOUT:   return %.loc20_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_anon_struct_union.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %A.elem.543: type = unbound_element_type %A, %i16 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem.c3f: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %.371: type = custom_layout_type {size=16, align=4, .a_0@0: %i16, .b_2@2: %i16, .c_4@4: %i32, .f_4@4: %i32, .g_8@8: %i16, .h_12@12: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fa3: <witness> = complete_type_witness %.371 [concrete]
// CHECK:STDOUT:   %pattern_type.9de: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %GetF.type: type = fn_type @GetF [concrete]
// CHECK:STDOUT:   %GetF: %GetF.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .GetF = %GetF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "anon_struct_union.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GetF.decl: %GetF.type = fn_decl @GetF [concrete = constants.%GetF] {
// CHECK:STDOUT:     %a.patt: %pattern_type.9de = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.9de = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %.loc6_15: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %A.ref: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_16.1: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16.1: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %.1: %A.elem.543 = field_decl a_0, element0 [concrete]
// CHECK:STDOUT:   %int_16.2: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16.2: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %.2: %A.elem.543 = field_decl b_2, element1 [concrete]
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.3: %A.elem.c3f = field_decl c_4, element2 [concrete]
// CHECK:STDOUT:   %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.4: %A.elem.c3f = field_decl f_4, element3 [concrete]
// CHECK:STDOUT:   %int_16.3: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16.3: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %.5: %A.elem.543 = field_decl g_8, element4 [concrete]
// CHECK:STDOUT:   %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.6: %A.elem.c3f = field_decl h_12, element5 [concrete]
// CHECK:STDOUT:   %.7: type = custom_layout_type {size=16, align=4, .a_0@0: %i16, .b_2@2: %i16, .c_4@4: %i32, .f_4@4: %i32, .g_8@8: %i16, .h_12@12: %i32} [concrete = constants.%.371]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.7 [concrete = constants.%complete_type.fa3]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .f_4 = %.4
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetF(%a.param: %A) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a
// CHECK:STDOUT:   %f_4.ref: %A.elem.c3f = name_ref f_4, @A.%.4 [concrete = @A.%.4]
// CHECK:STDOUT:   %.loc6_36.1: ref %i32 = class_element_access %a.ref, element3
// CHECK:STDOUT:   %.loc6_36.2: %i32 = bind_value %.loc6_36.1
// CHECK:STDOUT:   return %.loc6_36.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non_bitfields_in_type_with_bitfields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Struct: type = class_type @Struct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Struct.elem: type = unbound_element_type %Struct, %i32 [concrete]
// CHECK:STDOUT:   %.f9e: type = custom_layout_type {size=8, align=4, .a@0: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.d3c: <witness> = complete_type_witness %.f9e [concrete]
// CHECK:STDOUT:   %pattern_type.60c: type = pattern_type %Struct [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Union: type = class_type @Union [concrete]
// CHECK:STDOUT:   %Union.elem: type = unbound_element_type %Union, %i32 [concrete]
// CHECK:STDOUT:   %.891: type = custom_layout_type {size=4, align=4, .c@0: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.591: <witness> = complete_type_witness %.891 [concrete]
// CHECK:STDOUT:   %pattern_type.a90: type = pattern_type %Union [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Struct = %Struct.decl
// CHECK:STDOUT:     .Union = %Union.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Struct.decl: type = class_decl @Struct [concrete = constants.%Struct] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Union.decl: type = class_decl @Union [concrete = constants.%Union] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "with_bitfields.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %s.patt: %pattern_type.60c = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.60c = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %s.param: %Struct = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %Struct.ref [concrete = constants.%Struct] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Struct.ref: type = name_ref Struct, imports.%Struct.decl [concrete = constants.%Struct]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %Struct = bind_name s, %s.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %s.patt: %pattern_type.a90 = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.a90 = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %s.param: %Union = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %Union.ref [concrete = constants.%Union] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Union.ref: type = name_ref Union, imports.%Union.decl [concrete = constants.%Union]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %Union = bind_name s, %s.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Struct {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %Struct.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=8, align=4, .a@0: %i32} [concrete = constants.%.f9e]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.d3c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Struct
// CHECK:STDOUT:   .a = %.1
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Union {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %Union.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=4, align=4, .c@0: %i32} [concrete = constants.%.891]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.591]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Union
// CHECK:STDOUT:   .c = %.1
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %Struct) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Struct = name_ref s, %s
// CHECK:STDOUT:   %a.ref: %Struct.elem = name_ref a, @Struct.%.1 [concrete = @Struct.%.1]
// CHECK:STDOUT:   %.loc8_11.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc8_11.2: %i32 = bind_value %.loc8_11.1
// CHECK:STDOUT:   return %.loc8_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Union = name_ref s, %s
// CHECK:STDOUT:   %c.ref: %Union.elem = name_ref c, @Union.%.1 [concrete = @Union.%.1]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = bind_value %.loc14_11.1
// CHECK:STDOUT:   return %.loc14_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_bitfields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Struct: type = class_type @Struct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Struct.elem: type = unbound_element_type %Struct, %i32 [concrete]
// CHECK:STDOUT:   %.f9e: type = custom_layout_type {size=8, align=4, .a@0: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.d3c: <witness> = complete_type_witness %.f9e [concrete]
// CHECK:STDOUT:   %pattern_type.60c: type = pattern_type %Struct [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Union: type = class_type @Union [concrete]
// CHECK:STDOUT:   %Union.elem: type = unbound_element_type %Union, %i32 [concrete]
// CHECK:STDOUT:   %.891: type = custom_layout_type {size=4, align=4, .c@0: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.591: <witness> = complete_type_witness %.891 [concrete]
// CHECK:STDOUT:   %pattern_type.a90: type = pattern_type %Union [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Struct = %Struct.decl
// CHECK:STDOUT:     .Union = %Union.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Struct.decl: type = class_decl @Struct [concrete = constants.%Struct] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Union.decl: type = class_decl @Union [concrete = constants.%Union] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "with_bitfields.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %s.patt: %pattern_type.60c = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.60c = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %s.param: %Struct = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %Struct.ref [concrete = constants.%Struct] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Struct.ref: type = name_ref Struct, imports.%Struct.decl [concrete = constants.%Struct]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %Struct = bind_name s, %s.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %s.patt: %pattern_type.a90 = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.a90 = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %s.param: %Union = value_param call_param0
// CHECK:STDOUT:     %.loc23: type = splice_block %Union.ref [concrete = constants.%Union] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %Union.ref: type = name_ref Union, imports.%Union.decl [concrete = constants.%Union]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: %Union = bind_name s, %s.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Struct {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %Struct.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=8, align=4, .a@0: %i32} [concrete = constants.%.f9e]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.d3c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Struct
// CHECK:STDOUT:   .b = <poisoned>
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Union {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %Union.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=4, align=4, .c@0: %i32} [concrete = constants.%.891]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.591]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Union
// CHECK:STDOUT:   .d = <poisoned>
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %Struct) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Struct = name_ref s, %s
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%s.param: %Union) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %Union = name_ref s, %s
// CHECK:STDOUT:   %d.ref: <error> = name_ref d, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
