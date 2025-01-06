// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/adapt_copy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/adapt_copy.carbon

// --- adapt_copyable.carbon

library "[[@TEST_NAME]]";

class AdaptCopyable {
  adapt i32;
}

// We can copy an adapter by copying its value if its initializing
// representation is a copy of its value representation.
fn F(c: AdaptCopyable) -> AdaptCopyable {
  var d: AdaptCopyable = c;
  return d;
}

fn InTuple(c: (AdaptCopyable, u32)) -> (AdaptCopyable, u32) {
  var d: (AdaptCopyable, u32) = c;
  return d;
}

// --- adapt_copyable_tuple.carbon

library "[[@TEST_NAME]]";

class AdaptTuple {
  adapt (i32, i32);
}

fn F(c: AdaptTuple) -> AdaptTuple {
  var d: AdaptTuple = c;
  return d;
}

fn InTuple(c: (AdaptTuple, u32)) -> (AdaptTuple, u32) {
  var d: (AdaptTuple, u32) = c;
  return d;
}

// --- fail_adapt_not_copyable.carbon

library "[[@TEST_NAME]]";

class Noncopyable {
  // TODO: Ensure this remains non-copyable once we have rules for class copyability.
}

class AdaptNoncopyable {
  adapt Noncopyable;
}

fn G(a: AdaptNoncopyable) -> AdaptNoncopyable {
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+4]]:29: error: cannot copy value of type `AdaptNoncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var b: AdaptNoncopyable = a;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR:
  var b: AdaptNoncopyable = a;
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+4]]:10: error: cannot copy value of type `AdaptNoncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return b;
}

// --- fail_adapt_not_copyable_indirect.carbon

library "[[@TEST_NAME]]";

class Noncopyable {
  // TODO: Ensure this remains non-copyable once we have rules for class copyability.
}

class AdaptNoncopyableIndirect {
  adapt (i32, Noncopyable, i32);
}

fn H(a: AdaptNoncopyableIndirect) -> AdaptNoncopyableIndirect {
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+7]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+4]]:37: note: in copy of `AdaptNoncopyableIndirect` [InCopy]
  // CHECK:STDERR:   var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR:                                     ^
  // CHECK:STDERR:
  var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+6]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+3]]:10: note: in copy of `AdaptNoncopyableIndirect` [InCopy]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:          ^
  return b;
}

// --- adapt_copyable_struct.carbon

library "[[@TEST_NAME]]";

class AdaptStruct {
  adapt {.e: i32, .f: i32};
}

fn I(g: AdaptStruct) -> AdaptStruct {
  var h: AdaptStruct = g;
  return h;
}

fn InTuple(c: (AdaptStruct, u32)) -> (AdaptStruct, u32) {
  var d: (AdaptStruct, u32) = c;
  return d;
}

// CHECK:STDOUT: --- adapt_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptCopyable: type = class_type @AdaptCopyable [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%AdaptCopyable, %u32) [template]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [template]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .UInt = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptCopyable = %AdaptCopyable.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .InTuple = %InTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptCopyable.decl: type = class_decl @AdaptCopyable [template = constants.%AdaptCopyable] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %c.patt: %AdaptCopyable = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %AdaptCopyable = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %AdaptCopyable = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %AdaptCopyable = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc10_27: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [template = constants.%AdaptCopyable]
// CHECK:STDOUT:     %c.param: %AdaptCopyable = value_param runtime_param0
// CHECK:STDOUT:     %AdaptCopyable.ref.loc10_9: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [template = constants.%AdaptCopyable]
// CHECK:STDOUT:     %c: %AdaptCopyable = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptCopyable = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %AdaptCopyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [template = constants.%InTuple] {
// CHECK:STDOUT:     %c.patt: %tuple.type.2 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %tuple.type.2 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc15_41: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [template = constants.%AdaptCopyable]
// CHECK:STDOUT:     %int_32.loc15_56: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc15_56: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %.loc15_59.1: %tuple.type.1 = tuple_literal (%AdaptCopyable.ref.loc15_41, %u32.loc15_56)
// CHECK:STDOUT:     %.loc15_59.2: type = converted %.loc15_59.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %c.param: %tuple.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %.loc15_34.1: type = splice_block %.loc15_34.3 [template = constants.%tuple.type.2] {
// CHECK:STDOUT:       %AdaptCopyable.ref.loc15_16: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [template = constants.%AdaptCopyable]
// CHECK:STDOUT:       %int_32.loc15_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc15_31: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:       %.loc15_34.2: %tuple.type.1 = tuple_literal (%AdaptCopyable.ref.loc15_16, %u32.loc15_31)
// CHECK:STDOUT:       %.loc15_34.3: type = converted %.loc15_34.2, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.2 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptCopyable {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %i32.builtin [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptCopyable
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param_patt: %AdaptCopyable) -> %AdaptCopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %AdaptCopyable = var d
// CHECK:STDOUT:   %d: ref %AdaptCopyable = bind_name d, %d.var
// CHECK:STDOUT:   %c.ref: %AdaptCopyable = name_ref c, %c
// CHECK:STDOUT:   assign %d.var, %c.ref
// CHECK:STDOUT:   %d.ref: ref %AdaptCopyable = name_ref d, %d
// CHECK:STDOUT:   %.loc12: %AdaptCopyable = bind_value %d.ref
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param_patt: %tuple.type.2) -> %return.param_patt: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %tuple.type.2 = var d
// CHECK:STDOUT:   %d: ref %tuple.type.2 = bind_name d, %d.var
// CHECK:STDOUT:   %c.ref: %tuple.type.2 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc16_33.1: %AdaptCopyable = tuple_access %c.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc16_33.2: ref %AdaptCopyable = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc16_33.1: init %AdaptCopyable = initialize_from %tuple.elem0.loc16_33.1 to %tuple.elem0.loc16_33.2
// CHECK:STDOUT:   %tuple.elem1.loc16_33.1: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc16_33.2: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc16_33.2: init %u32 = initialize_from %tuple.elem1.loc16_33.1 to %tuple.elem1.loc16_33.2
// CHECK:STDOUT:   %.loc16_33.3: init %tuple.type.2 = tuple_init (%.loc16_33.1, %.loc16_33.2) to %d.var
// CHECK:STDOUT:   %.loc16_34: init %tuple.type.2 = converted %c.ref, %.loc16_33.3
// CHECK:STDOUT:   assign %d.var, %.loc16_34
// CHECK:STDOUT:   %d.ref: ref %tuple.type.2 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc17_10.1: ref %AdaptCopyable = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc17_10.1: %AdaptCopyable = bind_value %tuple.elem0.loc17_10.1
// CHECK:STDOUT:   %tuple.elem0.loc17_10.2: ref %AdaptCopyable = tuple_access %return, element0
// CHECK:STDOUT:   %.loc17_10.2: init %AdaptCopyable = initialize_from %.loc17_10.1 to %tuple.elem0.loc17_10.2
// CHECK:STDOUT:   %tuple.elem1.loc17_10.1: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc17_10.3: %u32 = bind_value %tuple.elem1.loc17_10.1
// CHECK:STDOUT:   %tuple.elem1.loc17_10.2: ref %u32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc17_10.4: init %u32 = initialize_from %.loc17_10.3 to %tuple.elem1.loc17_10.2
// CHECK:STDOUT:   %.loc17_10.5: init %tuple.type.2 = tuple_init (%.loc17_10.2, %.loc17_10.4) to %return
// CHECK:STDOUT:   %.loc17_11: init %tuple.type.2 = converted %d.ref, %.loc17_10.5
// CHECK:STDOUT:   return %.loc17_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptTuple: type = class_type @AdaptTuple [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %tuple.type.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%AdaptTuple, %u32) [template]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [template]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .UInt = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptTuple = %AdaptTuple.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .InTuple = %InTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptTuple.decl: type = class_decl @AdaptTuple [template = constants.%AdaptTuple] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %c.patt: %AdaptTuple = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %AdaptTuple = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %AdaptTuple = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %AdaptTuple = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc8_24: type = name_ref AdaptTuple, file.%AdaptTuple.decl [template = constants.%AdaptTuple]
// CHECK:STDOUT:     %c.param: %AdaptTuple = value_param runtime_param0
// CHECK:STDOUT:     %AdaptTuple.ref.loc8_9: type = name_ref AdaptTuple, file.%AdaptTuple.decl [template = constants.%AdaptTuple]
// CHECK:STDOUT:     %c: %AdaptTuple = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptTuple = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %AdaptTuple = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [template = constants.%InTuple] {
// CHECK:STDOUT:     %c.patt: %tuple.type.3 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %tuple.type.3 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %tuple.type.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.3 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc13_38: type = name_ref AdaptTuple, file.%AdaptTuple.decl [template = constants.%AdaptTuple]
// CHECK:STDOUT:     %int_32.loc13_50: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc13_50: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %.loc13_53.1: %tuple.type.1 = tuple_literal (%AdaptTuple.ref.loc13_38, %u32.loc13_50)
// CHECK:STDOUT:     %.loc13_53.2: type = converted %.loc13_53.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:     %c.param: %tuple.type.3 = value_param runtime_param0
// CHECK:STDOUT:     %.loc13_31.1: type = splice_block %.loc13_31.3 [template = constants.%tuple.type.3] {
// CHECK:STDOUT:       %AdaptTuple.ref.loc13_16: type = name_ref AdaptTuple, file.%AdaptTuple.decl [template = constants.%AdaptTuple]
// CHECK:STDOUT:       %int_32.loc13_28: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc13_28: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:       %.loc13_31.2: %tuple.type.1 = tuple_literal (%AdaptTuple.ref.loc13_16, %u32.loc13_28)
// CHECK:STDOUT:       %.loc13_31.3: type = converted %.loc13_31.2, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.3 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.3 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %tuple.type.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptTuple {
// CHECK:STDOUT:   %int_32.loc5_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc5_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_15: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: %tuple.type.1 = tuple_literal (%i32.loc5_10, %i32.loc5_15)
// CHECK:STDOUT:   %.loc5_19: type = converted %.loc5_18, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   adapt_decl %.loc5_19 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %tuple.type.2 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptTuple
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param_patt: %AdaptTuple) -> %return.param_patt: %AdaptTuple {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %AdaptTuple = var d
// CHECK:STDOUT:   %d: ref %AdaptTuple = bind_name d, %d.var
// CHECK:STDOUT:   %c.ref: %AdaptTuple = name_ref c, %c
// CHECK:STDOUT:   %.loc9_24.1: %tuple.type.2 = as_compatible %c.ref
// CHECK:STDOUT:   %tuple.elem0.loc9_24.1: %i32 = tuple_access %.loc9_24.1, element0
// CHECK:STDOUT:   %.loc9_24.2: ref %tuple.type.2 = as_compatible %d.var
// CHECK:STDOUT:   %tuple.elem0.loc9_24.2: ref %i32 = tuple_access %.loc9_24.2, element0
// CHECK:STDOUT:   %.loc9_24.3: init %i32 = initialize_from %tuple.elem0.loc9_24.1 to %tuple.elem0.loc9_24.2
// CHECK:STDOUT:   %tuple.elem1.loc9_24.1: %i32 = tuple_access %.loc9_24.1, element1
// CHECK:STDOUT:   %tuple.elem1.loc9_24.2: ref %i32 = tuple_access %.loc9_24.2, element1
// CHECK:STDOUT:   %.loc9_24.4: init %i32 = initialize_from %tuple.elem1.loc9_24.1 to %tuple.elem1.loc9_24.2
// CHECK:STDOUT:   %.loc9_24.5: init %tuple.type.2 = tuple_init (%.loc9_24.3, %.loc9_24.4) to %.loc9_24.2
// CHECK:STDOUT:   %.loc9_24.6: init %AdaptTuple = as_compatible %.loc9_24.5
// CHECK:STDOUT:   %.loc9_24.7: init %AdaptTuple = converted %c.ref, %.loc9_24.6
// CHECK:STDOUT:   assign %d.var, %.loc9_24.7
// CHECK:STDOUT:   %d.ref: ref %AdaptTuple = name_ref d, %d
// CHECK:STDOUT:   %.loc10_11.1: ref %tuple.type.2 = as_compatible %d.ref
// CHECK:STDOUT:   %tuple.elem0.loc10_11.1: ref %i32 = tuple_access %.loc10_11.1, element0
// CHECK:STDOUT:   %.loc10_11.2: %i32 = bind_value %tuple.elem0.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.3: ref %tuple.type.2 = as_compatible %return
// CHECK:STDOUT:   %tuple.elem0.loc10_11.2: ref %i32 = tuple_access %.loc10_11.3, element0
// CHECK:STDOUT:   %.loc10_11.4: init %i32 = initialize_from %.loc10_11.2 to %tuple.elem0.loc10_11.2
// CHECK:STDOUT:   %tuple.elem1.loc10_11.1: ref %i32 = tuple_access %.loc10_11.1, element1
// CHECK:STDOUT:   %.loc10_11.5: %i32 = bind_value %tuple.elem1.loc10_11.1
// CHECK:STDOUT:   %tuple.elem1.loc10_11.2: ref %i32 = tuple_access %.loc10_11.3, element1
// CHECK:STDOUT:   %.loc10_11.6: init %i32 = initialize_from %.loc10_11.5 to %tuple.elem1.loc10_11.2
// CHECK:STDOUT:   %.loc10_11.7: init %tuple.type.2 = tuple_init (%.loc10_11.4, %.loc10_11.6) to %.loc10_11.3
// CHECK:STDOUT:   %.loc10_11.8: init %AdaptTuple = as_compatible %.loc10_11.7
// CHECK:STDOUT:   %.loc10_11.9: init %AdaptTuple = converted %d.ref, %.loc10_11.8
// CHECK:STDOUT:   return %.loc10_11.9 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param_patt: %tuple.type.3) -> %return.param_patt: %tuple.type.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %tuple.type.3 = var d
// CHECK:STDOUT:   %d: ref %tuple.type.3 = bind_name d, %d.var
// CHECK:STDOUT:   %c.ref: %tuple.type.3 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc14_30.1: %AdaptTuple = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_30.1: %tuple.type.2 = as_compatible %tuple.elem0.loc14_30.1
// CHECK:STDOUT:   %tuple.elem0.loc14_30.2: %i32 = tuple_access %.loc14_30.1, element0
// CHECK:STDOUT:   %tuple.elem0.loc14_30.3: ref %AdaptTuple = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc14_30.2: ref %tuple.type.2 = as_compatible %tuple.elem0.loc14_30.3
// CHECK:STDOUT:   %tuple.elem0.loc14_30.4: ref %i32 = tuple_access %.loc14_30.2, element0
// CHECK:STDOUT:   %.loc14_30.3: init %i32 = initialize_from %tuple.elem0.loc14_30.2 to %tuple.elem0.loc14_30.4
// CHECK:STDOUT:   %tuple.elem1.loc14_30.1: %i32 = tuple_access %.loc14_30.1, element1
// CHECK:STDOUT:   %tuple.elem1.loc14_30.2: ref %i32 = tuple_access %.loc14_30.2, element1
// CHECK:STDOUT:   %.loc14_30.4: init %i32 = initialize_from %tuple.elem1.loc14_30.1 to %tuple.elem1.loc14_30.2
// CHECK:STDOUT:   %.loc14_30.5: init %tuple.type.2 = tuple_init (%.loc14_30.3, %.loc14_30.4) to %.loc14_30.2
// CHECK:STDOUT:   %.loc14_30.6: init %AdaptTuple = as_compatible %.loc14_30.5
// CHECK:STDOUT:   %.loc14_30.7: init %AdaptTuple = converted %tuple.elem0.loc14_30.1, %.loc14_30.6
// CHECK:STDOUT:   %tuple.elem1.loc14_30.3: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc14_30.4: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc14_30.8: init %u32 = initialize_from %tuple.elem1.loc14_30.3 to %tuple.elem1.loc14_30.4
// CHECK:STDOUT:   %.loc14_30.9: init %tuple.type.3 = tuple_init (%.loc14_30.7, %.loc14_30.8) to %d.var
// CHECK:STDOUT:   %.loc14_31: init %tuple.type.3 = converted %c.ref, %.loc14_30.9
// CHECK:STDOUT:   assign %d.var, %.loc14_31
// CHECK:STDOUT:   %d.ref: ref %tuple.type.3 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc15_10.1: ref %AdaptTuple = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc15_10.1: ref %tuple.type.2 = as_compatible %tuple.elem0.loc15_10.1
// CHECK:STDOUT:   %tuple.elem0.loc15_10.2: ref %i32 = tuple_access %.loc15_10.1, element0
// CHECK:STDOUT:   %.loc15_10.2: %i32 = bind_value %tuple.elem0.loc15_10.2
// CHECK:STDOUT:   %tuple.elem0.loc15_10.3: ref %AdaptTuple = tuple_access %return, element0
// CHECK:STDOUT:   %.loc15_10.3: ref %tuple.type.2 = as_compatible %tuple.elem0.loc15_10.3
// CHECK:STDOUT:   %tuple.elem0.loc15_10.4: ref %i32 = tuple_access %.loc15_10.3, element0
// CHECK:STDOUT:   %.loc15_10.4: init %i32 = initialize_from %.loc15_10.2 to %tuple.elem0.loc15_10.4
// CHECK:STDOUT:   %tuple.elem1.loc15_10.1: ref %i32 = tuple_access %.loc15_10.1, element1
// CHECK:STDOUT:   %.loc15_10.5: %i32 = bind_value %tuple.elem1.loc15_10.1
// CHECK:STDOUT:   %tuple.elem1.loc15_10.2: ref %i32 = tuple_access %.loc15_10.3, element1
// CHECK:STDOUT:   %.loc15_10.6: init %i32 = initialize_from %.loc15_10.5 to %tuple.elem1.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.7: init %tuple.type.2 = tuple_init (%.loc15_10.4, %.loc15_10.6) to %.loc15_10.3
// CHECK:STDOUT:   %.loc15_10.8: init %AdaptTuple = as_compatible %.loc15_10.7
// CHECK:STDOUT:   %.loc15_10.9: init %AdaptTuple = converted %tuple.elem0.loc15_10.1, %.loc15_10.8
// CHECK:STDOUT:   %tuple.elem1.loc15_10.3: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc15_10.10: %u32 = bind_value %tuple.elem1.loc15_10.3
// CHECK:STDOUT:   %tuple.elem1.loc15_10.4: ref %u32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc15_10.11: init %u32 = initialize_from %.loc15_10.10 to %tuple.elem1.loc15_10.4
// CHECK:STDOUT:   %.loc15_10.12: init %tuple.type.3 = tuple_init (%.loc15_10.9, %.loc15_10.11) to %return
// CHECK:STDOUT:   %.loc15_11: init %tuple.type.3 = converted %d.ref, %.loc15_10.12
// CHECK:STDOUT:   return %.loc15_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_not_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %AdaptNoncopyable: type = class_type @AdaptNoncopyable [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Noncopyable = %Noncopyable.decl
// CHECK:STDOUT:     .AdaptNoncopyable = %AdaptNoncopyable.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Noncopyable.decl: type = class_decl @Noncopyable [template = constants.%Noncopyable] {} {}
// CHECK:STDOUT:   %AdaptNoncopyable.decl: type = class_decl @AdaptNoncopyable [template = constants.%AdaptNoncopyable] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %a.patt: %AdaptNoncopyable = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %AdaptNoncopyable = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %AdaptNoncopyable = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %AdaptNoncopyable = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc12_30: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [template = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyable = value_param runtime_param0
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc12_9: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [template = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %a: %AdaptNoncopyable = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %AdaptNoncopyable = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %AdaptNoncopyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNoncopyable {
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [template = constants.%Noncopyable]
// CHECK:STDOUT:   adapt_decl %Noncopyable.ref [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNoncopyable
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param_patt: %AdaptNoncopyable) -> %return.param_patt: %AdaptNoncopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyable = var b
// CHECK:STDOUT:   %b: ref %AdaptNoncopyable = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyable = name_ref a, %a
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyable = name_ref b, %b
// CHECK:STDOUT:   %.loc22: %AdaptNoncopyable = bind_value %b.ref
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_not_copyable_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %AdaptNoncopyableIndirect: type = class_type @AdaptNoncopyableIndirect [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %Noncopyable, %i32) [template]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %tuple.type.2 [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Noncopyable = %Noncopyable.decl
// CHECK:STDOUT:     .AdaptNoncopyableIndirect = %AdaptNoncopyableIndirect.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Noncopyable.decl: type = class_decl @Noncopyable [template = constants.%Noncopyable] {} {}
// CHECK:STDOUT:   %AdaptNoncopyableIndirect.decl: type = class_decl @AdaptNoncopyableIndirect [template = constants.%AdaptNoncopyableIndirect] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %a.patt: %AdaptNoncopyableIndirect = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %AdaptNoncopyableIndirect = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %AdaptNoncopyableIndirect = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %AdaptNoncopyableIndirect = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc12_38: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [template = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyableIndirect = value_param runtime_param0
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc12_9: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [template = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %a: %AdaptNoncopyableIndirect = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %AdaptNoncopyableIndirect = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %AdaptNoncopyableIndirect = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNoncopyableIndirect {
// CHECK:STDOUT:   %int_32.loc9_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [template = constants.%Noncopyable]
// CHECK:STDOUT:   %int_32.loc9_28: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_28: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_31: %tuple.type.1 = tuple_literal (%i32.loc9_10, %Noncopyable.ref, %i32.loc9_28)
// CHECK:STDOUT:   %.loc9_32: type = converted %.loc9_31, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   adapt_decl %.loc9_32 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %tuple.type.2 [template = constants.%complete_type.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNoncopyableIndirect
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%a.param_patt: %AdaptNoncopyableIndirect) -> %return.param_patt: %AdaptNoncopyableIndirect {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyableIndirect = var b
// CHECK:STDOUT:   %b: ref %AdaptNoncopyableIndirect = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyableIndirect = name_ref a, %a
// CHECK:STDOUT:   %.loc20_38.1: %tuple.type.2 = as_compatible %a.ref
// CHECK:STDOUT:   %tuple.elem0.loc20_38.1: %i32 = tuple_access %.loc20_38.1, element0
// CHECK:STDOUT:   %.loc20_38.2: ref %tuple.type.2 = as_compatible %b.var
// CHECK:STDOUT:   %tuple.elem0.loc20_38.2: ref %i32 = tuple_access %.loc20_38.2, element0
// CHECK:STDOUT:   %.loc20_38.3: init %i32 = initialize_from %tuple.elem0.loc20_38.1 to %tuple.elem0.loc20_38.2
// CHECK:STDOUT:   %tuple.elem1.loc20: %Noncopyable = tuple_access %.loc20_38.1, element1
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyableIndirect = name_ref b, %b
// CHECK:STDOUT:   %.loc27_11.1: ref %tuple.type.2 = as_compatible %b.ref
// CHECK:STDOUT:   %tuple.elem0.loc27_11.1: ref %i32 = tuple_access %.loc27_11.1, element0
// CHECK:STDOUT:   %.loc27_11.2: %i32 = bind_value %tuple.elem0.loc27_11.1
// CHECK:STDOUT:   %.loc27_11.3: ref %tuple.type.2 = as_compatible %return
// CHECK:STDOUT:   %tuple.elem0.loc27_11.2: ref %i32 = tuple_access %.loc27_11.3, element0
// CHECK:STDOUT:   %.loc27_11.4: init %i32 = initialize_from %.loc27_11.2 to %tuple.elem0.loc27_11.2
// CHECK:STDOUT:   %tuple.elem1.loc27: ref %Noncopyable = tuple_access %.loc27_11.1, element1
// CHECK:STDOUT:   %.loc27_11.5: %Noncopyable = bind_value %tuple.elem1.loc27
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptStruct: type = class_type @AdaptStruct [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.e.f [template]
// CHECK:STDOUT:   %I.type: type = fn_type @I [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%AdaptStruct, %u32) [template]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [template]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .UInt = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptStruct = %AdaptStruct.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .InTuple = %InTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptStruct.decl: type = class_decl @AdaptStruct [template = constants.%AdaptStruct] {} {}
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [template = constants.%I] {
// CHECK:STDOUT:     %g.patt: %AdaptStruct = binding_pattern g
// CHECK:STDOUT:     %g.param_patt: %AdaptStruct = value_param_pattern %g.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %AdaptStruct = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %AdaptStruct = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc8_25: type = name_ref AdaptStruct, file.%AdaptStruct.decl [template = constants.%AdaptStruct]
// CHECK:STDOUT:     %g.param: %AdaptStruct = value_param runtime_param0
// CHECK:STDOUT:     %AdaptStruct.ref.loc8_9: type = name_ref AdaptStruct, file.%AdaptStruct.decl [template = constants.%AdaptStruct]
// CHECK:STDOUT:     %g: %AdaptStruct = bind_name g, %g.param
// CHECK:STDOUT:     %return.param: ref %AdaptStruct = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %AdaptStruct = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [template = constants.%InTuple] {
// CHECK:STDOUT:     %c.patt: %tuple.type.2 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %tuple.type.2 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc13_39: type = name_ref AdaptStruct, file.%AdaptStruct.decl [template = constants.%AdaptStruct]
// CHECK:STDOUT:     %int_32.loc13_52: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc13_52: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %.loc13_55.1: %tuple.type.1 = tuple_literal (%AdaptStruct.ref.loc13_39, %u32.loc13_52)
// CHECK:STDOUT:     %.loc13_55.2: type = converted %.loc13_55.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %c.param: %tuple.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %.loc13_32.1: type = splice_block %.loc13_32.3 [template = constants.%tuple.type.2] {
// CHECK:STDOUT:       %AdaptStruct.ref.loc13_16: type = name_ref AdaptStruct, file.%AdaptStruct.decl [template = constants.%AdaptStruct]
// CHECK:STDOUT:       %int_32.loc13_29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc13_29: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:       %.loc13_32.2: %tuple.type.1 = tuple_literal (%AdaptStruct.ref.loc13_16, %u32.loc13_29)
// CHECK:STDOUT:       %.loc13_32.3: type = converted %.loc13_32.2, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.2 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptStruct {
// CHECK:STDOUT:   %int_32.loc5_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc5_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_23: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [template = constants.%struct_type.e.f]
// CHECK:STDOUT:   adapt_decl %struct_type.e.f [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.e.f [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptStruct
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I(%g.param_patt: %AdaptStruct) -> %return.param_patt: %AdaptStruct {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %h.var: ref %AdaptStruct = var h
// CHECK:STDOUT:   %h: ref %AdaptStruct = bind_name h, %h.var
// CHECK:STDOUT:   %g.ref: %AdaptStruct = name_ref g, %g
// CHECK:STDOUT:   %.loc9_25.1: %struct_type.e.f = as_compatible %g.ref
// CHECK:STDOUT:   %.loc9_25.2: %i32 = struct_access %.loc9_25.1, element0
// CHECK:STDOUT:   %.loc9_25.3: ref %struct_type.e.f = as_compatible %h.var
// CHECK:STDOUT:   %.loc9_25.4: ref %i32 = struct_access %.loc9_25.3, element0
// CHECK:STDOUT:   %.loc9_25.5: init %i32 = initialize_from %.loc9_25.2 to %.loc9_25.4
// CHECK:STDOUT:   %.loc9_25.6: %i32 = struct_access %.loc9_25.1, element1
// CHECK:STDOUT:   %.loc9_25.7: ref %i32 = struct_access %.loc9_25.3, element1
// CHECK:STDOUT:   %.loc9_25.8: init %i32 = initialize_from %.loc9_25.6 to %.loc9_25.7
// CHECK:STDOUT:   %.loc9_25.9: init %struct_type.e.f = struct_init (%.loc9_25.5, %.loc9_25.8) to %.loc9_25.3
// CHECK:STDOUT:   %.loc9_25.10: init %AdaptStruct = as_compatible %.loc9_25.9
// CHECK:STDOUT:   %.loc9_25.11: init %AdaptStruct = converted %g.ref, %.loc9_25.10
// CHECK:STDOUT:   assign %h.var, %.loc9_25.11
// CHECK:STDOUT:   %h.ref: ref %AdaptStruct = name_ref h, %h
// CHECK:STDOUT:   %.loc10_11.1: ref %struct_type.e.f = as_compatible %h.ref
// CHECK:STDOUT:   %.loc10_11.2: ref %i32 = struct_access %.loc10_11.1, element0
// CHECK:STDOUT:   %.loc10_11.3: %i32 = bind_value %.loc10_11.2
// CHECK:STDOUT:   %.loc10_11.4: ref %struct_type.e.f = as_compatible %return
// CHECK:STDOUT:   %.loc10_11.5: ref %i32 = struct_access %.loc10_11.4, element0
// CHECK:STDOUT:   %.loc10_11.6: init %i32 = initialize_from %.loc10_11.3 to %.loc10_11.5
// CHECK:STDOUT:   %.loc10_11.7: ref %i32 = struct_access %.loc10_11.1, element1
// CHECK:STDOUT:   %.loc10_11.8: %i32 = bind_value %.loc10_11.7
// CHECK:STDOUT:   %.loc10_11.9: ref %i32 = struct_access %.loc10_11.4, element1
// CHECK:STDOUT:   %.loc10_11.10: init %i32 = initialize_from %.loc10_11.8 to %.loc10_11.9
// CHECK:STDOUT:   %.loc10_11.11: init %struct_type.e.f = struct_init (%.loc10_11.6, %.loc10_11.10) to %.loc10_11.4
// CHECK:STDOUT:   %.loc10_11.12: init %AdaptStruct = as_compatible %.loc10_11.11
// CHECK:STDOUT:   %.loc10_11.13: init %AdaptStruct = converted %h.ref, %.loc10_11.12
// CHECK:STDOUT:   return %.loc10_11.13 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param_patt: %tuple.type.2) -> %return.param_patt: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %tuple.type.2 = var d
// CHECK:STDOUT:   %d: ref %tuple.type.2 = bind_name d, %d.var
// CHECK:STDOUT:   %c.ref: %tuple.type.2 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc14_31.1: %AdaptStruct = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_31.1: %struct_type.e.f = as_compatible %tuple.elem0.loc14_31.1
// CHECK:STDOUT:   %.loc14_31.2: %i32 = struct_access %.loc14_31.1, element0
// CHECK:STDOUT:   %tuple.elem0.loc14_31.2: ref %AdaptStruct = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc14_31.3: ref %struct_type.e.f = as_compatible %tuple.elem0.loc14_31.2
// CHECK:STDOUT:   %.loc14_31.4: ref %i32 = struct_access %.loc14_31.3, element0
// CHECK:STDOUT:   %.loc14_31.5: init %i32 = initialize_from %.loc14_31.2 to %.loc14_31.4
// CHECK:STDOUT:   %.loc14_31.6: %i32 = struct_access %.loc14_31.1, element1
// CHECK:STDOUT:   %.loc14_31.7: ref %i32 = struct_access %.loc14_31.3, element1
// CHECK:STDOUT:   %.loc14_31.8: init %i32 = initialize_from %.loc14_31.6 to %.loc14_31.7
// CHECK:STDOUT:   %.loc14_31.9: init %struct_type.e.f = struct_init (%.loc14_31.5, %.loc14_31.8) to %.loc14_31.3
// CHECK:STDOUT:   %.loc14_31.10: init %AdaptStruct = as_compatible %.loc14_31.9
// CHECK:STDOUT:   %.loc14_31.11: init %AdaptStruct = converted %tuple.elem0.loc14_31.1, %.loc14_31.10
// CHECK:STDOUT:   %tuple.elem1.loc14_31.1: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %tuple.elem1.loc14_31.2: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc14_31.12: init %u32 = initialize_from %tuple.elem1.loc14_31.1 to %tuple.elem1.loc14_31.2
// CHECK:STDOUT:   %.loc14_31.13: init %tuple.type.2 = tuple_init (%.loc14_31.11, %.loc14_31.12) to %d.var
// CHECK:STDOUT:   %.loc14_32: init %tuple.type.2 = converted %c.ref, %.loc14_31.13
// CHECK:STDOUT:   assign %d.var, %.loc14_32
// CHECK:STDOUT:   %d.ref: ref %tuple.type.2 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc15_10.1: ref %AdaptStruct = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc15_10.1: ref %struct_type.e.f = as_compatible %tuple.elem0.loc15_10.1
// CHECK:STDOUT:   %.loc15_10.2: ref %i32 = struct_access %.loc15_10.1, element0
// CHECK:STDOUT:   %.loc15_10.3: %i32 = bind_value %.loc15_10.2
// CHECK:STDOUT:   %tuple.elem0.loc15_10.2: ref %AdaptStruct = tuple_access %return, element0
// CHECK:STDOUT:   %.loc15_10.4: ref %struct_type.e.f = as_compatible %tuple.elem0.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.5: ref %i32 = struct_access %.loc15_10.4, element0
// CHECK:STDOUT:   %.loc15_10.6: init %i32 = initialize_from %.loc15_10.3 to %.loc15_10.5
// CHECK:STDOUT:   %.loc15_10.7: ref %i32 = struct_access %.loc15_10.1, element1
// CHECK:STDOUT:   %.loc15_10.8: %i32 = bind_value %.loc15_10.7
// CHECK:STDOUT:   %.loc15_10.9: ref %i32 = struct_access %.loc15_10.4, element1
// CHECK:STDOUT:   %.loc15_10.10: init %i32 = initialize_from %.loc15_10.8 to %.loc15_10.9
// CHECK:STDOUT:   %.loc15_10.11: init %struct_type.e.f = struct_init (%.loc15_10.6, %.loc15_10.10) to %.loc15_10.4
// CHECK:STDOUT:   %.loc15_10.12: init %AdaptStruct = as_compatible %.loc15_10.11
// CHECK:STDOUT:   %.loc15_10.13: init %AdaptStruct = converted %tuple.elem0.loc15_10.1, %.loc15_10.12
// CHECK:STDOUT:   %tuple.elem1.loc15_10.1: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc15_10.14: %u32 = bind_value %tuple.elem1.loc15_10.1
// CHECK:STDOUT:   %tuple.elem1.loc15_10.2: ref %u32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc15_10.15: init %u32 = initialize_from %.loc15_10.14 to %tuple.elem1.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.16: init %tuple.type.2 = tuple_init (%.loc15_10.13, %.loc15_10.15) to %return
// CHECK:STDOUT:   %.loc15_11: init %tuple.type.2 = converted %d.ref, %.loc15_10.16
// CHECK:STDOUT:   return %.loc15_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
