// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/adapt_copy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/adapt_copy.carbon

// --- adapt_copyable.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class AdaptCopyable {
  adapt i32;
}
//@dump-sem-ir-end

// We can copy an adapter by copying its value if its initializing
// representation is a copy of its value representation.
fn F(c: AdaptCopyable) -> AdaptCopyable {
  //@dump-sem-ir-begin
  var d: AdaptCopyable = c;
  return d;
  //@dump-sem-ir-end
}

fn InTuple(c: (AdaptCopyable, u32)) -> (AdaptCopyable, u32) {
  var d: (AdaptCopyable, u32) = c;
  return d;
}

// --- adapt_copyable_tuple.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class AdaptTuple {
  adapt (i32, i32);
}
//@dump-sem-ir-end

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
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+4]]:37: note: in copy of `AdaptNoncopyableIndirect` [InCopy]
  // CHECK:STDERR:   var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR:                                     ^
  // CHECK:STDERR:
  var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+7]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+4]]:10: note: in copy of `AdaptNoncopyableIndirect` [InCopy]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return b;
}

// --- adapt_copyable_struct.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class AdaptStruct {
  adapt {.e: i32, .f: i32};
}
//@dump-sem-ir-end

fn I(g: AdaptStruct) -> AdaptStruct {
  //@dump-sem-ir-begin
  var h: AdaptStruct = g;
  return h;
  //@dump-sem-ir-end
}

fn InTuple(c: (AdaptStruct, u32)) -> (AdaptStruct, u32) {
  var d: (AdaptStruct, u32) = c;
  return d;
}

// CHECK:STDOUT: --- adapt_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptCopyable: type = class_type @AdaptCopyable [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.cdf: type = pattern_type %AdaptCopyable [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AdaptCopyable.decl: type = class_decl @AdaptCopyable [concrete = constants.%AdaptCopyable] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptCopyable {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%i32.builtin [concrete = constants.%complete_type.f8a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptCopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %AdaptCopyable) -> %AdaptCopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cdf = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cdf = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptCopyable = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %AdaptCopyable = name_ref c, %c
// CHECK:STDOUT:   assign %d.var, %c.ref
// CHECK:STDOUT:   %AdaptCopyable.ref.loc14: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:   %d: ref %AdaptCopyable = bind_name d, %d.var
// CHECK:STDOUT:   %d.ref: ref %AdaptCopyable = name_ref d, %d
// CHECK:STDOUT:   %.loc15: %AdaptCopyable = bind_value %d.ref
// CHECK:STDOUT:   return %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptTuple: type = class_type @AdaptTuple [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %complete_type.65d: <witness> = complete_type_witness %tuple.type.d07 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AdaptTuple.decl: type = class_decl @AdaptTuple [concrete = constants.%AdaptTuple] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptTuple {
// CHECK:STDOUT:   %int_32.loc6_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc6_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6_18: %tuple.type.24b = tuple_literal (%i32.loc6_10, %i32.loc6_15)
// CHECK:STDOUT:   %.loc6_19: type = converted %.loc6_18, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   adapt_decl %.loc6_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%tuple.type.d07 [concrete = constants.%complete_type.65d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptTuple
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptStruct: type = class_type @AdaptStruct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.511: <witness> = complete_type_witness %struct_type.e.f [concrete]
// CHECK:STDOUT:   %pattern_type.f45: type = pattern_type %AdaptStruct [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AdaptStruct.decl: type = class_decl @AdaptStruct [concrete = constants.%AdaptStruct] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptStruct {
// CHECK:STDOUT:   %int_32.loc6_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc6_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete = constants.%struct_type.e.f]
// CHECK:STDOUT:   adapt_decl %struct_type.e.f [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.e.f [concrete = constants.%complete_type.511]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptStruct
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I(%g.param: %AdaptStruct) -> %return.param: %AdaptStruct {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.f45 = binding_pattern h [concrete]
// CHECK:STDOUT:     %h.var_patt: %pattern_type.f45 = var_pattern %h.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %h.var: ref %AdaptStruct = var %h.var_patt
// CHECK:STDOUT:   %g.ref: %AdaptStruct = name_ref g, %g
// CHECK:STDOUT:   %.loc12_3.1: %struct_type.e.f = as_compatible %g.ref
// CHECK:STDOUT:   %.loc12_3.2: %i32 = struct_access %.loc12_3.1, element0
// CHECK:STDOUT:   %.loc12_3.3: ref %struct_type.e.f = as_compatible %h.var
// CHECK:STDOUT:   %.loc12_3.4: ref %i32 = struct_access %.loc12_3.3, element0
// CHECK:STDOUT:   %.loc12_3.5: init %i32 = initialize_from %.loc12_3.2 to %.loc12_3.4
// CHECK:STDOUT:   %.loc12_3.6: %i32 = struct_access %.loc12_3.1, element1
// CHECK:STDOUT:   %.loc12_3.7: ref %i32 = struct_access %.loc12_3.3, element1
// CHECK:STDOUT:   %.loc12_3.8: init %i32 = initialize_from %.loc12_3.6 to %.loc12_3.7
// CHECK:STDOUT:   %.loc12_3.9: init %struct_type.e.f = struct_init (%.loc12_3.5, %.loc12_3.8) to %.loc12_3.3
// CHECK:STDOUT:   %.loc12_3.10: init %AdaptStruct = as_compatible %.loc12_3.9
// CHECK:STDOUT:   %.loc12_3.11: init %AdaptStruct = converted %g.ref, %.loc12_3.10
// CHECK:STDOUT:   assign %h.var, %.loc12_3.11
// CHECK:STDOUT:   %AdaptStruct.ref.loc12: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:   %h: ref %AdaptStruct = bind_name h, %h.var
// CHECK:STDOUT:   %h.ref: ref %AdaptStruct = name_ref h, %h
// CHECK:STDOUT:   %.loc13_11.1: ref %struct_type.e.f = as_compatible %h.ref
// CHECK:STDOUT:   %.loc13_11.2: ref %i32 = struct_access %.loc13_11.1, element0
// CHECK:STDOUT:   %.loc13_11.3: %i32 = bind_value %.loc13_11.2
// CHECK:STDOUT:   %.loc13_11.4: ref %struct_type.e.f = as_compatible %return
// CHECK:STDOUT:   %.loc13_11.5: ref %i32 = struct_access %.loc13_11.4, element0
// CHECK:STDOUT:   %.loc13_11.6: init %i32 = initialize_from %.loc13_11.3 to %.loc13_11.5
// CHECK:STDOUT:   %.loc13_11.7: ref %i32 = struct_access %.loc13_11.1, element1
// CHECK:STDOUT:   %.loc13_11.8: %i32 = bind_value %.loc13_11.7
// CHECK:STDOUT:   %.loc13_11.9: ref %i32 = struct_access %.loc13_11.4, element1
// CHECK:STDOUT:   %.loc13_11.10: init %i32 = initialize_from %.loc13_11.8 to %.loc13_11.9
// CHECK:STDOUT:   %.loc13_11.11: init %struct_type.e.f = struct_init (%.loc13_11.6, %.loc13_11.10) to %.loc13_11.4
// CHECK:STDOUT:   %.loc13_11.12: init %AdaptStruct = as_compatible %.loc13_11.11
// CHECK:STDOUT:   %.loc13_11.13: init %AdaptStruct = converted %h.ref, %.loc13_11.12
// CHECK:STDOUT:   return %.loc13_11.13 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
