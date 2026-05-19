// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/adapt_copy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/adapt_copy.carbon

// TODO: Decide whether an adapter for a copyable type is copyable. As
// demonstrated in this test, our behavior is currently inconsistent.

// --- fail_adapt_copyable.carbon

library "[[@TEST_NAME]]";

class AdaptCopyable {
  adapt i32;
}

fn F(c: AdaptCopyable) -> AdaptCopyable {
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:26: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var d: AdaptCopyable = c;
  // CHECK:STDERR:                          ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:26: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var d: AdaptCopyable = c;
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  var d: AdaptCopyable = c;
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:10: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return d;
}

fn InTuple(c: (AdaptCopyable, u32)) -> (AdaptCopyable, u32) {
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:33: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var d: (AdaptCopyable, u32) = c;
  // CHECK:STDERR:                                 ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:33: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var d: (AdaptCopyable, u32) = c;
  // CHECK:STDERR:                                 ^
  // CHECK:STDERR:
  var d: (AdaptCopyable, u32) = c;
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:10: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
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
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+7]]:29: error: cannot copy value of type `AdaptNoncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var b: AdaptNoncopyable = a;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+4]]:29: note: type `AdaptNoncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var b: AdaptNoncopyable = a;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR:
  var b: AdaptNoncopyable = a;
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `AdaptNoncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+4]]:10: note: type `AdaptNoncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
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
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+10]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+7]]:3: note: type `Noncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+4]]:37: note: in copy of `AdaptNoncopyableIndirect` [InCopy]
  // CHECK:STDERR:   var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR:                                     ^
  // CHECK:STDERR:
  var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+10]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+7]]:3: note: type `Noncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
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

// CHECK:STDOUT: --- fail_adapt_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptCopyable: type = class_type @AdaptCopyable [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.87e: type = pattern_type %AdaptCopyable [concrete]
// CHECK:STDOUT:   %.5de: Core.Form = init_form %AdaptCopyable [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc16_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.523: %tuple.type.24b = tuple_value (%AdaptCopyable, %u32) [concrete]
// CHECK:STDOUT:   %tuple.type.74b: type = tuple_type (%AdaptCopyable, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.f72: type = pattern_type %tuple.type.74b [concrete]
// CHECK:STDOUT:   %.056: Core.Form = init_form %tuple.type.74b [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32.builtin: type = int_type unsigned, %int_32 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.5: type = fn_type @Destroy.Op.loc35_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.5: %Destroy.Op.type.af7ec0.5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptCopyable = %AdaptCopyable.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .InTuple = %InTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptCopyable.decl: type = class_decl @AdaptCopyable [concrete = constants.%AdaptCopyable] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.87e = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.87e = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.87e = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.87e = return_slot_pattern %return.param_patt, %AdaptCopyable.ref.loc8_27 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc8_27: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %AdaptCopyable.ref.loc8_27 [concrete = constants.%.5de]
// CHECK:STDOUT:     %c.param: %AdaptCopyable = value_param call_param0
// CHECK:STDOUT:     %AdaptCopyable.ref.loc8_9: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %c: %AdaptCopyable = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptCopyable = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptCopyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.f72 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.f72 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f72 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.f72 = return_slot_pattern %return.param_patt, %.loc27_59.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc27_41: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %u32.loc27_56: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc27_59.1: %tuple.type.24b = tuple_literal (%AdaptCopyable.ref.loc27_41, %u32.loc27_56) [concrete = constants.%tuple.523]
// CHECK:STDOUT:     %.loc27_59.2: type = converted %.loc27_59.1, constants.%tuple.type.74b [concrete = constants.%tuple.type.74b]
// CHECK:STDOUT:     %.loc27_59.3: Core.Form = init_form %.loc27_59.2 [concrete = constants.%.056]
// CHECK:STDOUT:     %c.param: %tuple.type.74b = value_param call_param0
// CHECK:STDOUT:     %.loc27_34.1: type = splice_block %.loc27_34.3 [concrete = constants.%tuple.type.74b] {
// CHECK:STDOUT:       %AdaptCopyable.ref.loc27_16: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:       %u32.loc27_31: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc27_34.2: %tuple.type.24b = tuple_literal (%AdaptCopyable.ref.loc27_16, %u32.loc27_31) [concrete = constants.%tuple.523]
// CHECK:STDOUT:       %.loc27_34.3: type = converted %.loc27_34.2, constants.%tuple.type.74b [concrete = constants.%tuple.type.74b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.74b = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.74b = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.74b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptCopyable {
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%i32.builtin [concrete = constants.%complete_type.f8a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptCopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %AdaptCopyable) -> out %return.param: %AdaptCopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.87e = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.87e = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptCopyable = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %AdaptCopyable = name_ref c, %c
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   %AdaptCopyable.ref.loc16: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:   %d: ref %AdaptCopyable = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %AdaptCopyable = name_ref d, %d
// CHECK:STDOUT:   %.loc24: %AdaptCopyable = acquire_value %d.ref
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc16_3.2(%self.param: ref %AdaptCopyable) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.74b) -> out %return.param: %tuple.type.74b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.f72 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.f72 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %tuple.type.74b = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.74b = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc35: %AdaptCopyable = tuple_access %c.ref, element0
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   %.loc35_29.1: type = splice_block %.loc35_29.3 [concrete = constants.%tuple.type.74b] {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc35: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %u32.loc35: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc35_29.2: %tuple.type.24b = tuple_literal (%AdaptCopyable.ref.loc35, %u32.loc35) [concrete = constants.%tuple.523]
// CHECK:STDOUT:     %.loc35_29.3: type = converted %.loc35_29.2, constants.%tuple.type.74b [concrete = constants.%tuple.type.74b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.74b = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %tuple.type.74b = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc43: ref %AdaptCopyable = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc43: %AdaptCopyable = acquire_value %tuple.elem0.loc43
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.1dc86d.5
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_3.1(%self.param: ref %u32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_3.2(%self.param: ref %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc35_3.3(%self.param: ref %tuple.type.74b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptTuple: type = class_type @AdaptTuple [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.8e3: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.87c: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.6b9: <witness> = complete_type_witness %tuple.type.87c [concrete]
// CHECK:STDOUT:   %pattern_type.b8a: type = pattern_type %AdaptTuple [concrete]
// CHECK:STDOUT:   %.4dc: Core.Form = init_form %AdaptTuple [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.66a: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet.66a [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.c8f: %tuple.type.24b = tuple_value (%AdaptTuple, %u32) [concrete]
// CHECK:STDOUT:   %tuple.type.68d: type = tuple_type (%AdaptTuple, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.a38: type = pattern_type %tuple.type.68d [concrete]
// CHECK:STDOUT:   %.cc0: Core.Form = init_form %tuple.type.68d [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32.builtin: type = int_type unsigned, %int_32 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.a58: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.dc5: %UInt.as.Copy.impl.Op.type.a58 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0f3: <witness> = impl_witness imports.%Copy.impl_witness_table.a92, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.3cf: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.6ad: %UInt.as.Copy.impl.Op.type.3cf = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.400: %Copy.type = facet_value %u32, (%Copy.impl_witness.0f3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.1c6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.400) [concrete]
// CHECK:STDOUT:   %.2e6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.1c6, %Copy.facet.400 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.6ad, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.7: type = fn_type @Destroy.Op.loc14_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.7: %Destroy.Op.type.af7ec0.7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.ca7: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.a58) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.dc5)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a92 = impl_witness_table (%Core.import_ref.ca7), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptTuple = %AdaptTuple.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .InTuple = %InTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptTuple.decl: type = class_decl @AdaptTuple [concrete = constants.%AdaptTuple] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.b8a = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.b8a = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b8a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.b8a = return_slot_pattern %return.param_patt, %AdaptTuple.ref.loc8_24 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc8_24: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %AdaptTuple.ref.loc8_24 [concrete = constants.%.4dc]
// CHECK:STDOUT:     %c.param: %AdaptTuple = value_param call_param0
// CHECK:STDOUT:     %AdaptTuple.ref.loc8_9: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %c: %AdaptTuple = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptTuple = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptTuple = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.a38 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.a38 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a38 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a38 = return_slot_pattern %return.param_patt, %.loc13_53.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc13_38: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %u32.loc13_50: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc13_53.1: %tuple.type.24b = tuple_literal (%AdaptTuple.ref.loc13_38, %u32.loc13_50) [concrete = constants.%tuple.c8f]
// CHECK:STDOUT:     %.loc13_53.2: type = converted %.loc13_53.1, constants.%tuple.type.68d [concrete = constants.%tuple.type.68d]
// CHECK:STDOUT:     %.loc13_53.3: Core.Form = init_form %.loc13_53.2 [concrete = constants.%.cc0]
// CHECK:STDOUT:     %c.param: %tuple.type.68d = value_param call_param0
// CHECK:STDOUT:     %.loc13_31.1: type = splice_block %.loc13_31.3 [concrete = constants.%tuple.type.68d] {
// CHECK:STDOUT:       %AdaptTuple.ref.loc13_16: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:       %u32.loc13_28: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc13_31.2: %tuple.type.24b = tuple_literal (%AdaptTuple.ref.loc13_16, %u32.loc13_28) [concrete = constants.%tuple.c8f]
// CHECK:STDOUT:       %.loc13_31.3: type = converted %.loc13_31.2, constants.%tuple.type.68d [concrete = constants.%tuple.type.68d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.68d = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.68d = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.68d = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptTuple {
// CHECK:STDOUT:   %i32.loc5_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc5_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: %tuple.type.24b = tuple_literal (%i32.loc5_10, %i32.loc5_15) [concrete = constants.%tuple.8e3]
// CHECK:STDOUT:   %.loc5_19: type = converted %.loc5_18, constants.%tuple.type.87c [concrete = constants.%tuple.type.87c]
// CHECK:STDOUT:   adapt_decl %.loc5_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%tuple.type.87c [concrete = constants.%complete_type.6b9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptTuple
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %AdaptTuple) -> out %return.param: %AdaptTuple {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.b8a = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.b8a = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptTuple = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %AdaptTuple = name_ref c, %c
// CHECK:STDOUT:   %.loc9_3.1: %tuple.type.87c = as_compatible %c.ref
// CHECK:STDOUT:   %tuple.elem0.loc9_3.1: %i32 = tuple_access %.loc9_3.1, element0
// CHECK:STDOUT:   %impl.elem0.loc9_3.1: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc9_3.1: <bound method> = bound_method %tuple.elem0.loc9_3.1, %impl.elem0.loc9_3.1
// CHECK:STDOUT:   %specific_fn.loc9_3.1: <specific function> = specific_function %impl.elem0.loc9_3.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.2: <bound method> = bound_method %tuple.elem0.loc9_3.1, %specific_fn.loc9_3.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.1: init %i32 = call %bound_method.loc9_3.2(%tuple.elem0.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %tuple.type.87c = as_compatible %d.var
// CHECK:STDOUT:   %tuple.elem0.loc9_3.2: ref %i32 = tuple_access %.loc9_3.2, element0
// CHECK:STDOUT:   %.loc9_3.3: init %i32 to %tuple.elem0.loc9_3.2 = in_place_init %Int.as.Copy.impl.Op.call.loc9_3.1
// CHECK:STDOUT:   %tuple.elem1.loc9_3.1: %i32 = tuple_access %.loc9_3.1, element1
// CHECK:STDOUT:   %impl.elem0.loc9_3.2: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc9_3.3: <bound method> = bound_method %tuple.elem1.loc9_3.1, %impl.elem0.loc9_3.2
// CHECK:STDOUT:   %specific_fn.loc9_3.2: <specific function> = specific_function %impl.elem0.loc9_3.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.4: <bound method> = bound_method %tuple.elem1.loc9_3.1, %specific_fn.loc9_3.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.2: init %i32 = call %bound_method.loc9_3.4(%tuple.elem1.loc9_3.1)
// CHECK:STDOUT:   %tuple.elem1.loc9_3.2: ref %i32 = tuple_access %.loc9_3.2, element1
// CHECK:STDOUT:   %.loc9_3.4: init %i32 to %tuple.elem1.loc9_3.2 = in_place_init %Int.as.Copy.impl.Op.call.loc9_3.2
// CHECK:STDOUT:   %.loc9_3.5: init %tuple.type.87c to %.loc9_3.2 = tuple_init (%.loc9_3.3, %.loc9_3.4)
// CHECK:STDOUT:   %.loc9_3.6: init %AdaptTuple = as_compatible %.loc9_3.5
// CHECK:STDOUT:   %.loc9_3.7: init %AdaptTuple = converted %c.ref, %.loc9_3.6
// CHECK:STDOUT:   assign %d.var, %.loc9_3.7
// CHECK:STDOUT:   %AdaptTuple.ref.loc9: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:   %d: ref %AdaptTuple = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %AdaptTuple = name_ref d, %d
// CHECK:STDOUT:   %.loc10_11.1: ref %tuple.type.87c = as_compatible %d.ref
// CHECK:STDOUT:   %tuple.elem0.loc10_11.1: ref %i32 = tuple_access %.loc10_11.1, element0
// CHECK:STDOUT:   %.loc10_11.2: %i32 = acquire_value %tuple.elem0.loc10_11.1
// CHECK:STDOUT:   %impl.elem0.loc10_11.1: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.2, %impl.elem0.loc10_11.1
// CHECK:STDOUT:   %specific_fn.loc10_11.1: <specific function> = specific_function %impl.elem0.loc10_11.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.2, %specific_fn.loc10_11.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.1: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.2)
// CHECK:STDOUT:   %.loc10_11.3: ref %tuple.type.87c = as_compatible %return.param
// CHECK:STDOUT:   %tuple.elem0.loc10_11.2: ref %i32 = tuple_access %.loc10_11.3, element0
// CHECK:STDOUT:   %.loc10_11.4: init %i32 to %tuple.elem0.loc10_11.2 = in_place_init %Int.as.Copy.impl.Op.call.loc10_11.1
// CHECK:STDOUT:   %tuple.elem1.loc10_11.1: ref %i32 = tuple_access %.loc10_11.1, element1
// CHECK:STDOUT:   %.loc10_11.5: %i32 = acquire_value %tuple.elem1.loc10_11.1
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.5, %impl.elem0.loc10_11.2
// CHECK:STDOUT:   %specific_fn.loc10_11.2: <specific function> = specific_function %impl.elem0.loc10_11.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.4: <bound method> = bound_method %.loc10_11.5, %specific_fn.loc10_11.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.2: init %i32 = call %bound_method.loc10_11.4(%.loc10_11.5)
// CHECK:STDOUT:   %tuple.elem1.loc10_11.2: ref %i32 = tuple_access %.loc10_11.3, element1
// CHECK:STDOUT:   %.loc10_11.6: init %i32 to %tuple.elem1.loc10_11.2 = in_place_init %Int.as.Copy.impl.Op.call.loc10_11.2
// CHECK:STDOUT:   %.loc10_11.7: init %tuple.type.87c to %.loc10_11.3 = tuple_init (%.loc10_11.4, %.loc10_11.6)
// CHECK:STDOUT:   %.loc10_11.8: init %AdaptTuple = as_compatible %.loc10_11.7
// CHECK:STDOUT:   %.loc10_11.9: init %AdaptTuple = converted %d.ref, %.loc10_11.8
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return %.loc10_11.9 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %tuple.type.87c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.4(%self.param: ref %AdaptTuple) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.68d) -> out %return.param: %tuple.type.68d {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.a38 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.a38 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %tuple.type.68d = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.68d = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc14_30.1: %AdaptTuple = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_30.1: %tuple.type.87c = as_compatible %tuple.elem0.loc14_30.1
// CHECK:STDOUT:   %tuple.elem0.loc14_30.2: %i32 = tuple_access %.loc14_30.1, element0
// CHECK:STDOUT:   %impl.elem0.loc14_30.1: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc14_30.1: <bound method> = bound_method %tuple.elem0.loc14_30.2, %impl.elem0.loc14_30.1
// CHECK:STDOUT:   %specific_fn.loc14_30.1: <specific function> = specific_function %impl.elem0.loc14_30.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_30.2: <bound method> = bound_method %tuple.elem0.loc14_30.2, %specific_fn.loc14_30.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_30.1: init %i32 = call %bound_method.loc14_30.2(%tuple.elem0.loc14_30.2)
// CHECK:STDOUT:   %tuple.elem0.loc14_30.3: ref %AdaptTuple = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc14_30.2: ref %tuple.type.87c = as_compatible %tuple.elem0.loc14_30.3
// CHECK:STDOUT:   %tuple.elem0.loc14_30.4: ref %i32 = tuple_access %.loc14_30.2, element0
// CHECK:STDOUT:   %.loc14_30.3: init %i32 to %tuple.elem0.loc14_30.4 = in_place_init %Int.as.Copy.impl.Op.call.loc14_30.1
// CHECK:STDOUT:   %tuple.elem1.loc14_30.1: %i32 = tuple_access %.loc14_30.1, element1
// CHECK:STDOUT:   %impl.elem0.loc14_30.2: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc14_30.3: <bound method> = bound_method %tuple.elem1.loc14_30.1, %impl.elem0.loc14_30.2
// CHECK:STDOUT:   %specific_fn.loc14_30.2: <specific function> = specific_function %impl.elem0.loc14_30.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_30.4: <bound method> = bound_method %tuple.elem1.loc14_30.1, %specific_fn.loc14_30.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_30.2: init %i32 = call %bound_method.loc14_30.4(%tuple.elem1.loc14_30.1)
// CHECK:STDOUT:   %tuple.elem1.loc14_30.2: ref %i32 = tuple_access %.loc14_30.2, element1
// CHECK:STDOUT:   %.loc14_30.4: init %i32 to %tuple.elem1.loc14_30.2 = in_place_init %Int.as.Copy.impl.Op.call.loc14_30.2
// CHECK:STDOUT:   %.loc14_30.5: init %tuple.type.87c to %.loc14_30.2 = tuple_init (%.loc14_30.3, %.loc14_30.4)
// CHECK:STDOUT:   %.loc14_30.6: init %AdaptTuple = as_compatible %.loc14_30.5
// CHECK:STDOUT:   %.loc14_30.7: init %AdaptTuple = converted %tuple.elem0.loc14_30.1, %.loc14_30.6
// CHECK:STDOUT:   %tuple.elem1.loc14_30.3: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %impl.elem0.loc14_30.3: %.2e6 = impl_witness_access constants.%Copy.impl_witness.0f3, element0 [concrete = constants.%UInt.as.Copy.impl.Op.6ad]
// CHECK:STDOUT:   %bound_method.loc14_30.5: <bound method> = bound_method %tuple.elem1.loc14_30.3, %impl.elem0.loc14_30.3
// CHECK:STDOUT:   %specific_fn.loc14_30.3: <specific function> = specific_function %impl.elem0.loc14_30.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_30.6: <bound method> = bound_method %tuple.elem1.loc14_30.3, %specific_fn.loc14_30.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc14: init %u32 = call %bound_method.loc14_30.6(%tuple.elem1.loc14_30.3)
// CHECK:STDOUT:   %tuple.elem1.loc14_30.4: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc14_30.8: init %u32 to %tuple.elem1.loc14_30.4 = in_place_init %UInt.as.Copy.impl.Op.call.loc14
// CHECK:STDOUT:   %.loc14_30.9: init %tuple.type.68d to %d.var = tuple_init (%.loc14_30.7, %.loc14_30.8)
// CHECK:STDOUT:   %.loc14_3: init %tuple.type.68d = converted %c.ref, %.loc14_30.9
// CHECK:STDOUT:   assign %d.var, %.loc14_3
// CHECK:STDOUT:   %.loc14_26.1: type = splice_block %.loc14_26.3 [concrete = constants.%tuple.type.68d] {
// CHECK:STDOUT:     %AdaptTuple.ref.loc14: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %u32.loc14: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc14_26.2: %tuple.type.24b = tuple_literal (%AdaptTuple.ref.loc14, %u32.loc14) [concrete = constants.%tuple.c8f]
// CHECK:STDOUT:     %.loc14_26.3: type = converted %.loc14_26.2, constants.%tuple.type.68d [concrete = constants.%tuple.type.68d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.68d = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %tuple.type.68d = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc15_10.1: ref %AdaptTuple = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc15_10.1: ref %tuple.type.87c = as_compatible %tuple.elem0.loc15_10.1
// CHECK:STDOUT:   %tuple.elem0.loc15_10.2: ref %i32 = tuple_access %.loc15_10.1, element0
// CHECK:STDOUT:   %.loc15_10.2: %i32 = acquire_value %tuple.elem0.loc15_10.2
// CHECK:STDOUT:   %impl.elem0.loc15_10.1: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc15_10.1: <bound method> = bound_method %.loc15_10.2, %impl.elem0.loc15_10.1
// CHECK:STDOUT:   %specific_fn.loc15_10.1: <specific function> = specific_function %impl.elem0.loc15_10.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.2: <bound method> = bound_method %.loc15_10.2, %specific_fn.loc15_10.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.1: init %i32 = call %bound_method.loc15_10.2(%.loc15_10.2)
// CHECK:STDOUT:   %tuple.elem0.loc15_10.3: ref %AdaptTuple = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc15_10.3: ref %tuple.type.87c = as_compatible %tuple.elem0.loc15_10.3
// CHECK:STDOUT:   %tuple.elem0.loc15_10.4: ref %i32 = tuple_access %.loc15_10.3, element0
// CHECK:STDOUT:   %.loc15_10.4: init %i32 to %tuple.elem0.loc15_10.4 = in_place_init %Int.as.Copy.impl.Op.call.loc15_10.1
// CHECK:STDOUT:   %tuple.elem1.loc15_10.1: ref %i32 = tuple_access %.loc15_10.1, element1
// CHECK:STDOUT:   %.loc15_10.5: %i32 = acquire_value %tuple.elem1.loc15_10.1
// CHECK:STDOUT:   %impl.elem0.loc15_10.2: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc15_10.3: <bound method> = bound_method %.loc15_10.5, %impl.elem0.loc15_10.2
// CHECK:STDOUT:   %specific_fn.loc15_10.2: <specific function> = specific_function %impl.elem0.loc15_10.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.4: <bound method> = bound_method %.loc15_10.5, %specific_fn.loc15_10.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.2: init %i32 = call %bound_method.loc15_10.4(%.loc15_10.5)
// CHECK:STDOUT:   %tuple.elem1.loc15_10.2: ref %i32 = tuple_access %.loc15_10.3, element1
// CHECK:STDOUT:   %.loc15_10.6: init %i32 to %tuple.elem1.loc15_10.2 = in_place_init %Int.as.Copy.impl.Op.call.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.7: init %tuple.type.87c to %.loc15_10.3 = tuple_init (%.loc15_10.4, %.loc15_10.6)
// CHECK:STDOUT:   %.loc15_10.8: init %AdaptTuple = as_compatible %.loc15_10.7
// CHECK:STDOUT:   %.loc15_10.9: init %AdaptTuple = converted %tuple.elem0.loc15_10.1, %.loc15_10.8
// CHECK:STDOUT:   %tuple.elem1.loc15_10.3: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc15_10.10: %u32 = acquire_value %tuple.elem1.loc15_10.3
// CHECK:STDOUT:   %impl.elem0.loc15_10.3: %.2e6 = impl_witness_access constants.%Copy.impl_witness.0f3, element0 [concrete = constants.%UInt.as.Copy.impl.Op.6ad]
// CHECK:STDOUT:   %bound_method.loc15_10.5: <bound method> = bound_method %.loc15_10.10, %impl.elem0.loc15_10.3
// CHECK:STDOUT:   %specific_fn.loc15_10.3: <specific function> = specific_function %impl.elem0.loc15_10.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.6: <bound method> = bound_method %.loc15_10.10, %specific_fn.loc15_10.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc15: init %u32 = call %bound_method.loc15_10.6(%.loc15_10.10)
// CHECK:STDOUT:   %tuple.elem1.loc15_10.4: ref %u32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc15_10.11: init %u32 to %tuple.elem1.loc15_10.4 = in_place_init %UInt.as.Copy.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_10.12: init %tuple.type.68d to %return.param = tuple_init (%.loc15_10.9, %.loc15_10.11)
// CHECK:STDOUT:   %.loc15_11: init %tuple.type.68d = converted %d.ref, %.loc15_10.12
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.1dc86d.7
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return %.loc15_11 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.1(%self.param: ref %u32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.2(%self.param: ref %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.3(%self.param: ref %tuple.type.68d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_not_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptNoncopyable: type = class_type @AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.146: type = pattern_type %AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %.5ee: Core.Form = init_form %AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc20_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Noncopyable = %Noncopyable.decl
// CHECK:STDOUT:     .AdaptNoncopyable = %AdaptNoncopyable.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Noncopyable.decl: type = class_decl @Noncopyable [concrete = constants.%Noncopyable] {} {}
// CHECK:STDOUT:   %AdaptNoncopyable.decl: type = class_decl @AdaptNoncopyable [concrete = constants.%AdaptNoncopyable] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.146 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.146 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.146 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.146 = return_slot_pattern %return.param_patt, %AdaptNoncopyable.ref.loc12_30 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc12_30: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %AdaptNoncopyable.ref.loc12_30 [concrete = constants.%.5ee]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyable = value_param call_param0
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc12_9: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %a: %AdaptNoncopyable = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %AdaptNoncopyable = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptNoncopyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNoncopyable {
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   adapt_decl %Noncopyable.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNoncopyable
// CHECK:STDOUT:   .Noncopyable = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %AdaptNoncopyable) -> out %return.param: %AdaptNoncopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.146 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.146 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyable = var %b.var_patt
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyable = name_ref a, %a
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %AdaptNoncopyable.ref.loc20: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:   %b: ref %AdaptNoncopyable = ref_binding b, %b.var
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyable = name_ref b, %b
// CHECK:STDOUT:   %.loc28: %AdaptNoncopyable = acquire_value %b.ref
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %b.var, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%b.var)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc20_3.2(%self.param: ref %AdaptNoncopyable) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_not_copyable_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptNoncopyableIndirect: type = class_type @AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.e36: %tuple.type.ff9 = tuple_value (%i32, %Noncopyable, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (%i32, %Noncopyable, %i32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f66: <witness> = complete_type_witness %tuple.type.985 [concrete]
// CHECK:STDOUT:   %pattern_type.c26: type = pattern_type %AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %.8e8: Core.Form = init_form %AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.6: type = fn_type @Destroy.Op.loc23_3.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.6: %Destroy.Op.type.af7ec0.6 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Noncopyable = %Noncopyable.decl
// CHECK:STDOUT:     .AdaptNoncopyableIndirect = %AdaptNoncopyableIndirect.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Noncopyable.decl: type = class_decl @Noncopyable [concrete = constants.%Noncopyable] {} {}
// CHECK:STDOUT:   %AdaptNoncopyableIndirect.decl: type = class_decl @AdaptNoncopyableIndirect [concrete = constants.%AdaptNoncopyableIndirect] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c26 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.c26 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c26 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c26 = return_slot_pattern %return.param_patt, %AdaptNoncopyableIndirect.ref.loc12_38 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc12_38: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %AdaptNoncopyableIndirect.ref.loc12_38 [concrete = constants.%.8e8]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyableIndirect = value_param call_param0
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc12_9: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %a: %AdaptNoncopyableIndirect = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %AdaptNoncopyableIndirect = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptNoncopyableIndirect = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNoncopyableIndirect {
// CHECK:STDOUT:   %i32.loc9_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %i32.loc9_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9_31: %tuple.type.ff9 = tuple_literal (%i32.loc9_10, %Noncopyable.ref, %i32.loc9_28) [concrete = constants.%tuple.e36]
// CHECK:STDOUT:   %.loc9_32: type = converted %.loc9_31, constants.%tuple.type.985 [concrete = constants.%tuple.type.985]
// CHECK:STDOUT:   adapt_decl %.loc9_32 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%tuple.type.985 [concrete = constants.%complete_type.f66]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNoncopyableIndirect
// CHECK:STDOUT:   .Noncopyable = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%a.param: %AdaptNoncopyableIndirect) -> out %return.param: %AdaptNoncopyableIndirect {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.c26 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.c26 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyableIndirect = var %b.var_patt
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyableIndirect = name_ref a, %a
// CHECK:STDOUT:   %.loc23_3.1: %tuple.type.985 = as_compatible %a.ref
// CHECK:STDOUT:   %tuple.elem0.loc23_3.1: %i32 = tuple_access %.loc23_3.1, element0
// CHECK:STDOUT:   %impl.elem0.loc23: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc23_3.1: <bound method> = bound_method %tuple.elem0.loc23_3.1, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_3.2: <bound method> = bound_method %tuple.elem0.loc23_3.1, %specific_fn.loc23
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc23: init %i32 = call %bound_method.loc23_3.2(%tuple.elem0.loc23_3.1)
// CHECK:STDOUT:   %.loc23_3.2: ref %tuple.type.985 = as_compatible %b.var
// CHECK:STDOUT:   %tuple.elem0.loc23_3.2: ref %i32 = tuple_access %.loc23_3.2, element0
// CHECK:STDOUT:   %.loc23_3.3: init %i32 to %tuple.elem0.loc23_3.2 = in_place_init %Int.as.Copy.impl.Op.call.loc23
// CHECK:STDOUT:   %tuple.elem1.loc23: %Noncopyable = tuple_access %.loc23_3.1, element1
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %AdaptNoncopyableIndirect.ref.loc23: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:   %b: ref %AdaptNoncopyableIndirect = ref_binding b, %b.var
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyableIndirect = name_ref b, %b
// CHECK:STDOUT:   %.loc34_11.1: ref %tuple.type.985 = as_compatible %b.ref
// CHECK:STDOUT:   %tuple.elem0.loc34_11.1: ref %i32 = tuple_access %.loc34_11.1, element0
// CHECK:STDOUT:   %.loc34_11.2: %i32 = acquire_value %tuple.elem0.loc34_11.1
// CHECK:STDOUT:   %impl.elem0.loc34: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc34_11.1: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34
// CHECK:STDOUT:   %specific_fn.loc34: <specific function> = specific_function %impl.elem0.loc34, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %specific_fn.loc34
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc34: init %i32 = call %bound_method.loc34_11.2(%.loc34_11.2)
// CHECK:STDOUT:   %.loc34_11.3: ref %tuple.type.985 = as_compatible %return.param
// CHECK:STDOUT:   %tuple.elem0.loc34_11.2: ref %i32 = tuple_access %.loc34_11.3, element0
// CHECK:STDOUT:   %.loc34_11.4: init %i32 to %tuple.elem0.loc34_11.2 = in_place_init %Int.as.Copy.impl.Op.call.loc34
// CHECK:STDOUT:   %tuple.elem1.loc34: ref %Noncopyable = tuple_access %.loc34_11.1, element1
// CHECK:STDOUT:   %.loc34_11.5: %Noncopyable = acquire_value %tuple.elem1.loc34
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %b.var, constants.%Destroy.Op.1dc86d.6
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%b.var)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.3(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.4(%self.param: ref %Noncopyable) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.5(%self.param: ref %tuple.type.985) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.6(%self.param: ref %AdaptNoncopyableIndirect) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptStruct: type = class_type @AdaptStruct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.8e4: <witness> = complete_type_witness %struct_type.e.f [concrete]
// CHECK:STDOUT:   %pattern_type.9fc: type = pattern_type %AdaptStruct [concrete]
// CHECK:STDOUT:   %.bb3: Core.Form = init_form %AdaptStruct [concrete]
// CHECK:STDOUT:   %I.type: type = fn_type @I [concrete]
// CHECK:STDOUT:   %I: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.66a: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet.66a [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.a59: %tuple.type.24b = tuple_value (%AdaptStruct, %u32) [concrete]
// CHECK:STDOUT:   %tuple.type.534: type = tuple_type (%AdaptStruct, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %tuple.type.534 [concrete]
// CHECK:STDOUT:   %.1415: Core.Form = init_form %tuple.type.534 [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32.builtin: type = int_type unsigned, %int_32 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.a58: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.dc5: %UInt.as.Copy.impl.Op.type.a58 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0f3: <witness> = impl_witness imports.%Copy.impl_witness_table.a92, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.3cf: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.6ad: %UInt.as.Copy.impl.Op.type.3cf = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.400: %Copy.type = facet_value %u32, (%Copy.impl_witness.0f3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.1c6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.400) [concrete]
// CHECK:STDOUT:   %.2e6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.1c6, %Copy.facet.400 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.6ad, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.7: type = fn_type @Destroy.Op.loc14_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.7: %Destroy.Op.type.af7ec0.7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.ca7: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.a58) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.dc5)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a92 = impl_witness_table (%Core.import_ref.ca7), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptStruct = %AdaptStruct.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .InTuple = %InTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptStruct.decl: type = class_decl @AdaptStruct [concrete = constants.%AdaptStruct] {} {}
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [concrete = constants.%I] {
// CHECK:STDOUT:     %g.param_patt: %pattern_type.9fc = value_param_pattern [concrete]
// CHECK:STDOUT:     %g.patt: %pattern_type.9fc = at_binding_pattern g, %g.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9fc = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.9fc = return_slot_pattern %return.param_patt, %AdaptStruct.ref.loc8_25 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc8_25: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %AdaptStruct.ref.loc8_25 [concrete = constants.%.bb3]
// CHECK:STDOUT:     %g.param: %AdaptStruct = value_param call_param0
// CHECK:STDOUT:     %AdaptStruct.ref.loc8_9: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %g: %AdaptStruct = value_binding g, %g.param
// CHECK:STDOUT:     %return.param: ref %AdaptStruct = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptStruct = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.8f9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.8f9 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8f9 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.8f9 = return_slot_pattern %return.param_patt, %.loc13_55.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc13_39: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %u32.loc13_52: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc13_55.1: %tuple.type.24b = tuple_literal (%AdaptStruct.ref.loc13_39, %u32.loc13_52) [concrete = constants.%tuple.a59]
// CHECK:STDOUT:     %.loc13_55.2: type = converted %.loc13_55.1, constants.%tuple.type.534 [concrete = constants.%tuple.type.534]
// CHECK:STDOUT:     %.loc13_55.3: Core.Form = init_form %.loc13_55.2 [concrete = constants.%.1415]
// CHECK:STDOUT:     %c.param: %tuple.type.534 = value_param call_param0
// CHECK:STDOUT:     %.loc13_32.1: type = splice_block %.loc13_32.3 [concrete = constants.%tuple.type.534] {
// CHECK:STDOUT:       %AdaptStruct.ref.loc13_16: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:       %u32.loc13_29: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc13_32.2: %tuple.type.24b = tuple_literal (%AdaptStruct.ref.loc13_16, %u32.loc13_29) [concrete = constants.%tuple.a59]
// CHECK:STDOUT:       %.loc13_32.3: type = converted %.loc13_32.2, constants.%tuple.type.534 [concrete = constants.%tuple.type.534]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.534 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.534 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.534 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptStruct {
// CHECK:STDOUT:   %i32.loc5_14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc5_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete = constants.%struct_type.e.f]
// CHECK:STDOUT:   adapt_decl %struct_type.e.f [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.e.f [concrete = constants.%complete_type.8e4]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptStruct
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I(%g.param: %AdaptStruct) -> out %return.param: %AdaptStruct {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.9fc = ref_binding_pattern h [concrete]
// CHECK:STDOUT:     %h.var_patt: %pattern_type.9fc = var_pattern %h.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %h.var: ref %AdaptStruct = var %h.var_patt
// CHECK:STDOUT:   %g.ref: %AdaptStruct = name_ref g, %g
// CHECK:STDOUT:   %.loc9_3.1: %struct_type.e.f = as_compatible %g.ref
// CHECK:STDOUT:   %.loc9_3.2: %i32 = struct_access %.loc9_3.1, element0
// CHECK:STDOUT:   %impl.elem0.loc9_3.1: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc9_3.1: <bound method> = bound_method %.loc9_3.2, %impl.elem0.loc9_3.1
// CHECK:STDOUT:   %specific_fn.loc9_3.1: <specific function> = specific_function %impl.elem0.loc9_3.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.2: <bound method> = bound_method %.loc9_3.2, %specific_fn.loc9_3.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.1: init %i32 = call %bound_method.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   %.loc9_3.3: ref %struct_type.e.f = as_compatible %h.var
// CHECK:STDOUT:   %.loc9_3.4: ref %i32 = struct_access %.loc9_3.3, element0
// CHECK:STDOUT:   %.loc9_3.5: init %i32 to %.loc9_3.4 = in_place_init %Int.as.Copy.impl.Op.call.loc9_3.1
// CHECK:STDOUT:   %.loc9_3.6: %i32 = struct_access %.loc9_3.1, element1
// CHECK:STDOUT:   %impl.elem0.loc9_3.2: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc9_3.3: <bound method> = bound_method %.loc9_3.6, %impl.elem0.loc9_3.2
// CHECK:STDOUT:   %specific_fn.loc9_3.2: <specific function> = specific_function %impl.elem0.loc9_3.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.4: <bound method> = bound_method %.loc9_3.6, %specific_fn.loc9_3.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.2: init %i32 = call %bound_method.loc9_3.4(%.loc9_3.6)
// CHECK:STDOUT:   %.loc9_3.7: ref %i32 = struct_access %.loc9_3.3, element1
// CHECK:STDOUT:   %.loc9_3.8: init %i32 to %.loc9_3.7 = in_place_init %Int.as.Copy.impl.Op.call.loc9_3.2
// CHECK:STDOUT:   %.loc9_3.9: init %struct_type.e.f to %.loc9_3.3 = struct_init (%.loc9_3.5, %.loc9_3.8)
// CHECK:STDOUT:   %.loc9_3.10: init %AdaptStruct = as_compatible %.loc9_3.9
// CHECK:STDOUT:   %.loc9_3.11: init %AdaptStruct = converted %g.ref, %.loc9_3.10
// CHECK:STDOUT:   assign %h.var, %.loc9_3.11
// CHECK:STDOUT:   %AdaptStruct.ref.loc9: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:   %h: ref %AdaptStruct = ref_binding h, %h.var
// CHECK:STDOUT:   %h.ref: ref %AdaptStruct = name_ref h, %h
// CHECK:STDOUT:   %.loc10_11.1: ref %struct_type.e.f = as_compatible %h.ref
// CHECK:STDOUT:   %.loc10_11.2: ref %i32 = struct_access %.loc10_11.1, element0
// CHECK:STDOUT:   %.loc10_11.3: %i32 = acquire_value %.loc10_11.2
// CHECK:STDOUT:   %impl.elem0.loc10_11.1: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.3, %impl.elem0.loc10_11.1
// CHECK:STDOUT:   %specific_fn.loc10_11.1: <specific function> = specific_function %impl.elem0.loc10_11.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.3, %specific_fn.loc10_11.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.1: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.3)
// CHECK:STDOUT:   %.loc10_11.4: ref %struct_type.e.f = as_compatible %return.param
// CHECK:STDOUT:   %.loc10_11.5: ref %i32 = struct_access %.loc10_11.4, element0
// CHECK:STDOUT:   %.loc10_11.6: init %i32 to %.loc10_11.5 = in_place_init %Int.as.Copy.impl.Op.call.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.7: ref %i32 = struct_access %.loc10_11.1, element1
// CHECK:STDOUT:   %.loc10_11.8: %i32 = acquire_value %.loc10_11.7
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.8, %impl.elem0.loc10_11.2
// CHECK:STDOUT:   %specific_fn.loc10_11.2: <specific function> = specific_function %impl.elem0.loc10_11.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.4: <bound method> = bound_method %.loc10_11.8, %specific_fn.loc10_11.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.2: init %i32 = call %bound_method.loc10_11.4(%.loc10_11.8)
// CHECK:STDOUT:   %.loc10_11.9: ref %i32 = struct_access %.loc10_11.4, element1
// CHECK:STDOUT:   %.loc10_11.10: init %i32 to %.loc10_11.9 = in_place_init %Int.as.Copy.impl.Op.call.loc10_11.2
// CHECK:STDOUT:   %.loc10_11.11: init %struct_type.e.f to %.loc10_11.4 = struct_init (%.loc10_11.6, %.loc10_11.10)
// CHECK:STDOUT:   %.loc10_11.12: init %AdaptStruct = as_compatible %.loc10_11.11
// CHECK:STDOUT:   %.loc10_11.13: init %AdaptStruct = converted %h.ref, %.loc10_11.12
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %h.var, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%h.var)
// CHECK:STDOUT:   return %.loc10_11.13 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %struct_type.e.f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.4(%self.param: ref %AdaptStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.534) -> out %return.param: %tuple.type.534 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.8f9 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.8f9 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %tuple.type.534 = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.534 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc14_31.1: %AdaptStruct = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_31.1: %struct_type.e.f = as_compatible %tuple.elem0.loc14_31.1
// CHECK:STDOUT:   %.loc14_31.2: %i32 = struct_access %.loc14_31.1, element0
// CHECK:STDOUT:   %impl.elem0.loc14_31.1: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc14_31.1: <bound method> = bound_method %.loc14_31.2, %impl.elem0.loc14_31.1
// CHECK:STDOUT:   %specific_fn.loc14_31.1: <specific function> = specific_function %impl.elem0.loc14_31.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_31.2: <bound method> = bound_method %.loc14_31.2, %specific_fn.loc14_31.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_31.1: init %i32 = call %bound_method.loc14_31.2(%.loc14_31.2)
// CHECK:STDOUT:   %tuple.elem0.loc14_31.2: ref %AdaptStruct = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc14_31.3: ref %struct_type.e.f = as_compatible %tuple.elem0.loc14_31.2
// CHECK:STDOUT:   %.loc14_31.4: ref %i32 = struct_access %.loc14_31.3, element0
// CHECK:STDOUT:   %.loc14_31.5: init %i32 to %.loc14_31.4 = in_place_init %Int.as.Copy.impl.Op.call.loc14_31.1
// CHECK:STDOUT:   %.loc14_31.6: %i32 = struct_access %.loc14_31.1, element1
// CHECK:STDOUT:   %impl.elem0.loc14_31.2: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc14_31.3: <bound method> = bound_method %.loc14_31.6, %impl.elem0.loc14_31.2
// CHECK:STDOUT:   %specific_fn.loc14_31.2: <specific function> = specific_function %impl.elem0.loc14_31.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_31.4: <bound method> = bound_method %.loc14_31.6, %specific_fn.loc14_31.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_31.2: init %i32 = call %bound_method.loc14_31.4(%.loc14_31.6)
// CHECK:STDOUT:   %.loc14_31.7: ref %i32 = struct_access %.loc14_31.3, element1
// CHECK:STDOUT:   %.loc14_31.8: init %i32 to %.loc14_31.7 = in_place_init %Int.as.Copy.impl.Op.call.loc14_31.2
// CHECK:STDOUT:   %.loc14_31.9: init %struct_type.e.f to %.loc14_31.3 = struct_init (%.loc14_31.5, %.loc14_31.8)
// CHECK:STDOUT:   %.loc14_31.10: init %AdaptStruct = as_compatible %.loc14_31.9
// CHECK:STDOUT:   %.loc14_31.11: init %AdaptStruct = converted %tuple.elem0.loc14_31.1, %.loc14_31.10
// CHECK:STDOUT:   %tuple.elem1.loc14_31.1: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %impl.elem0.loc14_31.3: %.2e6 = impl_witness_access constants.%Copy.impl_witness.0f3, element0 [concrete = constants.%UInt.as.Copy.impl.Op.6ad]
// CHECK:STDOUT:   %bound_method.loc14_31.5: <bound method> = bound_method %tuple.elem1.loc14_31.1, %impl.elem0.loc14_31.3
// CHECK:STDOUT:   %specific_fn.loc14_31.3: <specific function> = specific_function %impl.elem0.loc14_31.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_31.6: <bound method> = bound_method %tuple.elem1.loc14_31.1, %specific_fn.loc14_31.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc14: init %u32 = call %bound_method.loc14_31.6(%tuple.elem1.loc14_31.1)
// CHECK:STDOUT:   %tuple.elem1.loc14_31.2: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc14_31.12: init %u32 to %tuple.elem1.loc14_31.2 = in_place_init %UInt.as.Copy.impl.Op.call.loc14
// CHECK:STDOUT:   %.loc14_31.13: init %tuple.type.534 to %d.var = tuple_init (%.loc14_31.11, %.loc14_31.12)
// CHECK:STDOUT:   %.loc14_3: init %tuple.type.534 = converted %c.ref, %.loc14_31.13
// CHECK:STDOUT:   assign %d.var, %.loc14_3
// CHECK:STDOUT:   %.loc14_27.1: type = splice_block %.loc14_27.3 [concrete = constants.%tuple.type.534] {
// CHECK:STDOUT:     %AdaptStruct.ref.loc14: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %u32.loc14: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc14_27.2: %tuple.type.24b = tuple_literal (%AdaptStruct.ref.loc14, %u32.loc14) [concrete = constants.%tuple.a59]
// CHECK:STDOUT:     %.loc14_27.3: type = converted %.loc14_27.2, constants.%tuple.type.534 [concrete = constants.%tuple.type.534]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.534 = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %tuple.type.534 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc15_10.1: ref %AdaptStruct = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc15_10.1: ref %struct_type.e.f = as_compatible %tuple.elem0.loc15_10.1
// CHECK:STDOUT:   %.loc15_10.2: ref %i32 = struct_access %.loc15_10.1, element0
// CHECK:STDOUT:   %.loc15_10.3: %i32 = acquire_value %.loc15_10.2
// CHECK:STDOUT:   %impl.elem0.loc15_10.1: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc15_10.1: <bound method> = bound_method %.loc15_10.3, %impl.elem0.loc15_10.1
// CHECK:STDOUT:   %specific_fn.loc15_10.1: <specific function> = specific_function %impl.elem0.loc15_10.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.2: <bound method> = bound_method %.loc15_10.3, %specific_fn.loc15_10.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.1: init %i32 = call %bound_method.loc15_10.2(%.loc15_10.3)
// CHECK:STDOUT:   %tuple.elem0.loc15_10.2: ref %AdaptStruct = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc15_10.4: ref %struct_type.e.f = as_compatible %tuple.elem0.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.5: ref %i32 = struct_access %.loc15_10.4, element0
// CHECK:STDOUT:   %.loc15_10.6: init %i32 to %.loc15_10.5 = in_place_init %Int.as.Copy.impl.Op.call.loc15_10.1
// CHECK:STDOUT:   %.loc15_10.7: ref %i32 = struct_access %.loc15_10.1, element1
// CHECK:STDOUT:   %.loc15_10.8: %i32 = acquire_value %.loc15_10.7
// CHECK:STDOUT:   %impl.elem0.loc15_10.2: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc15_10.3: <bound method> = bound_method %.loc15_10.8, %impl.elem0.loc15_10.2
// CHECK:STDOUT:   %specific_fn.loc15_10.2: <specific function> = specific_function %impl.elem0.loc15_10.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.4: <bound method> = bound_method %.loc15_10.8, %specific_fn.loc15_10.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.2: init %i32 = call %bound_method.loc15_10.4(%.loc15_10.8)
// CHECK:STDOUT:   %.loc15_10.9: ref %i32 = struct_access %.loc15_10.4, element1
// CHECK:STDOUT:   %.loc15_10.10: init %i32 to %.loc15_10.9 = in_place_init %Int.as.Copy.impl.Op.call.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.11: init %struct_type.e.f to %.loc15_10.4 = struct_init (%.loc15_10.6, %.loc15_10.10)
// CHECK:STDOUT:   %.loc15_10.12: init %AdaptStruct = as_compatible %.loc15_10.11
// CHECK:STDOUT:   %.loc15_10.13: init %AdaptStruct = converted %tuple.elem0.loc15_10.1, %.loc15_10.12
// CHECK:STDOUT:   %tuple.elem1.loc15_10.1: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc15_10.14: %u32 = acquire_value %tuple.elem1.loc15_10.1
// CHECK:STDOUT:   %impl.elem0.loc15_10.3: %.2e6 = impl_witness_access constants.%Copy.impl_witness.0f3, element0 [concrete = constants.%UInt.as.Copy.impl.Op.6ad]
// CHECK:STDOUT:   %bound_method.loc15_10.5: <bound method> = bound_method %.loc15_10.14, %impl.elem0.loc15_10.3
// CHECK:STDOUT:   %specific_fn.loc15_10.3: <specific function> = specific_function %impl.elem0.loc15_10.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.6: <bound method> = bound_method %.loc15_10.14, %specific_fn.loc15_10.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc15: init %u32 = call %bound_method.loc15_10.6(%.loc15_10.14)
// CHECK:STDOUT:   %tuple.elem1.loc15_10.2: ref %u32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc15_10.15: init %u32 to %tuple.elem1.loc15_10.2 = in_place_init %UInt.as.Copy.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_10.16: init %tuple.type.534 to %return.param = tuple_init (%.loc15_10.13, %.loc15_10.15)
// CHECK:STDOUT:   %.loc15_11: init %tuple.type.534 = converted %d.ref, %.loc15_10.16
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.1dc86d.7
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return %.loc15_11 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.1(%self.param: ref %u32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.2(%self.param: ref %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.3(%self.param: ref %tuple.type.534) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
