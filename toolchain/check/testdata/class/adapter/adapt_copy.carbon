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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AdaptCopyable: type = class_type @AdaptCopyable [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.3b9: type = pattern_type %AdaptCopyable [concrete]
// CHECK:STDOUT:   %c.param_patt.a0c: %pattern_type.3b9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt.10d: %pattern_type.3b9 = wrapper_binding_pattern c, %c.param_patt.a0c [concrete]
// CHECK:STDOUT:   %.748: Core.Form = init_form %AdaptCopyable [concrete]
// CHECK:STDOUT:   %return.param_patt.c61: %pattern_type.3b9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.539: %pattern_type.3b9 = return_slot_pattern %return.param_patt.c61, %AdaptCopyable [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %d.patt.467: %pattern_type.3b9 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt.b13: %pattern_type.3b9 = var_pattern %d.patt.467 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc16_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc16_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.f36: %pattern_type.3b9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.205: %pattern_type.3b9 = wrapper_binding_pattern self, %self.param_patt.f36 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc16_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc16_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc16_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.13c: %tuple.type.693 = tuple_value (%AdaptCopyable, %u32) [concrete]
// CHECK:STDOUT:   %tuple.type.74d: type = tuple_type (%AdaptCopyable, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.0c4: type = pattern_type %tuple.type.74d [concrete]
// CHECK:STDOUT:   %c.param_patt.765: %pattern_type.0c4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt.aaf: %pattern_type.0c4 = wrapper_binding_pattern c, %c.param_patt.765 [concrete]
// CHECK:STDOUT:   %.24e: Core.Form = init_form %tuple.type.74d [concrete]
// CHECK:STDOUT:   %return.param_patt.040: %pattern_type.0c4 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.70d: %pattern_type.0c4 = return_slot_pattern %return.param_patt.040, %tuple.type.74d [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %d.patt.359: %pattern_type.0c4 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt.57a: %pattern_type.0c4 = var_pattern %d.patt.359 [concrete]
// CHECK:STDOUT:   %pattern_type.cf4: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %self.param_patt.f9a: %pattern_type.cf4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a22: %pattern_type.cf4 = wrapper_binding_pattern self, %self.param_patt.f9a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc35_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc35_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc35_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.001: %pattern_type.0c4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.97e: %pattern_type.0c4 = wrapper_binding_pattern self, %self.param_patt.001 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc35_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc35_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc35_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc16_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc16_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc16_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3b9 = ref_param_pattern [concrete = constants.%self.param_patt.f36]
// CHECK:STDOUT:     %self.patt: %pattern_type.3b9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.205]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptCopyable = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptCopyable = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc16_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3b9 = ref_param_pattern [concrete = constants.%self.param_patt.f36]
// CHECK:STDOUT:     %self.patt: %pattern_type.3b9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.205]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptCopyable = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptCopyable = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc35_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cf4 = ref_param_pattern [concrete = constants.%self.param_patt.f9a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cf4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc35_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cf4 = ref_param_pattern [concrete = constants.%self.param_patt.f9a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cf4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc35_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0c4 = ref_param_pattern [concrete = constants.%self.param_patt.001]
// CHECK:STDOUT:     %self.patt: %pattern_type.0c4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.97e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.74d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.74d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc35_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0c4 = ref_param_pattern [concrete = constants.%self.param_patt.001]
// CHECK:STDOUT:     %self.patt: %pattern_type.0c4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.97e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.74d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.74d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %c.param_patt: %pattern_type.3b9 = value_param_pattern [concrete = constants.%c.param_patt.a0c]
// CHECK:STDOUT:     %c.patt: %pattern_type.3b9 = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt.10d]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.3b9 = out_param_pattern [concrete = constants.%return.param_patt.c61]
// CHECK:STDOUT:     %return.patt: %pattern_type.3b9 = return_slot_pattern %return.param_patt, %AdaptCopyable.ref.loc8_27 [concrete = constants.%return.patt.539]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc8_27: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %AdaptCopyable.ref.loc8_27 [concrete = constants.%.748]
// CHECK:STDOUT:     %c.param: %AdaptCopyable = value_param call_param0
// CHECK:STDOUT:     %AdaptCopyable.ref.loc8_9: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %c: %AdaptCopyable = wrapper_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptCopyable = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptCopyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.0c4 = value_param_pattern [concrete = constants.%c.param_patt.765]
// CHECK:STDOUT:     %c.patt: %pattern_type.0c4 = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt.aaf]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0c4 = out_param_pattern [concrete = constants.%return.param_patt.040]
// CHECK:STDOUT:     %return.patt: %pattern_type.0c4 = return_slot_pattern %return.param_patt, %.loc27_59.2 [concrete = constants.%return.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc27_41: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %u32.loc27_56: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc27_59.1: %tuple.type.693 = tuple_literal (%AdaptCopyable.ref.loc27_41, %u32.loc27_56) [concrete = constants.%tuple.13c]
// CHECK:STDOUT:     %.loc27_59.2: type = converted %.loc27_59.1, constants.%tuple.type.74d [concrete = constants.%tuple.type.74d]
// CHECK:STDOUT:     %.loc27_59.3: Core.Form = init_form %.loc27_59.2 [concrete = constants.%.24e]
// CHECK:STDOUT:     %c.param: %tuple.type.74d = value_param call_param0
// CHECK:STDOUT:     %.loc27_34.1: type = splice_block %.loc27_34.3 [concrete = constants.%tuple.type.74d] {
// CHECK:STDOUT:       %AdaptCopyable.ref.loc27_16: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:       %u32.loc27_31: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc27_34.2: %tuple.type.693 = tuple_literal (%AdaptCopyable.ref.loc27_16, %u32.loc27_31) [concrete = constants.%tuple.13c]
// CHECK:STDOUT:       %.loc27_34.3: type = converted %.loc27_34.2, constants.%tuple.type.74d [concrete = constants.%tuple.type.74d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.74d = wrapper_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.74d = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.74d = return_slot %return.param
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
// CHECK:STDOUT:   %d.var: ref %AdaptCopyable = var_storage %d.var_patt
// CHECK:STDOUT:   %c.ref: %AdaptCopyable = name_ref c, %c
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   %AdaptCopyable.ref.loc16: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:   %d: ref %AdaptCopyable = wrapper_binding d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.3b9 = ref_binding_pattern d [concrete = constants.%d.patt.467]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.3b9 = var_pattern %d.patt [concrete = constants.%d.var_patt.b13]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.ref: ref %AdaptCopyable = name_ref d, %d
// CHECK:STDOUT:   %.loc24: %AdaptCopyable = acquire_value %d.ref
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %d.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%d.var)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc16_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc16_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc16_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc16_3.2(%self.param: ref %AdaptCopyable) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc16_3.2(%self.param: ref %AdaptCopyable) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc16_3.2(%self.param: ref %AdaptCopyable) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.74d) -> out %return.param: %tuple.type.74d {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %tuple.type.74d = var_storage %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.74d = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc35: %AdaptCopyable = tuple_access %c.ref, element0
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   %.loc35_29.1: type = splice_block %.loc35_29.3 [concrete = constants.%tuple.type.74d] {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc35: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %u32.loc35: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc35_29.2: %tuple.type.693 = tuple_literal (%AdaptCopyable.ref.loc35, %u32.loc35) [concrete = constants.%tuple.13c]
// CHECK:STDOUT:     %.loc35_29.3: type = converted %.loc35_29.2, constants.%tuple.type.74d [concrete = constants.%tuple.type.74d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.74d = wrapper_binding d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.0c4 = ref_binding_pattern d [concrete = constants.%d.patt.359]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.0c4 = var_pattern %d.patt [concrete = constants.%d.var_patt.57a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.ref: ref %tuple.type.74d = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc43: ref %AdaptCopyable = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc43: %AdaptCopyable = acquire_value %tuple.elem0.loc43
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %d.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%d.var)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc35_3.1(%self.param: ref %u32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc35_3.1(%self.param: ref %u32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc35_3.1(%self.param: ref %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc35_3.2(%self.param: ref %tuple.type.74d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %u32 = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc35_3.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc35_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc35_3.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %AdaptCopyable = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc35_3.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc35_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc35_3.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc35_3.2(%self.param: ref %tuple.type.74d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc35_3.2(%self.param: ref %tuple.type.74d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AdaptTuple: type = class_type @AdaptTuple [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.f51: %tuple.type.693 = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.259: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %complete_type.98d: <witness> = complete_type_witness %tuple.type.259 [concrete]
// CHECK:STDOUT:   %pattern_type.ea1: type = pattern_type %AdaptTuple [concrete]
// CHECK:STDOUT:   %c.param_patt.a12: %pattern_type.ea1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt.0c5: %pattern_type.ea1 = wrapper_binding_pattern c, %c.param_patt.a12 [concrete]
// CHECK:STDOUT:   %.6bf: Core.Form = init_form %AdaptTuple [concrete]
// CHECK:STDOUT:   %return.param_patt.511: %pattern_type.ea1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.ead: %pattern_type.ea1 = return_slot_pattern %return.param_patt.511, %AdaptTuple [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %d.patt.bc4: %pattern_type.ea1 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt.69e: %pattern_type.ea1 = var_pattern %d.patt.bc4 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.828: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.828) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet.828 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.b4f: type = pattern_type %tuple.type.259 [concrete]
// CHECK:STDOUT:   %self.param_patt.04f: %pattern_type.b4f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.03b: %pattern_type.b4f = wrapper_binding_pattern self, %self.param_patt.04f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.8a0: %pattern_type.ea1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.656: %pattern_type.ea1 = wrapper_binding_pattern self, %self.param_patt.8a0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.755: %tuple.type.693 = tuple_value (%AdaptTuple, %u32) [concrete]
// CHECK:STDOUT:   %tuple.type.3f9: type = tuple_type (%AdaptTuple, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.873: type = pattern_type %tuple.type.3f9 [concrete]
// CHECK:STDOUT:   %c.param_patt.4c3: %pattern_type.873 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt.510: %pattern_type.873 = wrapper_binding_pattern c, %c.param_patt.4c3 [concrete]
// CHECK:STDOUT:   %.649: Core.Form = init_form %tuple.type.3f9 [concrete]
// CHECK:STDOUT:   %return.param_patt.9d3: %pattern_type.873 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.8a0: %pattern_type.873 = return_slot_pattern %return.param_patt.9d3, %tuple.type.3f9 [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %d.patt.a41: %pattern_type.873 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt.720: %pattern_type.873 = var_pattern %d.patt.a41 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.1b4: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.795: %UInt.as.Copy.impl.Op.type.1b4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.338: <witness> = impl_witness imports.%Copy.impl_witness_table.af9, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.db5: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.462: %UInt.as.Copy.impl.Op.type.db5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c06: %Copy.type = facet_value %u32, (%Copy.impl_witness.338) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.059: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.c06) [concrete]
// CHECK:STDOUT:   %.355: type = fn_type_with_self_type %Copy.WithSelf.Op.type.059, %Copy.facet.c06 [concrete]
// CHECK:STDOUT:   %pattern_type.cf4: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.462, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %self.param_patt.f9a: %pattern_type.cf4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a22: %pattern_type.cf4 = wrapper_binding_pattern self, %self.param_patt.f9a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc14_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc14_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc14_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.2e6: %pattern_type.873 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e9b: %pattern_type.873 = wrapper_binding_pattern self, %self.param_patt.2e6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc14_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc14_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc14_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.0ed: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.1b4) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.795)]
// CHECK:STDOUT:   %Copy.impl_witness_table.af9 = impl_witness_table (%Core.import_ref.0ed), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b4f = ref_param_pattern [concrete = constants.%self.param_patt.04f]
// CHECK:STDOUT:     %self.patt: %pattern_type.b4f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.03b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.259 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.259 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b4f = ref_param_pattern [concrete = constants.%self.param_patt.04f]
// CHECK:STDOUT:     %self.patt: %pattern_type.b4f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.03b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.259 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.259 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ea1 = ref_param_pattern [concrete = constants.%self.param_patt.8a0]
// CHECK:STDOUT:     %self.patt: %pattern_type.ea1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.656]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptTuple = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptTuple = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ea1 = ref_param_pattern [concrete = constants.%self.param_patt.8a0]
// CHECK:STDOUT:     %self.patt: %pattern_type.ea1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.656]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptTuple = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptTuple = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc14_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cf4 = ref_param_pattern [concrete = constants.%self.param_patt.f9a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cf4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc14_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cf4 = ref_param_pattern [concrete = constants.%self.param_patt.f9a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cf4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc14_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.873 = ref_param_pattern [concrete = constants.%self.param_patt.2e6]
// CHECK:STDOUT:     %self.patt: %pattern_type.873 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e9b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.3f9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.3f9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc14_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.873 = ref_param_pattern [concrete = constants.%self.param_patt.2e6]
// CHECK:STDOUT:     %self.patt: %pattern_type.873 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e9b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.3f9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.3f9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %c.param_patt: %pattern_type.ea1 = value_param_pattern [concrete = constants.%c.param_patt.a12]
// CHECK:STDOUT:     %c.patt: %pattern_type.ea1 = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt.0c5]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ea1 = out_param_pattern [concrete = constants.%return.param_patt.511]
// CHECK:STDOUT:     %return.patt: %pattern_type.ea1 = return_slot_pattern %return.param_patt, %AdaptTuple.ref.loc8_24 [concrete = constants.%return.patt.ead]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc8_24: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %AdaptTuple.ref.loc8_24 [concrete = constants.%.6bf]
// CHECK:STDOUT:     %c.param: %AdaptTuple = value_param call_param0
// CHECK:STDOUT:     %AdaptTuple.ref.loc8_9: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %c: %AdaptTuple = wrapper_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptTuple = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptTuple = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.873 = value_param_pattern [concrete = constants.%c.param_patt.4c3]
// CHECK:STDOUT:     %c.patt: %pattern_type.873 = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt.510]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.873 = out_param_pattern [concrete = constants.%return.param_patt.9d3]
// CHECK:STDOUT:     %return.patt: %pattern_type.873 = return_slot_pattern %return.param_patt, %.loc13_53.2 [concrete = constants.%return.patt.8a0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc13_38: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %u32.loc13_50: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc13_53.1: %tuple.type.693 = tuple_literal (%AdaptTuple.ref.loc13_38, %u32.loc13_50) [concrete = constants.%tuple.755]
// CHECK:STDOUT:     %.loc13_53.2: type = converted %.loc13_53.1, constants.%tuple.type.3f9 [concrete = constants.%tuple.type.3f9]
// CHECK:STDOUT:     %.loc13_53.3: Core.Form = init_form %.loc13_53.2 [concrete = constants.%.649]
// CHECK:STDOUT:     %c.param: %tuple.type.3f9 = value_param call_param0
// CHECK:STDOUT:     %.loc13_31.1: type = splice_block %.loc13_31.3 [concrete = constants.%tuple.type.3f9] {
// CHECK:STDOUT:       %AdaptTuple.ref.loc13_16: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:       %u32.loc13_28: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc13_31.2: %tuple.type.693 = tuple_literal (%AdaptTuple.ref.loc13_16, %u32.loc13_28) [concrete = constants.%tuple.755]
// CHECK:STDOUT:       %.loc13_31.3: type = converted %.loc13_31.2, constants.%tuple.type.3f9 [concrete = constants.%tuple.type.3f9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.3f9 = wrapper_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.3f9 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.3f9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptTuple {
// CHECK:STDOUT:   %i32.loc5_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc5_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: %tuple.type.693 = tuple_literal (%i32.loc5_10, %i32.loc5_15) [concrete = constants.%tuple.f51]
// CHECK:STDOUT:   %.loc5_19: type = converted %.loc5_18, constants.%tuple.type.259 [concrete = constants.%tuple.type.259]
// CHECK:STDOUT:   adapt_decl %.loc5_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%tuple.type.259 [concrete = constants.%complete_type.98d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptTuple
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %AdaptTuple) -> out %return.param: %AdaptTuple {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %AdaptTuple = var_storage %d.var_patt
// CHECK:STDOUT:   %c.ref: %AdaptTuple = name_ref c, %c
// CHECK:STDOUT:   %.loc9_3.1: %tuple.type.259 = as_compatible %c.ref
// CHECK:STDOUT:   %tuple.elem0.loc9_3.1: %i32 = tuple_access %.loc9_3.1, element0
// CHECK:STDOUT:   %impl.elem0.loc9_3.1: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc9_3.1: <bound method> = bound_method %tuple.elem0.loc9_3.1, %impl.elem0.loc9_3.1
// CHECK:STDOUT:   %specific_fn.loc9_3.1: <specific function> = specific_function %impl.elem0.loc9_3.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.2: <bound method> = bound_method %tuple.elem0.loc9_3.1, %specific_fn.loc9_3.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.1: init %i32 = call %bound_method.loc9_3.2(%tuple.elem0.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %tuple.type.259 = as_compatible %d.var
// CHECK:STDOUT:   %tuple.elem0.loc9_3.2: ref %i32 = tuple_access %.loc9_3.2, element0
// CHECK:STDOUT:   %.loc9_3.3: init %i32 to %tuple.elem0.loc9_3.2 = in_place_init %Int.as.Copy.impl.Op.call.loc9_3.1
// CHECK:STDOUT:   %tuple.elem1.loc9_3.1: %i32 = tuple_access %.loc9_3.1, element1
// CHECK:STDOUT:   %impl.elem0.loc9_3.2: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc9_3.3: <bound method> = bound_method %tuple.elem1.loc9_3.1, %impl.elem0.loc9_3.2
// CHECK:STDOUT:   %specific_fn.loc9_3.2: <specific function> = specific_function %impl.elem0.loc9_3.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.4: <bound method> = bound_method %tuple.elem1.loc9_3.1, %specific_fn.loc9_3.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.2: init %i32 = call %bound_method.loc9_3.4(%tuple.elem1.loc9_3.1)
// CHECK:STDOUT:   %tuple.elem1.loc9_3.2: ref %i32 = tuple_access %.loc9_3.2, element1
// CHECK:STDOUT:   %.loc9_3.4: init %i32 to %tuple.elem1.loc9_3.2 = in_place_init %Int.as.Copy.impl.Op.call.loc9_3.2
// CHECK:STDOUT:   %.loc9_3.5: init %tuple.type.259 to %.loc9_3.2 = tuple_init (%.loc9_3.3, %.loc9_3.4)
// CHECK:STDOUT:   %.loc9_3.6: init %AdaptTuple = as_compatible %.loc9_3.5
// CHECK:STDOUT:   %.loc9_3.7: init %AdaptTuple = converted %c.ref, %.loc9_3.6
// CHECK:STDOUT:   assign %d.var, %.loc9_3.7
// CHECK:STDOUT:   %AdaptTuple.ref.loc9: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:   %d: ref %AdaptTuple = wrapper_binding d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.ea1 = ref_binding_pattern d [concrete = constants.%d.patt.bc4]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.ea1 = var_pattern %d.patt [concrete = constants.%d.var_patt.69e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.ref: ref %AdaptTuple = name_ref d, %d
// CHECK:STDOUT:   %.loc10_11.1: ref %tuple.type.259 = as_compatible %d.ref
// CHECK:STDOUT:   %tuple.elem0.loc10_11.1: ref %i32 = tuple_access %.loc10_11.1, element0
// CHECK:STDOUT:   %.loc10_11.2: %i32 = acquire_value %tuple.elem0.loc10_11.1
// CHECK:STDOUT:   %impl.elem0.loc10_11.1: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.2, %impl.elem0.loc10_11.1
// CHECK:STDOUT:   %specific_fn.loc10_11.1: <specific function> = specific_function %impl.elem0.loc10_11.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.2, %specific_fn.loc10_11.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.1: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.2)
// CHECK:STDOUT:   %.loc10_11.3: ref %tuple.type.259 = as_compatible %return.param
// CHECK:STDOUT:   %tuple.elem0.loc10_11.2: ref %i32 = tuple_access %.loc10_11.3, element0
// CHECK:STDOUT:   %.loc10_11.4: init %i32 to %tuple.elem0.loc10_11.2 = in_place_init %Int.as.Copy.impl.Op.call.loc10_11.1
// CHECK:STDOUT:   %tuple.elem1.loc10_11.1: ref %i32 = tuple_access %.loc10_11.1, element1
// CHECK:STDOUT:   %.loc10_11.5: %i32 = acquire_value %tuple.elem1.loc10_11.1
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.5, %impl.elem0.loc10_11.2
// CHECK:STDOUT:   %specific_fn.loc10_11.2: <specific function> = specific_function %impl.elem0.loc10_11.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.4: <bound method> = bound_method %.loc10_11.5, %specific_fn.loc10_11.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.2: init %i32 = call %bound_method.loc10_11.4(%.loc10_11.5)
// CHECK:STDOUT:   %tuple.elem1.loc10_11.2: ref %i32 = tuple_access %.loc10_11.3, element1
// CHECK:STDOUT:   %.loc10_11.6: init %i32 to %tuple.elem1.loc10_11.2 = in_place_init %Int.as.Copy.impl.Op.call.loc10_11.2
// CHECK:STDOUT:   %.loc10_11.7: init %tuple.type.259 to %.loc10_11.3 = tuple_init (%.loc10_11.4, %.loc10_11.6)
// CHECK:STDOUT:   %.loc10_11.8: init %AdaptTuple = as_compatible %.loc10_11.7
// CHECK:STDOUT:   %.loc10_11.9: init %AdaptTuple = converted %d.ref, %.loc10_11.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %d.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%d.var)
// CHECK:STDOUT:   return %.loc10_11.9 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %tuple.type.259) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %tuple.type.259) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %tuple.type.259) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.3(%self.param: ref %AdaptTuple) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.3(%self.param: ref %AdaptTuple) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.3(%self.param: ref %AdaptTuple) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.3f9) -> out %return.param: %tuple.type.3f9 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %tuple.type.3f9 = var_storage %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.3f9 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc14_30.1: %AdaptTuple = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_30.1: %tuple.type.259 = as_compatible %tuple.elem0.loc14_30.1
// CHECK:STDOUT:   %tuple.elem0.loc14_30.2: %i32 = tuple_access %.loc14_30.1, element0
// CHECK:STDOUT:   %impl.elem0.loc14_30.1: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc14_30.1: <bound method> = bound_method %tuple.elem0.loc14_30.2, %impl.elem0.loc14_30.1
// CHECK:STDOUT:   %specific_fn.loc14_30.1: <specific function> = specific_function %impl.elem0.loc14_30.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_30.2: <bound method> = bound_method %tuple.elem0.loc14_30.2, %specific_fn.loc14_30.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_30.1: init %i32 = call %bound_method.loc14_30.2(%tuple.elem0.loc14_30.2)
// CHECK:STDOUT:   %tuple.elem0.loc14_30.3: ref %AdaptTuple = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc14_30.2: ref %tuple.type.259 = as_compatible %tuple.elem0.loc14_30.3
// CHECK:STDOUT:   %tuple.elem0.loc14_30.4: ref %i32 = tuple_access %.loc14_30.2, element0
// CHECK:STDOUT:   %.loc14_30.3: init %i32 to %tuple.elem0.loc14_30.4 = in_place_init %Int.as.Copy.impl.Op.call.loc14_30.1
// CHECK:STDOUT:   %tuple.elem1.loc14_30.1: %i32 = tuple_access %.loc14_30.1, element1
// CHECK:STDOUT:   %impl.elem0.loc14_30.2: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc14_30.3: <bound method> = bound_method %tuple.elem1.loc14_30.1, %impl.elem0.loc14_30.2
// CHECK:STDOUT:   %specific_fn.loc14_30.2: <specific function> = specific_function %impl.elem0.loc14_30.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_30.4: <bound method> = bound_method %tuple.elem1.loc14_30.1, %specific_fn.loc14_30.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_30.2: init %i32 = call %bound_method.loc14_30.4(%tuple.elem1.loc14_30.1)
// CHECK:STDOUT:   %tuple.elem1.loc14_30.2: ref %i32 = tuple_access %.loc14_30.2, element1
// CHECK:STDOUT:   %.loc14_30.4: init %i32 to %tuple.elem1.loc14_30.2 = in_place_init %Int.as.Copy.impl.Op.call.loc14_30.2
// CHECK:STDOUT:   %.loc14_30.5: init %tuple.type.259 to %.loc14_30.2 = tuple_init (%.loc14_30.3, %.loc14_30.4)
// CHECK:STDOUT:   %.loc14_30.6: init %AdaptTuple = as_compatible %.loc14_30.5
// CHECK:STDOUT:   %.loc14_30.7: init %AdaptTuple = converted %tuple.elem0.loc14_30.1, %.loc14_30.6
// CHECK:STDOUT:   %tuple.elem1.loc14_30.3: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %impl.elem0.loc14_30.3: %.355 = impl_witness_access constants.%Copy.impl_witness.338, element0 [concrete = constants.%UInt.as.Copy.impl.Op.462]
// CHECK:STDOUT:   %bound_method.loc14_30.5: <bound method> = bound_method %tuple.elem1.loc14_30.3, %impl.elem0.loc14_30.3
// CHECK:STDOUT:   %specific_fn.loc14_30.3: <specific function> = specific_function %impl.elem0.loc14_30.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_30.6: <bound method> = bound_method %tuple.elem1.loc14_30.3, %specific_fn.loc14_30.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc14: init %u32 = call %bound_method.loc14_30.6(%tuple.elem1.loc14_30.3)
// CHECK:STDOUT:   %tuple.elem1.loc14_30.4: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc14_30.8: init %u32 to %tuple.elem1.loc14_30.4 = in_place_init %UInt.as.Copy.impl.Op.call.loc14
// CHECK:STDOUT:   %.loc14_30.9: init %tuple.type.3f9 to %d.var = tuple_init (%.loc14_30.7, %.loc14_30.8)
// CHECK:STDOUT:   %.loc14_3: init %tuple.type.3f9 = converted %c.ref, %.loc14_30.9
// CHECK:STDOUT:   assign %d.var, %.loc14_3
// CHECK:STDOUT:   %.loc14_26.1: type = splice_block %.loc14_26.3 [concrete = constants.%tuple.type.3f9] {
// CHECK:STDOUT:     %AdaptTuple.ref.loc14: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %u32.loc14: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc14_26.2: %tuple.type.693 = tuple_literal (%AdaptTuple.ref.loc14, %u32.loc14) [concrete = constants.%tuple.755]
// CHECK:STDOUT:     %.loc14_26.3: type = converted %.loc14_26.2, constants.%tuple.type.3f9 [concrete = constants.%tuple.type.3f9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.3f9 = wrapper_binding d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.873 = ref_binding_pattern d [concrete = constants.%d.patt.a41]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.873 = var_pattern %d.patt [concrete = constants.%d.var_patt.720]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.ref: ref %tuple.type.3f9 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc15_10.1: ref %AdaptTuple = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc15_10.1: ref %tuple.type.259 = as_compatible %tuple.elem0.loc15_10.1
// CHECK:STDOUT:   %tuple.elem0.loc15_10.2: ref %i32 = tuple_access %.loc15_10.1, element0
// CHECK:STDOUT:   %.loc15_10.2: %i32 = acquire_value %tuple.elem0.loc15_10.2
// CHECK:STDOUT:   %impl.elem0.loc15_10.1: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc15_10.1: <bound method> = bound_method %.loc15_10.2, %impl.elem0.loc15_10.1
// CHECK:STDOUT:   %specific_fn.loc15_10.1: <specific function> = specific_function %impl.elem0.loc15_10.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.2: <bound method> = bound_method %.loc15_10.2, %specific_fn.loc15_10.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.1: init %i32 = call %bound_method.loc15_10.2(%.loc15_10.2)
// CHECK:STDOUT:   %tuple.elem0.loc15_10.3: ref %AdaptTuple = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc15_10.3: ref %tuple.type.259 = as_compatible %tuple.elem0.loc15_10.3
// CHECK:STDOUT:   %tuple.elem0.loc15_10.4: ref %i32 = tuple_access %.loc15_10.3, element0
// CHECK:STDOUT:   %.loc15_10.4: init %i32 to %tuple.elem0.loc15_10.4 = in_place_init %Int.as.Copy.impl.Op.call.loc15_10.1
// CHECK:STDOUT:   %tuple.elem1.loc15_10.1: ref %i32 = tuple_access %.loc15_10.1, element1
// CHECK:STDOUT:   %.loc15_10.5: %i32 = acquire_value %tuple.elem1.loc15_10.1
// CHECK:STDOUT:   %impl.elem0.loc15_10.2: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc15_10.3: <bound method> = bound_method %.loc15_10.5, %impl.elem0.loc15_10.2
// CHECK:STDOUT:   %specific_fn.loc15_10.2: <specific function> = specific_function %impl.elem0.loc15_10.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.4: <bound method> = bound_method %.loc15_10.5, %specific_fn.loc15_10.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.2: init %i32 = call %bound_method.loc15_10.4(%.loc15_10.5)
// CHECK:STDOUT:   %tuple.elem1.loc15_10.2: ref %i32 = tuple_access %.loc15_10.3, element1
// CHECK:STDOUT:   %.loc15_10.6: init %i32 to %tuple.elem1.loc15_10.2 = in_place_init %Int.as.Copy.impl.Op.call.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.7: init %tuple.type.259 to %.loc15_10.3 = tuple_init (%.loc15_10.4, %.loc15_10.6)
// CHECK:STDOUT:   %.loc15_10.8: init %AdaptTuple = as_compatible %.loc15_10.7
// CHECK:STDOUT:   %.loc15_10.9: init %AdaptTuple = converted %tuple.elem0.loc15_10.1, %.loc15_10.8
// CHECK:STDOUT:   %tuple.elem1.loc15_10.3: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc15_10.10: %u32 = acquire_value %tuple.elem1.loc15_10.3
// CHECK:STDOUT:   %impl.elem0.loc15_10.3: %.355 = impl_witness_access constants.%Copy.impl_witness.338, element0 [concrete = constants.%UInt.as.Copy.impl.Op.462]
// CHECK:STDOUT:   %bound_method.loc15_10.5: <bound method> = bound_method %.loc15_10.10, %impl.elem0.loc15_10.3
// CHECK:STDOUT:   %specific_fn.loc15_10.3: <specific function> = specific_function %impl.elem0.loc15_10.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.6: <bound method> = bound_method %.loc15_10.10, %specific_fn.loc15_10.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc15: init %u32 = call %bound_method.loc15_10.6(%.loc15_10.10)
// CHECK:STDOUT:   %tuple.elem1.loc15_10.4: ref %u32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc15_10.11: init %u32 to %tuple.elem1.loc15_10.4 = in_place_init %UInt.as.Copy.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_10.12: init %tuple.type.3f9 to %return.param = tuple_init (%.loc15_10.9, %.loc15_10.11)
// CHECK:STDOUT:   %.loc15_11: init %tuple.type.3f9 = converted %d.ref, %.loc15_10.12
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %d.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%d.var)
// CHECK:STDOUT:   return %.loc15_11 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc14_3.1(%self.param: ref %u32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc14_3.1(%self.param: ref %u32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc14_3.1(%self.param: ref %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc14_3.2(%self.param: ref %tuple.type.3f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %u32 = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc14_3.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc14_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc14_3.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %AdaptTuple = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc14_3.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc14_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc14_3.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc14_3.2(%self.param: ref %tuple.type.3f9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc14_3.2(%self.param: ref %tuple.type.3f9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_not_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptNoncopyable: type = class_type @AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.d14: type = pattern_type %AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.d14 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d14 = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %.a97: Core.Form = init_form %AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %return.param_patt.f81: %pattern_type.d14 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.1dc: %pattern_type.d14 = return_slot_pattern %return.param_patt.f81, %AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d14 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.d14 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc20_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.944: type = pattern_type %Noncopyable [concrete]
// CHECK:STDOUT:   %self.param_patt.648: %pattern_type.944 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e5a: %pattern_type.944 = wrapper_binding_pattern self, %self.param_patt.648 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc20_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.f20: %pattern_type.d14 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.b09: %pattern_type.d14 = wrapper_binding_pattern self, %self.param_patt.f20 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc20_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc20_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc20_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.944 = ref_param_pattern [concrete = constants.%self.param_patt.648]
// CHECK:STDOUT:     %self.patt: %pattern_type.944 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e5a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Noncopyable = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Noncopyable = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc20_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.944 = ref_param_pattern [concrete = constants.%self.param_patt.648]
// CHECK:STDOUT:     %self.patt: %pattern_type.944 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e5a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Noncopyable = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Noncopyable = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d14 = ref_param_pattern [concrete = constants.%self.param_patt.f20]
// CHECK:STDOUT:     %self.patt: %pattern_type.d14 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b09]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptNoncopyable = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptNoncopyable = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc20_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d14 = ref_param_pattern [concrete = constants.%self.param_patt.f20]
// CHECK:STDOUT:     %self.patt: %pattern_type.d14 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b09]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptNoncopyable = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptNoncopyable = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.d14 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.d14 = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d14 = out_param_pattern [concrete = constants.%return.param_patt.f81]
// CHECK:STDOUT:     %return.patt: %pattern_type.d14 = return_slot_pattern %return.param_patt, %AdaptNoncopyable.ref.loc12_30 [concrete = constants.%return.patt.1dc]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc12_30: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %AdaptNoncopyable.ref.loc12_30 [concrete = constants.%.a97]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyable = value_param call_param0
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc12_9: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %a: %AdaptNoncopyable = wrapper_binding a, %a.param
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
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyable = var_storage %b.var_patt
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyable = name_ref a, %a
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %AdaptNoncopyable.ref.loc20: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:   %b: ref %AdaptNoncopyable = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d14 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.d14 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyable = name_ref b, %b
// CHECK:STDOUT:   %.loc28: %AdaptNoncopyable = acquire_value %b.ref
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%b.var)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_3.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_3.2(%self.param: ref %Noncopyable) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_3.2(%self.param: ref %Noncopyable) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_3.2(%self.param: ref %Noncopyable) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_3.3(%self.param: ref %AdaptNoncopyable) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_3.3(%self.param: ref %AdaptNoncopyable) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_3.3(%self.param: ref %AdaptNoncopyable) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_not_copyable_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %tuple.type.531: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.465: %tuple.type.531 = tuple_value (%i32, %Noncopyable, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.928: type = tuple_type (%i32, %Noncopyable, %i32) [concrete]
// CHECK:STDOUT:   %complete_type.3bc: <witness> = complete_type_witness %tuple.type.928 [concrete]
// CHECK:STDOUT:   %pattern_type.be3: type = pattern_type %AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.be3 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.be3 = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %.284: Core.Form = init_form %AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %return.param_patt.6c3: %pattern_type.be3 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.de6: %pattern_type.be3 = return_slot_pattern %return.param_patt.6c3, %AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.be3 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.be3 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc23_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc23_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc23_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc23_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc23_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.944: type = pattern_type %Noncopyable [concrete]
// CHECK:STDOUT:   %self.param_patt.648: %pattern_type.944 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e5a: %pattern_type.944 = wrapper_binding_pattern self, %self.param_patt.648 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc23_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc23_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc23_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.be2: type = pattern_type %tuple.type.928 [concrete]
// CHECK:STDOUT:   %self.param_patt.118: %pattern_type.be2 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4ea: %pattern_type.be2 = wrapper_binding_pattern self, %self.param_patt.118 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc23_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc23_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.f8a: %pattern_type.be3 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.021: %pattern_type.be3 = wrapper_binding_pattern self, %self.param_patt.f8a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc23_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc23_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc23_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc23_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc23_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc23_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc23_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc23_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.944 = ref_param_pattern [concrete = constants.%self.param_patt.648]
// CHECK:STDOUT:     %self.patt: %pattern_type.944 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e5a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Noncopyable = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Noncopyable = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc23_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.944 = ref_param_pattern [concrete = constants.%self.param_patt.648]
// CHECK:STDOUT:     %self.patt: %pattern_type.944 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e5a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Noncopyable = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Noncopyable = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc23_3.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.be2 = ref_param_pattern [concrete = constants.%self.param_patt.118]
// CHECK:STDOUT:     %self.patt: %pattern_type.be2 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.928 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.928 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc23_3.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.be2 = ref_param_pattern [concrete = constants.%self.param_patt.118]
// CHECK:STDOUT:     %self.patt: %pattern_type.be2 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.928 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.928 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc23_3.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.be3 = ref_param_pattern [concrete = constants.%self.param_patt.f8a]
// CHECK:STDOUT:     %self.patt: %pattern_type.be3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.021]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptNoncopyableIndirect = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptNoncopyableIndirect = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc23_3.5 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.be3 = ref_param_pattern [concrete = constants.%self.param_patt.f8a]
// CHECK:STDOUT:     %self.patt: %pattern_type.be3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.021]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptNoncopyableIndirect = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptNoncopyableIndirect = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.be3 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.be3 = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.be3 = out_param_pattern [concrete = constants.%return.param_patt.6c3]
// CHECK:STDOUT:     %return.patt: %pattern_type.be3 = return_slot_pattern %return.param_patt, %AdaptNoncopyableIndirect.ref.loc12_38 [concrete = constants.%return.patt.de6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc12_38: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %AdaptNoncopyableIndirect.ref.loc12_38 [concrete = constants.%.284]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyableIndirect = value_param call_param0
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc12_9: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %a: %AdaptNoncopyableIndirect = wrapper_binding a, %a.param
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
// CHECK:STDOUT:   %.loc9_31: %tuple.type.531 = tuple_literal (%i32.loc9_10, %Noncopyable.ref, %i32.loc9_28) [concrete = constants.%tuple.465]
// CHECK:STDOUT:   %.loc9_32: type = converted %.loc9_31, constants.%tuple.type.928 [concrete = constants.%tuple.type.928]
// CHECK:STDOUT:   adapt_decl %.loc9_32 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%tuple.type.928 [concrete = constants.%complete_type.3bc]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNoncopyableIndirect
// CHECK:STDOUT:   .Noncopyable = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%a.param: %AdaptNoncopyableIndirect) -> out %return.param: %AdaptNoncopyableIndirect {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyableIndirect = var_storage %b.var_patt
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyableIndirect = name_ref a, %a
// CHECK:STDOUT:   %.loc23_3.1: %tuple.type.928 = as_compatible %a.ref
// CHECK:STDOUT:   %tuple.elem0.loc23_3.1: %i32 = tuple_access %.loc23_3.1, element0
// CHECK:STDOUT:   %impl.elem0.loc23: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc23_3.1: <bound method> = bound_method %tuple.elem0.loc23_3.1, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_3.2: <bound method> = bound_method %tuple.elem0.loc23_3.1, %specific_fn.loc23
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc23: init %i32 = call %bound_method.loc23_3.2(%tuple.elem0.loc23_3.1)
// CHECK:STDOUT:   %.loc23_3.2: ref %tuple.type.928 = as_compatible %b.var
// CHECK:STDOUT:   %tuple.elem0.loc23_3.2: ref %i32 = tuple_access %.loc23_3.2, element0
// CHECK:STDOUT:   %.loc23_3.3: init %i32 to %tuple.elem0.loc23_3.2 = in_place_init %Int.as.Copy.impl.Op.call.loc23
// CHECK:STDOUT:   %tuple.elem1.loc23: %Noncopyable = tuple_access %.loc23_3.1, element1
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %AdaptNoncopyableIndirect.ref.loc23: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:   %b: ref %AdaptNoncopyableIndirect = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.be3 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.be3 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyableIndirect = name_ref b, %b
// CHECK:STDOUT:   %.loc34_11.1: ref %tuple.type.928 = as_compatible %b.ref
// CHECK:STDOUT:   %tuple.elem0.loc34_11.1: ref %i32 = tuple_access %.loc34_11.1, element0
// CHECK:STDOUT:   %.loc34_11.2: %i32 = acquire_value %tuple.elem0.loc34_11.1
// CHECK:STDOUT:   %impl.elem0.loc34: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc34_11.1: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34
// CHECK:STDOUT:   %specific_fn.loc34: <specific function> = specific_function %impl.elem0.loc34, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %specific_fn.loc34
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc34: init %i32 = call %bound_method.loc34_11.2(%.loc34_11.2)
// CHECK:STDOUT:   %.loc34_11.3: ref %tuple.type.928 = as_compatible %return.param
// CHECK:STDOUT:   %tuple.elem0.loc34_11.2: ref %i32 = tuple_access %.loc34_11.3, element0
// CHECK:STDOUT:   %.loc34_11.4: init %i32 to %tuple.elem0.loc34_11.2 = in_place_init %Int.as.Copy.impl.Op.call.loc34
// CHECK:STDOUT:   %tuple.elem1.loc34: ref %Noncopyable = tuple_access %.loc34_11.1, element1
// CHECK:STDOUT:   %.loc34_11.5: %Noncopyable = acquire_value %tuple.elem1.loc34
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%b.var)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc23_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc23_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc23_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc23_3.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc23_3.2(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc23_3.2(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc23_3.3(%self.param: ref %Noncopyable) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc23_3.3(%self.param: ref %Noncopyable) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc23_3.3(%self.param: ref %Noncopyable) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc23_3.4(%self.param: ref %tuple.type.928) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc23_3.1: <bound method> = bound_method %tuple.elem2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc23_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc23_3.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %Noncopyable = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc23_3.2: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc23_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc23_3.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc23_3.3: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc23_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc23_3.3(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc23_3.4(%self.param: ref %tuple.type.928) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc23_3.4(%self.param: ref %tuple.type.928) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc23_3.5(%self.param: ref %AdaptNoncopyableIndirect) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc23_3.5(%self.param: ref %AdaptNoncopyableIndirect) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc23_3.5(%self.param: ref %AdaptNoncopyableIndirect) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %AdaptStruct: type = class_type @AdaptStruct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.bf4: <witness> = complete_type_witness %struct_type.e.f [concrete]
// CHECK:STDOUT:   %pattern_type.27f: type = pattern_type %AdaptStruct [concrete]
// CHECK:STDOUT:   %g.param_patt: %pattern_type.27f = value_param_pattern [concrete]
// CHECK:STDOUT:   %g.patt: %pattern_type.27f = wrapper_binding_pattern g, %g.param_patt [concrete]
// CHECK:STDOUT:   %.ae3: Core.Form = init_form %AdaptStruct [concrete]
// CHECK:STDOUT:   %return.param_patt.427: %pattern_type.27f = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.0e8: %pattern_type.27f = return_slot_pattern %return.param_patt.427, %AdaptStruct [concrete]
// CHECK:STDOUT:   %I.type: type = fn_type @I [concrete]
// CHECK:STDOUT:   %I: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %h.patt: %pattern_type.27f = ref_binding_pattern h [concrete]
// CHECK:STDOUT:   %h.var_patt: %pattern_type.27f = var_pattern %h.patt [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.828: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.828) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet.828 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.8f5: type = pattern_type %struct_type.e.f [concrete]
// CHECK:STDOUT:   %self.param_patt.9af: %pattern_type.8f5 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.339: %pattern_type.8f5 = wrapper_binding_pattern self, %self.param_patt.9af [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.58e: %pattern_type.27f = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.682: %pattern_type.27f = wrapper_binding_pattern self, %self.param_patt.58e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.199: %tuple.type.693 = tuple_value (%AdaptStruct, %u32) [concrete]
// CHECK:STDOUT:   %tuple.type.785: type = tuple_type (%AdaptStruct, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.6f0: type = pattern_type %tuple.type.785 [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.6f0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.6f0 = wrapper_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %.078: Core.Form = init_form %tuple.type.785 [concrete]
// CHECK:STDOUT:   %return.param_patt.f00: %pattern_type.6f0 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.605: %pattern_type.6f0 = return_slot_pattern %return.param_patt.f00, %tuple.type.785 [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.6f0 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt: %pattern_type.6f0 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.1b4: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.795: %UInt.as.Copy.impl.Op.type.1b4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.338: <witness> = impl_witness imports.%Copy.impl_witness_table.af9, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.db5: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.462: %UInt.as.Copy.impl.Op.type.db5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c06: %Copy.type = facet_value %u32, (%Copy.impl_witness.338) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.059: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.c06) [concrete]
// CHECK:STDOUT:   %.355: type = fn_type_with_self_type %Copy.WithSelf.Op.type.059, %Copy.facet.c06 [concrete]
// CHECK:STDOUT:   %pattern_type.cf4: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.462, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %self.param_patt.f9a: %pattern_type.cf4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a22: %pattern_type.cf4 = wrapper_binding_pattern self, %self.param_patt.f9a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc14_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc14_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc14_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.7fb: %pattern_type.6f0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e65: %pattern_type.6f0 = wrapper_binding_pattern self, %self.param_patt.7fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc14_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc14_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc14_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.0ed: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.1b4) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.795)]
// CHECK:STDOUT:   %Copy.impl_witness_table.af9 = impl_witness_table (%Core.import_ref.0ed), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8f5 = ref_param_pattern [concrete = constants.%self.param_patt.9af]
// CHECK:STDOUT:     %self.patt: %pattern_type.8f5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.339]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.e.f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.e.f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8f5 = ref_param_pattern [concrete = constants.%self.param_patt.9af]
// CHECK:STDOUT:     %self.patt: %pattern_type.8f5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.339]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.e.f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.e.f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.27f = ref_param_pattern [concrete = constants.%self.param_patt.58e]
// CHECK:STDOUT:     %self.patt: %pattern_type.27f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.682]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptStruct = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptStruct = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.27f = ref_param_pattern [concrete = constants.%self.param_patt.58e]
// CHECK:STDOUT:     %self.patt: %pattern_type.27f = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.682]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %AdaptStruct = ref_param call_param0
// CHECK:STDOUT:     %self: ref %AdaptStruct = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc14_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cf4 = ref_param_pattern [concrete = constants.%self.param_patt.f9a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cf4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc14_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cf4 = ref_param_pattern [concrete = constants.%self.param_patt.f9a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cf4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc14_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6f0 = ref_param_pattern [concrete = constants.%self.param_patt.7fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6f0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e65]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.785 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.785 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc14_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6f0 = ref_param_pattern [concrete = constants.%self.param_patt.7fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.6f0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e65]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.785 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.785 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %g.param_patt: %pattern_type.27f = value_param_pattern [concrete = constants.%g.param_patt]
// CHECK:STDOUT:     %g.patt: %pattern_type.27f = wrapper_binding_pattern g, %g.param_patt [concrete = constants.%g.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.27f = out_param_pattern [concrete = constants.%return.param_patt.427]
// CHECK:STDOUT:     %return.patt: %pattern_type.27f = return_slot_pattern %return.param_patt, %AdaptStruct.ref.loc8_25 [concrete = constants.%return.patt.0e8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc8_25: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %AdaptStruct.ref.loc8_25 [concrete = constants.%.ae3]
// CHECK:STDOUT:     %g.param: %AdaptStruct = value_param call_param0
// CHECK:STDOUT:     %AdaptStruct.ref.loc8_9: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %g: %AdaptStruct = wrapper_binding g, %g.param
// CHECK:STDOUT:     %return.param: ref %AdaptStruct = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptStruct = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.6f0 = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.6f0 = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6f0 = out_param_pattern [concrete = constants.%return.param_patt.f00]
// CHECK:STDOUT:     %return.patt: %pattern_type.6f0 = return_slot_pattern %return.param_patt, %.loc13_55.2 [concrete = constants.%return.patt.605]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc13_39: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %u32.loc13_52: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc13_55.1: %tuple.type.693 = tuple_literal (%AdaptStruct.ref.loc13_39, %u32.loc13_52) [concrete = constants.%tuple.199]
// CHECK:STDOUT:     %.loc13_55.2: type = converted %.loc13_55.1, constants.%tuple.type.785 [concrete = constants.%tuple.type.785]
// CHECK:STDOUT:     %.loc13_55.3: Core.Form = init_form %.loc13_55.2 [concrete = constants.%.078]
// CHECK:STDOUT:     %c.param: %tuple.type.785 = value_param call_param0
// CHECK:STDOUT:     %.loc13_32.1: type = splice_block %.loc13_32.3 [concrete = constants.%tuple.type.785] {
// CHECK:STDOUT:       %AdaptStruct.ref.loc13_16: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:       %u32.loc13_29: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc13_32.2: %tuple.type.693 = tuple_literal (%AdaptStruct.ref.loc13_16, %u32.loc13_29) [concrete = constants.%tuple.199]
// CHECK:STDOUT:       %.loc13_32.3: type = converted %.loc13_32.2, constants.%tuple.type.785 [concrete = constants.%tuple.type.785]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.785 = wrapper_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.785 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.785 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptStruct {
// CHECK:STDOUT:   %i32.loc5_14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc5_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete = constants.%struct_type.e.f]
// CHECK:STDOUT:   adapt_decl %struct_type.e.f [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.e.f [concrete = constants.%complete_type.bf4]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptStruct
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I(%g.param: %AdaptStruct) -> out %return.param: %AdaptStruct {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %h.var: ref %AdaptStruct = var_storage %h.var_patt
// CHECK:STDOUT:   %g.ref: %AdaptStruct = name_ref g, %g
// CHECK:STDOUT:   %.loc9_3.1: %struct_type.e.f = as_compatible %g.ref
// CHECK:STDOUT:   %.loc9_3.2: %i32 = struct_access %.loc9_3.1, element0
// CHECK:STDOUT:   %impl.elem0.loc9_3.1: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc9_3.1: <bound method> = bound_method %.loc9_3.2, %impl.elem0.loc9_3.1
// CHECK:STDOUT:   %specific_fn.loc9_3.1: <specific function> = specific_function %impl.elem0.loc9_3.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.2: <bound method> = bound_method %.loc9_3.2, %specific_fn.loc9_3.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.1: init %i32 = call %bound_method.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   %.loc9_3.3: ref %struct_type.e.f = as_compatible %h.var
// CHECK:STDOUT:   %.loc9_3.4: ref %i32 = struct_access %.loc9_3.3, element0
// CHECK:STDOUT:   %.loc9_3.5: init %i32 to %.loc9_3.4 = in_place_init %Int.as.Copy.impl.Op.call.loc9_3.1
// CHECK:STDOUT:   %.loc9_3.6: %i32 = struct_access %.loc9_3.1, element1
// CHECK:STDOUT:   %impl.elem0.loc9_3.2: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
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
// CHECK:STDOUT:   %h: ref %AdaptStruct = wrapper_binding h, %h.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.27f = ref_binding_pattern h [concrete = constants.%h.patt]
// CHECK:STDOUT:     %h.var_patt: %pattern_type.27f = var_pattern %h.patt [concrete = constants.%h.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %h.ref: ref %AdaptStruct = name_ref h, %h
// CHECK:STDOUT:   %.loc10_11.1: ref %struct_type.e.f = as_compatible %h.ref
// CHECK:STDOUT:   %.loc10_11.2: ref %i32 = struct_access %.loc10_11.1, element0
// CHECK:STDOUT:   %.loc10_11.3: %i32 = acquire_value %.loc10_11.2
// CHECK:STDOUT:   %impl.elem0.loc10_11.1: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.3, %impl.elem0.loc10_11.1
// CHECK:STDOUT:   %specific_fn.loc10_11.1: <specific function> = specific_function %impl.elem0.loc10_11.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.3, %specific_fn.loc10_11.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.1: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.3)
// CHECK:STDOUT:   %.loc10_11.4: ref %struct_type.e.f = as_compatible %return.param
// CHECK:STDOUT:   %.loc10_11.5: ref %i32 = struct_access %.loc10_11.4, element0
// CHECK:STDOUT:   %.loc10_11.6: init %i32 to %.loc10_11.5 = in_place_init %Int.as.Copy.impl.Op.call.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.7: ref %i32 = struct_access %.loc10_11.1, element1
// CHECK:STDOUT:   %.loc10_11.8: %i32 = acquire_value %.loc10_11.7
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.8, %impl.elem0.loc10_11.2
// CHECK:STDOUT:   %specific_fn.loc10_11.2: <specific function> = specific_function %impl.elem0.loc10_11.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.4: <bound method> = bound_method %.loc10_11.8, %specific_fn.loc10_11.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.2: init %i32 = call %bound_method.loc10_11.4(%.loc10_11.8)
// CHECK:STDOUT:   %.loc10_11.9: ref %i32 = struct_access %.loc10_11.4, element1
// CHECK:STDOUT:   %.loc10_11.10: init %i32 to %.loc10_11.9 = in_place_init %Int.as.Copy.impl.Op.call.loc10_11.2
// CHECK:STDOUT:   %.loc10_11.11: init %struct_type.e.f to %.loc10_11.4 = struct_init (%.loc10_11.6, %.loc10_11.10)
// CHECK:STDOUT:   %.loc10_11.12: init %AdaptStruct = as_compatible %.loc10_11.11
// CHECK:STDOUT:   %.loc10_11.13: init %AdaptStruct = converted %h.ref, %.loc10_11.12
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %h.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%h.var)
// CHECK:STDOUT:   return %.loc10_11.13 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %struct_type.e.f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_3.1: ref %i32 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %struct_type.e.f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %struct_type.e.f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.3(%self.param: ref %AdaptStruct) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.3(%self.param: ref %AdaptStruct) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.3(%self.param: ref %AdaptStruct) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.785) -> out %return.param: %tuple.type.785 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %tuple.type.785 = var_storage %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.785 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc14_31.1: %AdaptStruct = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_31.1: %struct_type.e.f = as_compatible %tuple.elem0.loc14_31.1
// CHECK:STDOUT:   %.loc14_31.2: %i32 = struct_access %.loc14_31.1, element0
// CHECK:STDOUT:   %impl.elem0.loc14_31.1: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:   %bound_method.loc14_31.1: <bound method> = bound_method %.loc14_31.2, %impl.elem0.loc14_31.1
// CHECK:STDOUT:   %specific_fn.loc14_31.1: <specific function> = specific_function %impl.elem0.loc14_31.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_31.2: <bound method> = bound_method %.loc14_31.2, %specific_fn.loc14_31.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_31.1: init %i32 = call %bound_method.loc14_31.2(%.loc14_31.2)
// CHECK:STDOUT:   %tuple.elem0.loc14_31.2: ref %AdaptStruct = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc14_31.3: ref %struct_type.e.f = as_compatible %tuple.elem0.loc14_31.2
// CHECK:STDOUT:   %.loc14_31.4: ref %i32 = struct_access %.loc14_31.3, element0
// CHECK:STDOUT:   %.loc14_31.5: init %i32 to %.loc14_31.4 = in_place_init %Int.as.Copy.impl.Op.call.loc14_31.1
// CHECK:STDOUT:   %.loc14_31.6: %i32 = struct_access %.loc14_31.1, element1
// CHECK:STDOUT:   %impl.elem0.loc14_31.2: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
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
// CHECK:STDOUT:   %impl.elem0.loc14_31.3: %.355 = impl_witness_access constants.%Copy.impl_witness.338, element0 [concrete = constants.%UInt.as.Copy.impl.Op.462]
// CHECK:STDOUT:   %bound_method.loc14_31.5: <bound method> = bound_method %tuple.elem1.loc14_31.1, %impl.elem0.loc14_31.3
// CHECK:STDOUT:   %specific_fn.loc14_31.3: <specific function> = specific_function %impl.elem0.loc14_31.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_31.6: <bound method> = bound_method %tuple.elem1.loc14_31.1, %specific_fn.loc14_31.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc14: init %u32 = call %bound_method.loc14_31.6(%tuple.elem1.loc14_31.1)
// CHECK:STDOUT:   %tuple.elem1.loc14_31.2: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc14_31.12: init %u32 to %tuple.elem1.loc14_31.2 = in_place_init %UInt.as.Copy.impl.Op.call.loc14
// CHECK:STDOUT:   %.loc14_31.13: init %tuple.type.785 to %d.var = tuple_init (%.loc14_31.11, %.loc14_31.12)
// CHECK:STDOUT:   %.loc14_3: init %tuple.type.785 = converted %c.ref, %.loc14_31.13
// CHECK:STDOUT:   assign %d.var, %.loc14_3
// CHECK:STDOUT:   %.loc14_27.1: type = splice_block %.loc14_27.3 [concrete = constants.%tuple.type.785] {
// CHECK:STDOUT:     %AdaptStruct.ref.loc14: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %u32.loc14: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc14_27.2: %tuple.type.693 = tuple_literal (%AdaptStruct.ref.loc14, %u32.loc14) [concrete = constants.%tuple.199]
// CHECK:STDOUT:     %.loc14_27.3: type = converted %.loc14_27.2, constants.%tuple.type.785 [concrete = constants.%tuple.type.785]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.785 = wrapper_binding d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.6f0 = ref_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.6f0 = var_pattern %d.patt [concrete = constants.%d.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.ref: ref %tuple.type.785 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc15_10.1: ref %AdaptStruct = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc15_10.1: ref %struct_type.e.f = as_compatible %tuple.elem0.loc15_10.1
// CHECK:STDOUT:   %.loc15_10.2: ref %i32 = struct_access %.loc15_10.1, element0
// CHECK:STDOUT:   %.loc15_10.3: %i32 = acquire_value %.loc15_10.2
// CHECK:STDOUT:   %impl.elem0.loc15_10.1: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
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
// CHECK:STDOUT:   %impl.elem0.loc15_10.2: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
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
// CHECK:STDOUT:   %impl.elem0.loc15_10.3: %.355 = impl_witness_access constants.%Copy.impl_witness.338, element0 [concrete = constants.%UInt.as.Copy.impl.Op.462]
// CHECK:STDOUT:   %bound_method.loc15_10.5: <bound method> = bound_method %.loc15_10.14, %impl.elem0.loc15_10.3
// CHECK:STDOUT:   %specific_fn.loc15_10.3: <specific function> = specific_function %impl.elem0.loc15_10.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.6: <bound method> = bound_method %.loc15_10.14, %specific_fn.loc15_10.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc15: init %u32 = call %bound_method.loc15_10.6(%.loc15_10.14)
// CHECK:STDOUT:   %tuple.elem1.loc15_10.2: ref %u32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc15_10.15: init %u32 to %tuple.elem1.loc15_10.2 = in_place_init %UInt.as.Copy.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_10.16: init %tuple.type.785 to %return.param = tuple_init (%.loc15_10.13, %.loc15_10.15)
// CHECK:STDOUT:   %.loc15_11: init %tuple.type.785 = converted %d.ref, %.loc15_10.16
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %d.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%d.var)
// CHECK:STDOUT:   return %.loc15_11 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc14_3.1(%self.param: ref %u32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc14_3.1(%self.param: ref %u32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc14_3.1(%self.param: ref %u32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc14_3.2(%self.param: ref %tuple.type.785) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %u32 = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc14_3.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc14_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc14_3.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %AdaptStruct = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc14_3.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc14_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc14_3.2(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc14_3.2(%self.param: ref %tuple.type.785) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc14_3.2(%self.param: ref %tuple.type.785) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
