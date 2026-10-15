// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy_format.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy_format.carbon

// --- check_destroy_format.carbon

library "[[@TEST_NAME]]";

// Identifies whether a type can be destroyed.

// Use this function to confirm `DestroyFormat::NonTrivial` (pass) and
// `DestroyFormat::NoDestroy` (fail).
fn CheckHasDestruction[T: Core.Destroy](ref _: T) {}

// Identifies whether a type has trivial destruction.
//
// Use this function to confirm `DestroyFormat::Trivial` (pass),
// `DestroyFormat::NonTrivial` (fail), and `DestroyFormat::NoDestroy`
// (fail).
fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](ref _: T) {}

// --- trivial_destruction.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

//@dump-sem-ir-begin
class C {}

fn Test(var x: C) {
  CheckHasTrivialDestruction(ref x);
}
//@dump-sem-ir-end

// --- fail_destroy_custom_impl.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class C {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

fn Test(var x: C) {
  // CHECK:STDERR: fail_destroy_custom_impl.carbon:[[@LINE+8]]:3: error: cannot convert type `C` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(ref x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_destroy_custom_impl.carbon:[[@LINE-12]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](ref _: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(ref x);
  CheckHasDestruction(ref x);
}

// --- fail_non_trivial_member.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

class Member {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class C {
  var member: Member;
}

fn Test(var x: C) {
  // CHECK:STDERR: fail_non_trivial_member.carbon:[[@LINE+8]]:3: error: cannot convert type `C` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(ref x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_non_trivial_member.carbon:[[@LINE-16]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](ref _: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(ref x);
  CheckHasDestruction(ref x);
}

// --- fail_non_trivial_base.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";

base class Base {
  impl as Core.DestroyImpl {
    fn Op(unused self) {}
  }
}

class C {
  extend base: Base;
}

fn Test(var x: C) {
  // CHECK:STDERR: fail_non_trivial_base.carbon:[[@LINE+8]]:3: error: cannot convert type `C` into type implementing `Core.TrivialDestroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasTrivialDestruction(ref x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_non_trivial_base.carbon:[[@LINE-16]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:5:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasTrivialDestruction[T: Core.TrivialDestroy](ref _: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(ref x);
  CheckHasDestruction(ref x);
}

// --- no_destroy.h

struct DeletedDestructor {
  ~DeletedDestructor() = delete;
};

struct ProtectedDestructor {
protected:
  ~ProtectedDestructor() = default;
};

struct PrivateDestructor {
private:
  ~PrivateDestructor() = default;
};

// --- fail_deleted_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn Test(ref x: Cpp.DeletedDestructor) {
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE+20]]:28: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    28 |   CheckHasDestruction(ref x);
  // CHECK:STDERR:       |                            ^
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_destroy.h:3:3: note: '~DeletedDestructor' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:     3 |   ~DeletedDestructor() = delete;
  // CHECK:STDERR:       |   ^
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE-11]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:4:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](ref _: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE+8]]:3: error: cannot convert type `Cpp.DeletedDestructor` into type implementing `Core.SubobjectDestroy & Core.DestroyImpl` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   CheckHasDestruction(ref x);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deleted_destructor.carbon:[[@LINE-19]]:1: in import [InImport]
  // CHECK:STDERR: check_destroy_format.carbon:4:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckHasDestruction[T: Core.Destroy](ref _: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  CheckHasTrivialDestruction(ref x);
  CheckHasDestruction(ref x);
}

// --- fail_protected_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn Test(ref x: Cpp.ProtectedDestructor) {
  CheckHasTrivialDestruction(ref x);
  CheckHasDestruction(ref x);
}

// --- fail_private_destructor.carbon

library "[[@TEST_NAME]]";

import library "check_destroy_format";
import Cpp library "no_destroy.h";

fn Test(ref x: Cpp.PrivateDestructor) {
  CheckHasTrivialDestruction(ref x);
  CheckHasDestruction(ref x);
}

// CHECK:STDOUT: --- trivial_destruction.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.98b = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.98b = var_param_pattern %x.patt [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.type: type = fn_type @CheckHasTrivialDestruction [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy> [concrete]
// CHECK:STDOUT:   %custom_witness.41a: <witness> = custom_witness (), @TrivialDestroy [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet: %TrivialDestroy.type = facet_value %C, (%custom_witness.41a) [concrete]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction, @CheckHasTrivialDestruction(%TrivialDestroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.CheckHasTrivialDestruction: %CheckHasTrivialDestruction.type = import_ref Main//check_destroy_format, CheckHasTrivialDestruction, loaded [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %x.patt: %pattern_type.98b = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.98b = var_param_pattern %x.patt [concrete = constants.%x.param_patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %x: ref %C = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%x.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %CheckHasTrivialDestruction.ref: %CheckHasTrivialDestruction.type = name_ref CheckHasTrivialDestruction, imports.%Main.CheckHasTrivialDestruction [concrete = constants.%CheckHasTrivialDestruction]
// CHECK:STDOUT:   %x.ref: ref %C = name_ref x, %x
// CHECK:STDOUT:   %.loc10_30: %C = ref_tag %x.ref
// CHECK:STDOUT:   %TrivialDestroy.facet.loc10_35.1: %TrivialDestroy.type = facet_value constants.%C, (constants.%custom_witness.41a) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc10_35.1: %TrivialDestroy.type = converted constants.%C, %TrivialDestroy.facet.loc10_35.1 [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc10_35.2: %TrivialDestroy.type = facet_value constants.%C, (constants.%custom_witness.41a) [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %.loc10_35.2: %TrivialDestroy.type = converted constants.%C, %TrivialDestroy.facet.loc10_35.2 [concrete = constants.%TrivialDestroy.facet]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.specific_fn: <specific function> = specific_function %CheckHasTrivialDestruction.ref, @CheckHasTrivialDestruction(constants.%TrivialDestroy.facet) [concrete = constants.%CheckHasTrivialDestruction.specific_fn]
// CHECK:STDOUT:   %CheckHasTrivialDestruction.call: init %empty_tuple.type = call %CheckHasTrivialDestruction.specific_fn(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_35.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_35.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
