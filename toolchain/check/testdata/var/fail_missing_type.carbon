// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_missing_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_missing_type.carbon

// --- fail_no_prior_definition.carbon

library "[[@TEST_NAME]]";

class AClassType {}

// CHECK:STDERR: fail_no_prior_definition.carbon:[[@LINE+4]]:5: error: name `some_object_defined_earlier` not found [NameNotFound]
// CHECK:STDERR: var some_object_defined_earlier = AClassType;
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var some_object_defined_earlier = AClassType;

// --- fail_with_prior_definition.carbon

library "[[@TEST_NAME]]";

class AClassType {
  fn Make() -> AClassType {
    return {};
  }
}

var some_object_defined_earlier: AClassType = AClassType.Make();

// CHECK:STDERR: fail_with_prior_definition.carbon:[[@LINE+4]]:5: error: semantics TODO: `expression pattern` [SemanticsTodo]
// CHECK:STDERR: var some_object_defined_earlier = AClassType;
// CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var some_object_defined_earlier = AClassType;

fn UseObject(ref object: AClassType);

fn CheckLookup() {
  UseObject(ref some_object_defined_earlier);
}

// CHECK:STDOUT: --- fail_no_prior_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AClassType: type = class_type @AClassType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .AClassType = %AClassType.decl
// CHECK:STDOUT:     .some_object_defined_earlier = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AClassType.decl: type = class_decl @AClassType [concrete = constants.%AClassType] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %expr_patt: <error> = expr_pattern %some_object_defined_earlier.ref in [concrete] {
// CHECK:STDOUT:       %some_object_defined_earlier.ref: <error> = name_ref some_object_defined_earlier, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AClassType {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AClassType
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AClassType.ref: type = name_ref AClassType, file.%AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_with_prior_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AClassType: type = class_type @AClassType [concrete]
// CHECK:STDOUT:   %.fac: Core.Form = init_form %AClassType [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %AClassType [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type = return_slot_pattern %return.param_patt, %AClassType [concrete]
// CHECK:STDOUT:   %AClassType.Make.type: type = fn_type @AClassType.Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AClassType.Make: %AClassType.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %AClassType.val: %AClassType = struct_value () [concrete]
// CHECK:STDOUT:   %some_object_defined_earlier.patt: %pattern_type = ref_binding_pattern some_object_defined_earlier [concrete]
// CHECK:STDOUT:   %some_object_defined_earlier.var_patt: %pattern_type = var_pattern %some_object_defined_earlier.patt [concrete]
// CHECK:STDOUT:   %object.param_patt: %pattern_type = ref_param_pattern [concrete]
// CHECK:STDOUT:   %object.patt: %pattern_type = at_binding_pattern object, %object.param_patt [concrete]
// CHECK:STDOUT:   %UseObject.type: type = fn_type @UseObject [concrete]
// CHECK:STDOUT:   %UseObject: %UseObject.type = struct_value () [concrete]
// CHECK:STDOUT:   %CheckLookup.type: type = fn_type @CheckLookup [concrete]
// CHECK:STDOUT:   %CheckLookup: %CheckLookup.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .AClassType = %AClassType.decl
// CHECK:STDOUT:     .some_object_defined_earlier = %some_object_defined_earlier
// CHECK:STDOUT:     .UseObject = %UseObject.decl
// CHECK:STDOUT:     .CheckLookup = %CheckLookup.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AClassType.decl: type = class_decl @AClassType [concrete = constants.%AClassType] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %some_object_defined_earlier.patt: %pattern_type = ref_binding_pattern some_object_defined_earlier [concrete = constants.%some_object_defined_earlier.patt]
// CHECK:STDOUT:     %some_object_defined_earlier.var_patt: %pattern_type = var_pattern %some_object_defined_earlier.patt [concrete = constants.%some_object_defined_earlier.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %some_object_defined_earlier.var: ref %AClassType = var %some_object_defined_earlier.var_patt [concrete]
// CHECK:STDOUT:   %AClassType.ref: type = name_ref AClassType, %AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:   %some_object_defined_earlier: ref %AClassType = ref_binding some_object_defined_earlier, %some_object_defined_earlier.var [concrete = %some_object_defined_earlier.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %expr_patt: %pattern_type = expr_pattern %some_object_defined_earlier.ref in [concrete] {
// CHECK:STDOUT:       %some_object_defined_earlier.ref: ref %AClassType = name_ref some_object_defined_earlier, %some_object_defined_earlier [concrete = %some_object_defined_earlier.var]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseObject.decl: %UseObject.type = fn_decl @UseObject [concrete = constants.%UseObject] {
// CHECK:STDOUT:     %object.param_patt: %pattern_type = ref_param_pattern [concrete = constants.%object.param_patt]
// CHECK:STDOUT:     %object.patt: %pattern_type = at_binding_pattern object, %object.param_patt [concrete = constants.%object.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %object.param: ref %AClassType = ref_param call_param0
// CHECK:STDOUT:     %AClassType.ref: type = name_ref AClassType, file.%AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:     %object: ref %AClassType = ref_binding object, %object.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CheckLookup.decl: %CheckLookup.type = fn_decl @CheckLookup [concrete = constants.%CheckLookup] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AClassType {
// CHECK:STDOUT:   %AClassType.Make.decl: %AClassType.Make.type = fn_decl @AClassType.Make [concrete = constants.%AClassType.Make] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern %return.param_patt, %AClassType.ref [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AClassType.ref: type = name_ref AClassType, file.%AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:     %.loc5: Core.Form = init_form %AClassType.ref [concrete = constants.%.fac]
// CHECK:STDOUT:     %return.param: ref %AClassType = out_param call_param0
// CHECK:STDOUT:     %return: ref %AClassType = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AClassType
// CHECK:STDOUT:   .AClassType = <poisoned>
// CHECK:STDOUT:   .Make = %AClassType.Make.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AClassType.Make() -> out %return.param: %AClassType {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_13.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_13.2: init %AClassType to %return.param = class_init () [concrete = constants.%AClassType.val]
// CHECK:STDOUT:   %.loc6_14: init %AClassType = converted %.loc6_13.1, %.loc6_13.2 [concrete = constants.%AClassType.val]
// CHECK:STDOUT:   return %.loc6_14 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UseObject(%object.param: ref %AClassType);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CheckLookup() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %UseObject.ref: %UseObject.type = name_ref UseObject, file.%UseObject.decl [concrete = constants.%UseObject]
// CHECK:STDOUT:   %some_object_defined_earlier.ref: ref %AClassType = name_ref some_object_defined_earlier, file.%some_object_defined_earlier [concrete = file.%some_object_defined_earlier.var]
// CHECK:STDOUT:   %.loc21: %AClassType = ref_tag %some_object_defined_earlier.ref
// CHECK:STDOUT:   %UseObject.call: init %empty_tuple.type = call %UseObject.ref(%some_object_defined_earlier.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AClassType.ref.loc10: type = name_ref AClassType, file.%AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:   %Make.ref: %AClassType.Make.type = name_ref Make, @AClassType.%AClassType.Make.decl [concrete = constants.%AClassType.Make]
// CHECK:STDOUT:   %.loc10: ref %AClassType = splice_block file.%some_object_defined_earlier.var [concrete = file.%some_object_defined_earlier.var] {}
// CHECK:STDOUT:   %AClassType.Make.call: init %AClassType to %.loc10 = call %Make.ref()
// CHECK:STDOUT:   assign file.%some_object_defined_earlier.var, %AClassType.Make.call
// CHECK:STDOUT:   %AClassType.ref.loc16: type = name_ref AClassType, file.%AClassType.decl [concrete = constants.%AClassType]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
