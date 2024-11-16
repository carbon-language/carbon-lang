// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/identity.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/identity.carbon

// `expr as T` should behave the same as `expr` if `T` is the type of `expr`.

class X {
  // ...
}

fn Value(n: X) {
  let m: X = n as X;
}

fn Reference(p: X*) {
  let q: X* = &(*p as X);
}

fn Make() -> X;

fn Initializing() {
  var x: X = (Make() as X);
}

// CHECK:STDOUT: --- identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Value.type: type = fn_type @Value [template]
// CHECK:STDOUT:   %Value: %Value.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %X [template]
// CHECK:STDOUT:   %Reference.type: type = fn_type @Reference [template]
// CHECK:STDOUT:   %Reference: %Reference.type = struct_value () [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %Initializing.type: type = fn_type @Initializing [template]
// CHECK:STDOUT:   %Initializing: %Initializing.type = struct_value () [template]
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
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Value = %Value.decl
// CHECK:STDOUT:     .Reference = %Reference.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Initializing = %Initializing.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   %Value.decl: %Value.type = fn_decl @Value [template = constants.%Value] {
// CHECK:STDOUT:     %n.patt: %X = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %X = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref.loc17: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %n.param: %X = value_param runtime_param0
// CHECK:STDOUT:     %n: %X = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Reference.decl: %Reference.type = fn_decl @Reference [template = constants.%Reference] {
// CHECK:STDOUT:     %p.patt: %.4 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.4 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref.loc21: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %.loc21: type = ptr_type %X [template = constants.%.4]
// CHECK:STDOUT:     %p.param: %.4 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.4 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:     %return.param: ref %X = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Initializing.decl: %Initializing.type = fn_decl @Initializing [template = constants.%Initializing] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Value(%n.param_patt: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc18_10: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %n.ref: %X = name_ref n, %n
// CHECK:STDOUT:   %X.ref.loc18_19: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %m: %X = bind_name m, %n.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Reference(%p.param_patt: %.4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc22_10: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc22_11: type = ptr_type %X [template = constants.%.4]
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc22_17: ref %X = deref %p.ref
// CHECK:STDOUT:   %X.ref.loc22_23: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %.loc22_15: %.4 = addr_of %.loc22_17
// CHECK:STDOUT:   %q: %.4 = bind_name q, %.loc22_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Initializing() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %X.ref.loc28_10: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %x.var: ref %X = var x
// CHECK:STDOUT:   %x: ref %X = bind_name x, %x.var
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc28: ref %X = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %X = call %Make.ref() to %.loc28
// CHECK:STDOUT:   %X.ref.loc28_25: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
