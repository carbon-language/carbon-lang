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
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Value.type: type = fn_type @Value [concrete]
// CHECK:STDOUT:   %Value: %Value.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %Reference.type: type = fn_type @Reference [concrete]
// CHECK:STDOUT:   %Reference: %Reference.type = struct_value () [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Initializing.type: type = fn_type @Initializing [concrete]
// CHECK:STDOUT:   %Initializing: %Initializing.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Value = %Value.decl
// CHECK:STDOUT:     .Reference = %Reference.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Initializing = %Initializing.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %Value.decl: %Value.type = fn_decl @Value [concrete = constants.%Value] {
// CHECK:STDOUT:     %n.patt: %X = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %X = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref.loc17: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %n: %X = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Reference.decl: %Reference.type = fn_decl @Reference [concrete = constants.%Reference] {
// CHECK:STDOUT:     %p.patt: %ptr.d17 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.d17 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr.d17 = value_param call_param0
// CHECK:STDOUT:     %.loc21: type = splice_block %ptr.loc21 [concrete = constants.%ptr.d17] {
// CHECK:STDOUT:       %X.ref.loc21: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:       %ptr.loc21: type = ptr_type %X.ref.loc21 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.d17 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %X = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %X = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %return.param: ref %X = out_param call_param0
// CHECK:STDOUT:     %return: ref %X = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Initializing.decl: %Initializing.type = fn_decl @Initializing [concrete = constants.%Initializing] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Value(%n.param_patt: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %X = binding_pattern m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.ref: %X = name_ref n, %n
// CHECK:STDOUT:   %X.ref.loc18_19: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %X.ref.loc18_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %m: %X = bind_name m, %n.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Reference(%p.param_patt: %ptr.d17) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %q.patt: %ptr.d17 = binding_pattern q
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.ref: %ptr.d17 = name_ref p, %p
// CHECK:STDOUT:   %.loc22_17: ref %X = deref %p.ref
// CHECK:STDOUT:   %X.ref.loc22_23: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %addr: %ptr.d17 = addr_of %.loc22_17
// CHECK:STDOUT:   %.loc22_11: type = splice_block %ptr.loc22 [concrete = constants.%ptr.d17] {
// CHECK:STDOUT:     %X.ref.loc22_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %ptr.loc22: type = ptr_type %X.ref.loc22_10 [concrete = constants.%ptr.d17]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %q: %ptr.d17 = bind_name q, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %X;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Initializing() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %X = binding_pattern x
// CHECK:STDOUT:     %.loc28_3.1: %X = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %X = var x
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc28_3.2: ref %X = splice_block %x.var {}
// CHECK:STDOUT:   %Make.call: init %X = call %Make.ref() to %.loc28_3.2
// CHECK:STDOUT:   %X.ref.loc28_25: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   assign %x.var, %Make.call
// CHECK:STDOUT:   %X.ref.loc28_10: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %x: ref %X = bind_name x, %x.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
