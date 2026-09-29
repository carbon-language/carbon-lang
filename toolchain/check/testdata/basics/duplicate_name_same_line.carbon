// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/duplicate_name_same_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/duplicate_name_same_line.carbon

fn A() {
  if (true) {
    // This declares `n` in two different scopes, it's just showing the IR
    // behavior of having both on the same line.
    //@dump-sem-ir-begin
    var unused n: () = (); } else { var unused n: () = ();
    //@dump-sem-ir-end
  }
}

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %n.patt.785a0d.1: %pattern_type.cb1 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt.961036.1: %pattern_type.cb1 = var_pattern %n.patt.785a0d.1 [concrete]
// CHECK:STDOUT:   %self.param_patt.65a: %pattern_type.cb1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.df1: %pattern_type.cb1 = wrapper_binding_pattern self, %self.param_patt.65a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %n.patt.785a0d.2: %pattern_type.cb1 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt.961036.2: %pattern_type.cb1 = var_pattern %n.patt.785a0d.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc18 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = ref_param_pattern [concrete = constants.%self.param_patt.65a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.df1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_tuple.type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_tuple.type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc18 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = ref_param_pattern [concrete = constants.%self.param_patt.65a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.df1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_tuple.type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_tuple.type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %n.var.loc18_5: ref %empty_tuple.type = var_storage %n.var_patt.loc18_5
// CHECK:STDOUT:   %.loc18_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_25.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_5: init %empty_tuple.type = converted %.loc18_25.1, %.loc18_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %n.var.loc18_5, %.loc18_5
// CHECK:STDOUT:   %.loc18_20.1: type = splice_block %.loc18_20.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_20.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_20.3: type = converted %.loc18_20.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_17: ref %empty_tuple.type = wrapper_binding n, %n.var.loc18_5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_17: %pattern_type.cb1 = ref_binding_pattern n [concrete = constants.%n.patt.785a0d.1]
// CHECK:STDOUT:     %n.var_patt.loc18_5: %pattern_type.cb1 = var_pattern %n.patt.loc18_17 [concrete = constants.%n.var_patt.961036.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc18_5: <bound method> = bound_method %n.var.loc18_5, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc18_5: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc18_5(%n.var.loc18_5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %n.var.loc18_37: ref %empty_tuple.type = var_storage %n.var_patt.loc18_37
// CHECK:STDOUT:   %.loc18_57.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_57.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_37: init %empty_tuple.type = converted %.loc18_57.1, %.loc18_57.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %n.var.loc18_37, %.loc18_37
// CHECK:STDOUT:   %.loc18_52.1: type = splice_block %.loc18_52.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc18_52.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18_52.3: type = converted %.loc18_52.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc18_49: ref %empty_tuple.type = wrapper_binding n, %n.var.loc18_37
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc18_49: %pattern_type.cb1 = ref_binding_pattern n [concrete = constants.%n.patt.785a0d.2]
// CHECK:STDOUT:     %n.var_patt.loc18_37: %pattern_type.cb1 = var_pattern %n.patt.loc18_49 [concrete = constants.%n.var_patt.961036.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc18_37: <bound method> = bound_method %n.var.loc18_37, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc18_37: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc18_37(%n.var.loc18_37)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc18(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc18(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc18(%self.param: ref %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
