// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_builtin_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_builtin_call.carbon

// --- int_literal.carbon

library "[[@TEST_NAME]]";

fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();

// --- generic_impl.carbon

library "[[@TEST_NAME]]";

import library "int_literal";

interface Add {
  fn Op(self, other: Self) -> Self;
}

fn Int(n: IntLiteral) -> type = "int.make_type_signed";

class MyInt(N:! IntLiteral) {
  adapt Int(N);
}

impl forall [N:! IntLiteral] MyInt(N) as Add {
  fn Op(self, other: Self) -> Self = "int.sadd";
}

fn Double[N:! IntLiteral](x: MyInt(N)) -> MyInt(N) {
  return x.(Add.Op)(x);
}

// --- use_generic_impl.carbon

library "[[@TEST_NAME]]";

import library "generic_impl";

fn CallImportedDouble(n: MyInt(64)) -> MyInt(64) {
  //@dump-sem-ir-begin
  return Double(n);
  //@dump-sem-ir-end
}

// --- convert_symbolic.carbon

library "[[@TEST_NAME]]";

import library "int_literal";

fn Int(n: IntLiteral) -> type = "int.make_type_signed";
fn ToLiteral(n: Int(32)) -> IntLiteral = "int.convert_checked";
fn FromLiteral(n: IntLiteral) -> Int(32) = "int.convert_checked";

fn Make(N:! Int(32)) -> Int(ToLiteral(N)) { return Make(N); }

class OtherInt {
  adapt Int(32);
  fn ToLiteral(self) -> IntLiteral;
};

fn OtherInt.ToLiteral(self) -> IntLiteral = "int.convert_checked";

fn MakeFromClass(N:! OtherInt) -> Int(N.ToLiteral()) { return MakeFromClass(N); }

// --- use_convert_symbolic.carbon

library "[[@TEST_NAME]]";

import library "convert_symbolic";

//@dump-sem-ir-begin
var m: Int(64) = Make(FromLiteral(64));
var n: Int(64) = MakeFromClass(FromLiteral(64) as OtherInt);
//@dump-sem-ir-end


// CHECK:STDOUT: --- use_generic_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %MyInt.9fc: type = class_type @MyInt, @MyInt(%int_64) [concrete]
// CHECK:STDOUT:   %Double.type: type = fn_type @Double [concrete]
// CHECK:STDOUT:   %Double: %Double.type = struct_value () [concrete]
// CHECK:STDOUT:   %Double.specific_fn: <specific function> = specific_function %Double, @Double(%int_64) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Double: %Double.type = import_ref Main//generic_impl, Double, loaded [concrete = constants.%Double]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallImportedDouble(%n.param: %MyInt.9fc) -> out %return.param: %MyInt.9fc {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Double.ref: %Double.type = name_ref Double, imports.%Main.Double [concrete = constants.%Double]
// CHECK:STDOUT:   %n.ref: %MyInt.9fc = name_ref n, %n
// CHECK:STDOUT:   %Double.specific_fn: <specific function> = specific_function %Double.ref, @Double(constants.%int_64) [concrete = constants.%Double.specific_fn]
// CHECK:STDOUT:   %Double.call: init %MyInt.9fc = call %Double.specific_fn(%n.ref)
// CHECK:STDOUT:   return %Double.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_convert_symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64.fab: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64.builtin: type = int_type signed, %int_64.fab [concrete]
// CHECK:STDOUT:   %pattern_type.e13: type = pattern_type %i64.builtin [concrete]
// CHECK:STDOUT:   %m.patt: %pattern_type.e13 = ref_binding_pattern m [concrete]
// CHECK:STDOUT:   %m.var_patt: %pattern_type.e13 = var_pattern %m.patt [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %FromLiteral.type: type = fn_type @FromLiteral [concrete]
// CHECK:STDOUT:   %FromLiteral: %FromLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64.f82: %i32.builtin = int_value 64 [concrete]
// CHECK:STDOUT:   %Make.specific_fn.bb3: <specific function> = specific_function %Make, @Make(%int_64.f82) [concrete]
// CHECK:STDOUT:   %n.patt.914: %pattern_type.e13 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.e13 = var_pattern %n.patt.914 [concrete]
// CHECK:STDOUT:   %MakeFromClass.type: type = fn_type @MakeFromClass [concrete]
// CHECK:STDOUT:   %MakeFromClass: %MakeFromClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %OtherInt: type = class_type @OtherInt [concrete]
// CHECK:STDOUT:   %int_64.420: %OtherInt = int_value 64 [concrete]
// CHECK:STDOUT:   %MakeFromClass.specific_fn.3a7: <specific function> = specific_function %MakeFromClass, @MakeFromClass(%int_64.420) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Int: %Int.type = import_ref Main//convert_symbolic, Int, loaded [concrete = constants.%Int]
// CHECK:STDOUT:   %Main.FromLiteral: %FromLiteral.type = import_ref Main//convert_symbolic, FromLiteral, loaded [concrete = constants.%FromLiteral]
// CHECK:STDOUT:   %Main.Make: %Make.type = import_ref Main//convert_symbolic, Make, loaded [concrete = constants.%Make]
// CHECK:STDOUT:   %Main.OtherInt: type = import_ref Main//convert_symbolic, OtherInt, loaded [concrete = constants.%OtherInt]
// CHECK:STDOUT:   %Main.MakeFromClass: %MakeFromClass.type = import_ref Main//convert_symbolic, MakeFromClass, loaded [concrete = constants.%MakeFromClass]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %pattern_type.e13 = ref_binding_pattern m [concrete = constants.%m.patt]
// CHECK:STDOUT:     %m.var_patt: %pattern_type.e13 = var_pattern %m.patt [concrete = constants.%m.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.var: ref %i64.builtin = var %m.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_14.1: type = splice_block %.loc7_14.3 [concrete = constants.%i64.builtin] {
// CHECK:STDOUT:     %Int.ref.loc7: %Int.type = name_ref Int, imports.%Main.Int [concrete = constants.%Int]
// CHECK:STDOUT:     %int_64.loc7: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:     %Int.call.loc7: init type = call %Int.ref.loc7(%int_64.loc7) [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc7_14.2: type = value_of_initializer %Int.call.loc7 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc7_14.3: type = converted %Int.call.loc7, %.loc7_14.2 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m: ref %i64.builtin = wrapper_binding m, %m.var [concrete = %m.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.e13 = ref_binding_pattern n [concrete = constants.%n.patt.914]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.e13 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i64.builtin = var %n.var_patt [concrete]
// CHECK:STDOUT:   %.loc8_14.1: type = splice_block %.loc8_14.3 [concrete = constants.%i64.builtin] {
// CHECK:STDOUT:     %Int.ref.loc8: %Int.type = name_ref Int, imports.%Main.Int [concrete = constants.%Int]
// CHECK:STDOUT:     %int_64.loc8: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:     %Int.call.loc8: init type = call %Int.ref.loc8(%int_64.loc8) [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc8_14.2: type = value_of_initializer %Int.call.loc8 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:     %.loc8_14.3: type = converted %Int.call.loc8, %.loc8_14.2 [concrete = constants.%i64.builtin]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i64.builtin = wrapper_binding n, %n.var [concrete = %n.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%Main.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %FromLiteral.ref.loc7: %FromLiteral.type = name_ref FromLiteral, imports.%Main.FromLiteral [concrete = constants.%FromLiteral]
// CHECK:STDOUT:   %int_64.loc7: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:   %FromLiteral.call.loc7: init %i32.builtin = call %FromLiteral.ref.loc7(%int_64.loc7) [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %.loc7_38.1: %i32.builtin = value_of_initializer %FromLiteral.call.loc7 [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %.loc7_38.2: %i32.builtin = converted %FromLiteral.call.loc7, %.loc7_38.1 [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make.ref, @Make(constants.%int_64.f82) [concrete = constants.%Make.specific_fn.bb3]
// CHECK:STDOUT:   %Make.call: init %i64.builtin = call %Make.specific_fn()
// CHECK:STDOUT:   assign file.%m.var, %Make.call
// CHECK:STDOUT:   %MakeFromClass.ref: %MakeFromClass.type = name_ref MakeFromClass, imports.%Main.MakeFromClass [concrete = constants.%MakeFromClass]
// CHECK:STDOUT:   %FromLiteral.ref.loc8: %FromLiteral.type = name_ref FromLiteral, imports.%Main.FromLiteral [concrete = constants.%FromLiteral]
// CHECK:STDOUT:   %int_64.loc8: Core.IntLiteral = int_value 64 [concrete = constants.%int_64.fab]
// CHECK:STDOUT:   %FromLiteral.call.loc8: init %i32.builtin = call %FromLiteral.ref.loc8(%int_64.loc8) [concrete = constants.%int_64.f82]
// CHECK:STDOUT:   %OtherInt.ref: type = name_ref OtherInt, imports.%Main.OtherInt [concrete = constants.%OtherInt]
// CHECK:STDOUT:   %.loc8_48.1: init %OtherInt = as_compatible %FromLiteral.call.loc8 [concrete = constants.%int_64.420]
// CHECK:STDOUT:   %.loc8_48.2: init %OtherInt = converted %FromLiteral.call.loc8, %.loc8_48.1 [concrete = constants.%int_64.420]
// CHECK:STDOUT:   %.loc8_59.1: %OtherInt = value_of_initializer %.loc8_48.2 [concrete = constants.%int_64.420]
// CHECK:STDOUT:   %.loc8_59.2: %OtherInt = converted %.loc8_48.2, %.loc8_59.1 [concrete = constants.%int_64.420]
// CHECK:STDOUT:   %MakeFromClass.specific_fn: <specific function> = specific_function %MakeFromClass.ref, @MakeFromClass(constants.%int_64.420) [concrete = constants.%MakeFromClass.specific_fn.3a7]
// CHECK:STDOUT:   %MakeFromClass.call: init %i64.builtin = call %MakeFromClass.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %MakeFromClass.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
