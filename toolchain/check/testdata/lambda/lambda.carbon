// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/lambda/lambda.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/lambda/lambda.carbon

// --- empty.carbon
library "[[@TEST_NAME]]";

fn TestEmpty() {
  //@dump-sem-ir-begin
  alias empty = fn () {};
  empty();
  //@dump-sem-ir-end
}

// --- identity.carbon
library "[[@TEST_NAME]]";

fn TestIdentity() {
  //@dump-sem-ir-begin
  alias id_i32 = fn (x: i32) -> i32 { return x; };
  let unused a: i32 = id_i32(42);
  let unused b: i32 = (fn (x: i32) -> i32 { return x; })(10);
  //@dump-sem-ir-end
}

// --- terse_identity.carbon
library "[[@TEST_NAME]]";

fn TestTerse() {
  //@dump-sem-ir-begin
  alias terse_id_i32 = fn (x: i32) => x;
  let unused e: i32 = terse_id_i32(42);
  let unused f: i32 = (fn (x: i32) => x)(10);
  //@dump-sem-ir-end
}

//@dump-sem-ir-begin
alias terse_id_fn = fn (x: i32) => x;
//@dump-sem-ir-end

// --- generic_identity.carbon
library "[[@TEST_NAME]]";

fn TestGeneric() {
  //@dump-sem-ir-begin
  alias generic_id = fn [T: Core.Copy](x: T) -> T { return x; };
  let unused c: i32 = generic_id(42 as i32);
  let unused d: bool = generic_id(true);
  //@dump-sem-ir-end
}

// --- generic_nested.carbon
library "[[@TEST_NAME]]";

fn TestGenericNested() {
  //@dump-sem-ir-begin
  alias generic_id = fn [T: Core.Copy](x: T) -> T {
    return fn(y: T) -> T {
      return fn(z: T) -> T {
        return z;
      }(y);
    }(x);
  };
  let unused c: i32 = generic_id(42 as i32);
  let unused d: bool = generic_id(true);
  //@dump-sem-ir-end
}

// --- extended_types.carbon
library "[[@TEST_NAME]]";

fn TestExtendedTypes() {
  alias with_var = fn (var x: i32) -> i32 {
    returned var r: i32 = x;
    return var;
  };
  let _: i32 = with_var(5);

  alias with_ref_return = fn (ref x: i32) ->? form(ref i32) {
    return x;
  };
  var v: i32 = 1;
  let unused ref r: i32 = with_ref_return(ref v);
}

// --- default_arguments.carbon
library "[[@TEST_NAME]]";

fn TestDefaultArgument() {
  alias with_default = fn (x: i32 = 7) -> i32 {
    return x;
  };
  let _: i32 = with_default();
  let _: i32 = with_default(8);
}

// --- terse_with_default_arguments.carbon
library "[[@TEST_NAME]]";

fn TestTerseWithDefaultArgument() {
  alias terse_with_default = fn (x: i32 = 7) => x;
  let _: i32 = terse_with_default();
  let _: i32 = terse_with_default(8);
}

// --- nested.carbon
library "[[@TEST_NAME]]";

fn TestNested() {
  alias outer = fn () -> i32 {
    alias inner = fn (x: i32) -> i32 { return x; };
    return inner(3);
  };
  let _: i32 = outer();
}

// --- terse_nested.carbon
library "[[@TEST_NAME]]";

fn TestTerseNested() {
  alias terse_outer = fn () => (fn (x: i32) => x)(3);
  let _: i32 = terse_outer();
}

// --- control_flow.carbon
library "[[@TEST_NAME]]";

fn TestControlFlow() {
  //@dump-sem-ir-begin
  alias loop_in_lambda = fn () {
    while (true) {
      break;
    }
  };
  loop_in_lambda();
  //@dump-sem-ir-end
}

// --- as_generic_argument.carbon
library "[[@TEST_NAME]]";

fn PassToGeneric[F: type](unused f: F) {}

alias id_fn = fn (x: i32) -> i32 { return x; };
alias terse_id_fn = fn (x: i32) => x;

fn TestAsGenericArgument() {
  PassToGeneric(id_fn);
  PassToGeneric(terse_id_fn);
}

// --- as_template_argument.carbon
library "[[@TEST_NAME]]";

fn PassToTemplate[template F: type](template f: F) { f(42); }

alias id_fn = fn (x: i32) -> i32 { return x; };

fn TestAsTemplateArgument() {
  PassToTemplate(id_fn);
}

// --- fail_todo_as_non_constant_template_argument.carbon
library "[[@TEST_NAME]]";

// TODO: This should work even when `f` is not constant.
fn PassToTemplate[template F: type](f: F) { f(42); }

alias id_fn = fn (x: i32) -> i32 { return x; };

fn TestAsTemplateArgument() {
  // CHECK:STDERR: fail_todo_as_non_constant_template_argument.carbon:[[@LINE+3]]:3-16: error: unable to monomorphize specific `PassToTemplate(<type of function expression>)` [ResolvingSpecificHere]
  // CHECK:STDERR: fail_todo_as_non_constant_template_argument.carbon:[[@LINE-6]]:45-49: note: value of type `<type of function expression>` is not callable [CallToNonCallable]
  // CHECK:STDERR:
  PassToTemplate(id_fn);
}

// --- in_class.carbon
library "[[@TEST_NAME]]";

class InClass {
  var field: i32 = (fn (x: i32) -> i32 { return x; })(1);
  var terse_field: i32 = (fn (x: i32) => x)(1);
}

// --- fail_self.carbon
library "[[@TEST_NAME]]";

class C {
  fn M() {
    // CHECK:STDERR: fail_self.carbon:[[@LINE+2]]:17-26: error: `self` parameter in function expression [SelfParameterInLambda]
    // CHECK:STDERR:
    (fn (unused self: Self) {});
    // CHECK:STDERR: fail_self.carbon:[[@LINE+2]]:17-26: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
    // CHECK:STDERR:
    (fn [unused self: Self]() {});
  }
}

fn F() {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+2]]:15-23: error: `self` parameter in function expression [SelfParameterInLambda]
  // CHECK:STDERR:
  (fn (unused self: i32) {});
  // CHECK:STDERR: fail_self.carbon:[[@LINE+2]]:30-38: error: `self` parameter in function expression [SelfParameterInLambda]
  // CHECK:STDERR:
  (fn (unused x: i32, unused self: i32) {});
}

// --- fail_return.carbon
library "[[@TEST_NAME]]";

fn Outer() -> i32 {
  // CHECK:STDERR: fail_return.carbon:[[@LINE+2]]:18: error: missing `return` at end of function with declared return type [MissingReturnStatement]
  // CHECK:STDERR:
  (fn () -> i32 {});
  // CHECK:STDERR: fail_return.carbon:[[@LINE+3]]:20-28: error: cannot implicitly convert expression of type `Core.IntLiteral` to `bool` [ConversionFailure]
  // CHECK:STDERR: fail_return.carbon:[[@LINE+2]]:20-28: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  (fn () -> bool { return 1; });
  // CHECK:STDERR: fail_return.carbon:[[@LINE+3]]:12-20: error: no return expression should be provided in this context [ReturnStatementDisallowExpr]
  // CHECK:STDERR: fail_return.carbon:[[@LINE+2]]:4-10: note: there was no return type provided [ReturnTypeOmittedNote]
  // CHECK:STDERR:
  (fn () { return 1; });
  // CHECK:STDERR: fail_return.carbon:[[@LINE+2]]:10-15: error: cannot specify a return type with `=>` [ReturnTypeInTerseFunction]
  // CHECK:STDERR:
  (fn () -> i32 => 0);
  // CHECK:STDERR: fail_return.carbon:[[@LINE+2]]:20-36: error: cannot specify a return form with `=>` [ReturnTypeInTerseFunction]
  // CHECK:STDERR:
  (fn (ref x: i32) ->? form(ref i32) => x);

  return 0;
}

// --- fail_unused.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_unused.carbon:[[@LINE+3]]:8-13: warning: binding `x` unused [UnusedBinding]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_unused.carbon:[[@LINE+1]]:23-28: error: variable `y` marked `unused` but used [UnusedButUsed]
  (fn (x: i32, unused y: i32) -> i32 {
    let z: i32 = 1;
    // CHECK:STDERR: fail_unused.carbon:[[@LINE+4]]:12: note: usage here [UnusedButUsedHere]
    // CHECK:STDERR:
    // CHECK:STDERR: fail_unused.carbon:[[@LINE-3]]:9-14: warning: binding `z` unused [UnusedBinding]
    // CHECK:STDERR:
    return y;
  });
  // CHECK:STDERR: fail_unused.carbon:[[@LINE+5]]:8-13: warning: binding `x` unused [UnusedBinding]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_unused.carbon:[[@LINE+3]]:23-28: error: variable `y` marked `unused` but used [UnusedButUsed]
  // CHECK:STDERR: fail_unused.carbon:[[@LINE+2]]:34: note: usage here [UnusedButUsedHere]
  // CHECK:STDERR:
  (fn (x: i32, unused y: i32) => y);
}

// --- fail_break_continue.carbon
library "[[@TEST_NAME]]";

fn F() {
  while (true) {
    (fn () {
      // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+2]]:7-11: error: `break` can only be used in a loop [BreakOutsideLoop]
      // CHECK:STDERR:
      break;
    });
    (fn () {
      // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+2]]:7-14: error: `continue` can only be used in a loop [ContinueOutsideLoop]
      // CHECK:STDERR:
      continue;
    });
    break;
  }
}

// --- fail_default_values.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_default_values.carbon:[[@LINE+3]]:34-39: error: this pattern is missing a required default value [RequiredPatternDefaultValueMissing]
  // CHECK:STDERR: fail_default_values.carbon:[[@LINE+2]]:15-20: note: all patterns to the right of this first pattern with a default value must also specify a default value [RequiredPatternDefaultValueFirstDefault]
  // CHECK:STDERR:
  (fn (unused x: i32 = 1, unused y: i32) {});
}

// --- fail_incomplete_type.carbon
library "[[@TEST_NAME]]";

class Incomplete;

fn F() {
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE+3]]:15-27: error: parameter has incomplete type `Incomplete` in function definition [IncompleteTypeInFunctionParam]
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE-4]]:1-17: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  (fn (unused x: Incomplete) {});
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE+5]]:10-22: error: function returns incomplete type `Incomplete` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE-8]]:1-17: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE+2]]:25: error: missing `return` at end of function with declared return type [MissingReturnStatement]
  // CHECK:STDERR:
  (fn () -> Incomplete {});
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE+3]]:27-28: error: function returns incomplete type `Incomplete` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE-14]]:1-17: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  (fn (p: Incomplete*) => *p);
}

// --- fail_conversion.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_conversion.carbon:[[@LINE+3]]:16-22: error: cannot implicitly convert expression of type `<type of function expression>` to `i32` [ConversionFailure]
  // CHECK:STDERR: fail_conversion.carbon:[[@LINE+2]]:16-22: note: type `<type of function expression>` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  let _: i32 = fn () {};
}

// --- fail_todo_positional_params.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_todo_positional_params.carbon:[[@LINE+2]]:4-7: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
  // CHECK:STDERR:
  (fn {});
  // CHECK:STDERR: fail_todo_positional_params.carbon:[[@LINE+2]]:4-14: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
  // CHECK:STDERR:
  (fn -> i32 { return 0; });
  // CHECK:STDERR: fail_todo_positional_params.carbon:[[@LINE+2]]:4-8: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
  // CHECK:STDERR:
  (fn => 0);
}

// --- dependent_terse_return.carbon
library "[[@TEST_NAME]]";

fn F() {
  (fn [T: Core.Copy](x: T) => x);
}

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %fn.type: type = fn_type @fn [concrete]
// CHECK:STDOUT:   %fn: %fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestEmpty() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn.decl: %fn.type = fn_decl @fn [concrete = constants.%fn] {} {}
// CHECK:STDOUT:   %empty: %fn.type = alias_binding empty, %fn.decl [concrete = constants.%fn]
// CHECK:STDOUT:   %empty.ref: %fn.type = name_ref empty, %empty [concrete = constants.%fn]
// CHECK:STDOUT:   %fn.call: init %empty_tuple.type = call %empty.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @fn() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %fn.type.4ee: type = fn_type @fn.loc5 [concrete]
// CHECK:STDOUT:   %fn.eaf: %fn.type.4ee = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.98e: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.db5: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %fn.type.ee2: type = fn_type @fn.loc7 [concrete]
// CHECK:STDOUT:   %fn.eb8: %fn.type.ee2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.02d: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.002: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.6a3: %i32 = int_value 10 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIdentity() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn.decl.loc5: %fn.type.4ee = fn_decl @fn.loc5 [concrete = constants.%fn.eaf] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc5_33 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5_33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5: Core.Form = init_form %i32.loc5_33 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc5_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %id_i32: %fn.type.4ee = alias_binding id_i32, %fn.decl.loc5 [concrete = constants.%fn.eaf]
// CHECK:STDOUT:   %id_i32.ref: %fn.type.4ee = name_ref id_i32, %id_i32 [concrete = constants.%fn.eaf]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc6: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc6_30.1: <bound method> = bound_method %int_42, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_30.2: <bound method> = bound_method %int_42, %specific_fn.loc6 [concrete = constants.%bound_method.98e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_30.2(%int_42) [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc6_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc6_30.2: %i32 = converted %int_42, %.loc6_30.1 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %fn.call.loc6: init %i32 = call %id_i32.ref(%.loc6_30.2)
// CHECK:STDOUT:   %.loc6_32.1: %i32 = value_of_initializer %fn.call.loc6
// CHECK:STDOUT:   %.loc6_32.2: %i32 = converted %fn.call.loc6, %.loc6_32.1
// CHECK:STDOUT:   %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc6_32.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %fn.decl.loc7: %fn.type.ee2 = fn_decl @fn.loc7 [concrete = constants.%fn.eb8] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc7_39 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc7_39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7: Core.Form = init_form %i32.loc7_39 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc7_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc7: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_58.1: <bound method> = bound_method %int_10, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.02d]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_58.2: <bound method> = bound_method %int_10, %specific_fn.loc7 [concrete = constants.%bound_method.002]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_58.2(%int_10) [concrete = constants.%int_10.6a3]
// CHECK:STDOUT:   %.loc7_58.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_10.6a3]
// CHECK:STDOUT:   %.loc7_58.2: %i32 = converted %int_10, %.loc7_58.1 [concrete = constants.%int_10.6a3]
// CHECK:STDOUT:   %fn.call.loc7: init %i32 = call %fn.decl.loc7(%.loc7_58.2)
// CHECK:STDOUT:   %.loc7_60.1: %i32 = value_of_initializer %fn.call.loc7
// CHECK:STDOUT:   %.loc7_60.2: %i32 = converted %fn.call.loc7, %.loc7_60.1
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %.loc7_60.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @fn.loc5(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc5_46.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_46.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc5_46.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @fn.loc7(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc7_52.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_52.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_52.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- terse_identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %fn.type.975: type = fn_type @fn.loc5 [concrete]
// CHECK:STDOUT:   %fn.460: %fn.type.975 = struct_value () [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.d94 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.98e: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.db5: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type.d94 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   %fn.type.aea: type = fn_type @fn.loc7 [concrete]
// CHECK:STDOUT:   %fn.4d2: %fn.type.aea = struct_value () [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.02d: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.002: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.6a3: %i32 = int_value 10 [concrete]
// CHECK:STDOUT:   %fn.type.d0d: type = fn_type @fn.loc12 [concrete]
// CHECK:STDOUT:   %fn.5b0: %fn.type.d0d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %fn.decl: %fn.type.d0d = fn_decl @fn.loc12 [concrete = constants.%fn.5b0] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc12_36 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc12_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %i32.loc12_36: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12_36.2: Core.Form = init_form %i32.loc12_36 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %terse_id_fn: %fn.type.d0d = alias_binding terse_id_fn, %fn.decl [concrete = constants.%fn.5b0]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestTerse() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn.decl.loc5: %fn.type.975 = fn_decl @fn.loc5 [concrete = constants.%fn.460] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc5_39 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc5_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %i32.loc5_39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_39.2: Core.Form = init_form %i32.loc5_39 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %terse_id_i32: %fn.type.975 = alias_binding terse_id_i32, %fn.decl.loc5 [concrete = constants.%fn.460]
// CHECK:STDOUT:   %terse_id_i32.ref: %fn.type.975 = name_ref terse_id_i32, %terse_id_i32 [concrete = constants.%fn.460]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc6: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc6_36.1: <bound method> = bound_method %int_42, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.81c]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_36.2: <bound method> = bound_method %int_42, %specific_fn.loc6 [concrete = constants.%bound_method.98e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_36.2(%int_42) [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc6_36.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc6_36.2: %i32 = converted %int_42, %.loc6_36.1 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %fn.call.loc6: init %i32 = call %terse_id_i32.ref(%.loc6_36.2)
// CHECK:STDOUT:   %.loc6_38.1: %i32 = value_of_initializer %fn.call.loc6
// CHECK:STDOUT:   %.loc6_38.2: %i32 = converted %fn.call.loc6, %.loc6_38.1
// CHECK:STDOUT:   %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %e: %i32 = wrapper_binding e, %.loc6_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.d94 = value_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %fn.decl.loc7: %fn.type.aea = fn_decl @fn.loc7 [concrete = constants.%fn.4d2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc7_39 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc7_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %i32.loc7_39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_39.2: Core.Form = init_form %i32.loc7_39 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc7: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_42.1: <bound method> = bound_method %int_10, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.02d]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_42.2: <bound method> = bound_method %int_10, %specific_fn.loc7 [concrete = constants.%bound_method.002]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_42.2(%int_10) [concrete = constants.%int_10.6a3]
// CHECK:STDOUT:   %.loc7_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_10.6a3]
// CHECK:STDOUT:   %.loc7_42.2: %i32 = converted %int_10, %.loc7_42.1 [concrete = constants.%int_10.6a3]
// CHECK:STDOUT:   %fn.call.loc7: init %i32 = call %fn.decl.loc7(%.loc7_42.2)
// CHECK:STDOUT:   %.loc7_44.1: %i32 = value_of_initializer %fn.call.loc7
// CHECK:STDOUT:   %.loc7_44.2: %i32 = converted %fn.call.loc7, %.loc7_44.1
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %f: %i32 = wrapper_binding f, %.loc7_44.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.d94 = value_binding_pattern f [concrete = constants.%f.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @fn.loc5(%x.param: %i32) -> out %return.param: %i32 [deduced] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc5_39.1: init %i32 = splice_block %Int.as.Copy.impl.Op.call {
// CHECK:STDOUT:     %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:     %bound_method.loc5_39.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc5_39.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc5_39.2(%x.ref)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return %.loc5_39.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @fn.loc7(%x.param: %i32) -> out %return.param: %i32 [deduced] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc7_39.1: init %i32 = splice_block %Int.as.Copy.impl.Op.call {
// CHECK:STDOUT:     %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:     %bound_method.loc7_39.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_39.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_39.2(%x.ref)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return %.loc7_39.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @fn.loc12(%x.param: %i32) -> out %return.param: %i32 [deduced] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc12_36.1: init %i32 = splice_block %Int.as.Copy.impl.Op.call {
// CHECK:STDOUT:     %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:     %bound_method.loc12_36.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_36.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc12_36.2(%x.ref)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return %.loc12_36.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.1f4: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.cec: %pattern_type.2c66b4.2 = wrapper_binding_pattern x, %x.param_patt.1f4 [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %fn.type: type = fn_type @fn [concrete]
// CHECK:STDOUT:   %fn: %fn.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2f9c3.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.8bf: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2f9c3.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb5: %.8bf = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ac: <specific function> = specific_impl_function %impl.elem0.bb5, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.d94 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.b93: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.396: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.84a: %Core.IntLiteral.as.As.impl.Convert.type.396 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e09: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b0: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.94f: %Core.IntLiteral.as.As.impl.Convert.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b93 = facet_value Core.IntLiteral, (%As.impl_witness.e09) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2be: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %As.WithSelf.Convert.type.2be, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.94f [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.94f, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.db5: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %x.param_patt.051: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0f7: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt.051 [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.656: <specific function> = specific_function %fn, @fn(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.831 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet.7af: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %x.param_patt.37d: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e47: %pattern_type.831 = wrapper_binding_pattern x, %x.param_patt.37d [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %return.param_patt.5a1: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.c0c: %pattern_type.831 = return_slot_pattern %return.param_patt.5a1, bool [concrete]
// CHECK:STDOUT:   %fn.specific_fn.10a: <specific function> = specific_function %fn, @fn(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.127: <witness> = complete_type_witness bool [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c98: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %.68a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c98, %Copy.facet.7af [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.9e0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.396) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.84a)]
// CHECK:STDOUT:   %As.impl_witness_table.1c2 = impl_witness_table (%Core.import_ref.9e0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGeneric() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn.decl: %fn.type = fn_decl @fn [concrete = constants.%fn] {
// CHECK:STDOUT:     %T.patt.loc5_27.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_27.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:     %x.param_patt.loc5_41.1: @fn.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc5_41.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:     %x.patt.loc5_41.1: @fn.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc5_41.1 [symbolic = %x.patt.loc5_41.2 (constants.%x.patt.cec)]
// CHECK:STDOUT:     %return.param_patt.loc5_49.1: @fn.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc5_49.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:     %return.patt.loc5_46.1: @fn.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc5_49.1, %.loc5_49.3 [symbolic = %return.patt.loc5_46.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5_49: %Copy.type = name_ref T, %T.loc5_27.2 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc5_49: type = facet_access_type %T.ref.loc5_49 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_49.3: type = converted %T.ref.loc5_49, %T.as_type.loc5_49 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_49.4: Core.Form = init_form %.loc5_49.3 [symbolic = %.loc5_49.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %.loc5_33: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_27.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.param: @fn.%T.as_type.loc5_43.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_43.1: type = splice_block %.loc5_43.2 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc5_43: %Copy.type = name_ref T, %T.loc5_27.2 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc5_43.2: type = facet_access_type %T.ref.loc5_43 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_43.2: type = converted %T.ref.loc5_43, %T.as_type.loc5_43.2 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @fn.%T.as_type.loc5_43.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @fn.%T.as_type.loc5_43.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @fn.%T.as_type.loc5_43.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %generic_id: %fn.type = alias_binding generic_id, %fn.decl [concrete = constants.%fn]
// CHECK:STDOUT:   %generic_id.ref.loc6: %fn.type = name_ref generic_id, %generic_id [concrete = constants.%fn]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %i32.loc6_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.979 = impl_witness_access constants.%As.impl_witness.e09, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.94f]
// CHECK:STDOUT:   %bound_method.loc6_37.1: <bound method> = bound_method %int_42, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_37.2: <bound method> = bound_method %int_42, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc6_37.2(%int_42) [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc6_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc6_37.2: %i32 = converted %int_42, %.loc6_37.1 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %Copy.facet.loc6_43.1: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc6_43.1: %Copy.type = converted constants.%i32, %Copy.facet.loc6_43.1 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %Copy.facet.loc6_43.2: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc6_43.2: %Copy.type = converted constants.%i32, %Copy.facet.loc6_43.2 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %fn.specific_fn.loc6: <specific function> = specific_function %generic_id.ref.loc6, @fn(constants.%Copy.facet.95d) [concrete = constants.%fn.specific_fn.656]
// CHECK:STDOUT:   %fn.call.loc6: init %i32 = call %fn.specific_fn.loc6(%.loc6_37.2)
// CHECK:STDOUT:   %.loc6_43.3: %i32 = value_of_initializer %fn.call.loc6
// CHECK:STDOUT:   %.loc6_43.4: %i32 = converted %fn.call.loc6, %.loc6_43.3
// CHECK:STDOUT:   %i32.loc6_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: %i32 = wrapper_binding c, %.loc6_43.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.d94 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %generic_id.ref.loc7: %fn.type = name_ref generic_id, %generic_id [concrete = constants.%fn]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %Copy.facet.loc7_39.1: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5e1) [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %.loc7_39.1: %Copy.type = converted bool, %Copy.facet.loc7_39.1 [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %Copy.facet.loc7_39.2: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5e1) [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %.loc7_39.2: %Copy.type = converted bool, %Copy.facet.loc7_39.2 [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %fn.specific_fn.loc7: <specific function> = specific_function %generic_id.ref.loc7, @fn(constants.%Copy.facet.7af) [concrete = constants.%fn.specific_fn.10a]
// CHECK:STDOUT:   %fn.call.loc7: init bool = call %fn.specific_fn.loc7(%true)
// CHECK:STDOUT:   %.loc7_39.3: bool = value_of_initializer %fn.call.loc7
// CHECK:STDOUT:   %.loc7_39.4: bool = converted %fn.call.loc7, %.loc7_39.3
// CHECK:STDOUT:   %.loc7_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %d: bool = wrapper_binding d, %.loc7_39.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.831 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @fn(%T.loc5_27.2: %Copy.type) {
// CHECK:STDOUT:   %T.patt.loc5_27.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_27.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   %T.loc5_27.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc5_43.1: type = facet_access_type %T.loc5_27.1 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc5_43.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %x.param_patt.loc5_41.2: @fn.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc5_41.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:   %x.patt.loc5_41.2: @fn.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc5_41.2 [symbolic = %x.patt.loc5_41.2 (constants.%x.patt.cec)]
// CHECK:STDOUT:   %.loc5_49.2: Core.Form = init_form %T.as_type.loc5_43.1 [symbolic = %.loc5_49.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc5_49.2: @fn.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc5_49.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc5_46.2: @fn.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc5_49.2, %T.as_type.loc5_43.1 [symbolic = %return.patt.loc5_46.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc5_43.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc5_27.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2f9c3.2)]
// CHECK:STDOUT:   %.loc5_60.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc5_27.1 [symbolic = %.loc5_60.3 (constants.%.8bf)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc5_27.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc5_60.2: @fn.%.loc5_60.3 (%.8bf) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_60.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:   %specific_impl_fn.loc5_60.2: <specific function> = specific_impl_function %impl.elem0.loc5_60.2, @Copy.WithSelf.Op(%T.loc5_27.1) [symbolic = %specific_impl_fn.loc5_60.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @fn.%T.as_type.loc5_43.1 (%T.as_type)) -> out %return.param: @fn.%T.as_type.loc5_43.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @fn.%T.as_type.loc5_43.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc5_60.1: @fn.%.loc5_60.3 (%.8bf) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc5_60.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:     %bound_method.loc5_60.1: <bound method> = bound_method %x.ref, %impl.elem0.loc5_60.1
// CHECK:STDOUT:     %.loc5_60.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %.loc5_60.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc5_60.1: <specific function> = specific_impl_function %impl.elem0.loc5_60.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc5_60.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:     %bound_method.loc5_60.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc5_60.1
// CHECK:STDOUT:     %.loc5_49.1: ref @fn.%T.as_type.loc5_43.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @fn.%T.as_type.loc5_43.1 (%T.as_type) to %.loc5_49.1 = call %bound_method.loc5_60.2(%x.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc5_27.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_27.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc5_43.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %x.param_patt.loc5_41.2 => constants.%x.param_patt.1f4
// CHECK:STDOUT:   %x.patt.loc5_41.2 => constants.%x.patt.cec
// CHECK:STDOUT:   %.loc5_49.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc5_46.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn(constants.%Copy.facet.95d) {
// CHECK:STDOUT:   %T.patt.loc5_27.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_27.1 => constants.%Copy.facet.95d
// CHECK:STDOUT:   %T.as_type.loc5_43.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %x.param_patt.loc5_41.2 => constants.%x.param_patt.051
// CHECK:STDOUT:   %x.patt.loc5_41.2 => constants.%x.patt.0f7
// CHECK:STDOUT:   %.loc5_49.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc5_46.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.f76
// CHECK:STDOUT:   %.loc5_60.3 => constants.%.2b9
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.8af
// CHECK:STDOUT:   %impl.elem0.loc5_60.2 => constants.%Int.as.Copy.impl.Op.cae
// CHECK:STDOUT:   %specific_impl_fn.loc5_60.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn(constants.%Copy.facet.7af) {
// CHECK:STDOUT:   %T.patt.loc5_27.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_27.1 => constants.%Copy.facet.7af
// CHECK:STDOUT:   %T.as_type.loc5_43.1 => bool
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.831
// CHECK:STDOUT:   %x.param_patt.loc5_41.2 => constants.%x.param_patt.37d
// CHECK:STDOUT:   %x.patt.loc5_41.2 => constants.%x.patt.e47
// CHECK:STDOUT:   %.loc5_49.2 => constants.%.f34
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt.5a1
// CHECK:STDOUT:   %return.patt.loc5_46.2 => constants.%return.patt.c0c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.127
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.c98
// CHECK:STDOUT:   %.loc5_60.3 => constants.%.68a
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.5e1
// CHECK:STDOUT:   %impl.elem0.loc5_60.2 => constants.%bool.as.Copy.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc5_60.2 => constants.%bool.as.Copy.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt.1f4: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.cec: %pattern_type.2c66b4.2 = wrapper_binding_pattern x, %x.param_patt.1f4 [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %fn.type.d4a: type = fn_type @fn.loc5 [concrete]
// CHECK:STDOUT:   %fn.9dd: %fn.type.d4a = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %y.param_patt.ec0: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.d6d: %pattern_type.2c66b4.2 = wrapper_binding_pattern y, %y.param_patt.ec0 [symbolic]
// CHECK:STDOUT:   %fn.type.7d1: type = fn_type @fn.loc6, @fn.loc5(%T.f84) [symbolic]
// CHECK:STDOUT:   %fn.dd0: %fn.type.7d1 = struct_value () [symbolic]
// CHECK:STDOUT:   %z.param_patt.b42: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %z.patt.513: %pattern_type.2c66b4.2 = wrapper_binding_pattern z, %z.param_patt.b42 [symbolic]
// CHECK:STDOUT:   %fn.type.531: type = fn_type @fn.loc7, @fn.loc6(%T.f84) [symbolic]
// CHECK:STDOUT:   %fn.dc1: %fn.type.531 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2f9c3.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.8bf: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2f9c3.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb5: %.8bf = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ac: <specific function> = specific_impl_function %impl.elem0.bb5, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %fn.specific_fn.6d4: <specific function> = specific_function %fn.dc1, @fn.loc7(%T.f84) [symbolic]
// CHECK:STDOUT:   %fn.specific_fn.eed: <specific function> = specific_function %fn.dd0, @fn.loc6(%T.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.d94 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.b93: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.396: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.84a: %Core.IntLiteral.as.As.impl.Convert.type.396 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e09: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b0: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.94f: %Core.IntLiteral.as.As.impl.Convert.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b93 = facet_value Core.IntLiteral, (%As.impl_witness.e09) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2be: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %As.WithSelf.Convert.type.2be, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.94f [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.94f, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.db5: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %x.param_patt.051: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0f7: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt.051 [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.d27: <specific function> = specific_function %fn.9dd, @fn.loc5(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.831 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet.7af: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %x.param_patt.37d: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e47: %pattern_type.831 = wrapper_binding_pattern x, %x.param_patt.37d [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %return.param_patt.5a1: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.c0c: %pattern_type.831 = return_slot_pattern %return.param_patt.5a1, bool [concrete]
// CHECK:STDOUT:   %fn.specific_fn.8de: <specific function> = specific_function %fn.9dd, @fn.loc5(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %fn.type.949: type = fn_type @fn.loc6, @fn.loc5(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %fn.62e: %fn.type.949 = struct_value () [concrete]
// CHECK:STDOUT:   %y.param_patt.665: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.e18: %pattern_type.d94 = wrapper_binding_pattern y, %y.param_patt.665 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.458: <specific function> = specific_function %fn.62e, @fn.loc6(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %complete_type.127: <witness> = complete_type_witness bool [concrete]
// CHECK:STDOUT:   %fn.type.319: type = fn_type @fn.loc6, @fn.loc5(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %fn.1af: %fn.type.319 = struct_value () [concrete]
// CHECK:STDOUT:   %y.param_patt.077: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.252: %pattern_type.831 = wrapper_binding_pattern y, %y.param_patt.077 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.c2d: <specific function> = specific_function %fn.1af, @fn.loc6(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %fn.type.70b: type = fn_type @fn.loc7, @fn.loc6(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %fn.37d: %fn.type.70b = struct_value () [concrete]
// CHECK:STDOUT:   %z.param_patt.896: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %z.patt.660: %pattern_type.d94 = wrapper_binding_pattern z, %z.param_patt.896 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.6d6: <specific function> = specific_function %fn.37d, @fn.loc7(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %fn.type.c1a: type = fn_type @fn.loc7, @fn.loc6(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %fn.d5a: %fn.type.c1a = struct_value () [concrete]
// CHECK:STDOUT:   %z.param_patt.a49: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %z.patt.246: %pattern_type.831 = wrapper_binding_pattern z, %z.param_patt.a49 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.8ab: <specific function> = specific_function %fn.d5a, @fn.loc7(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c98: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %.68a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c98, %Copy.facet.7af [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.9e0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.396) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.84a)]
// CHECK:STDOUT:   %As.impl_witness_table.1c2 = impl_witness_table (%Core.import_ref.9e0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGenericNested() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn.decl: %fn.type.d4a = fn_decl @fn.loc5 [concrete = constants.%fn.9dd] {
// CHECK:STDOUT:     %T.patt.loc5_27.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_27.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:     %x.param_patt.loc5_41.1: @fn.loc5.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc5_41.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:     %x.patt.loc5_41.1: @fn.loc5.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc5_41.1 [symbolic = %x.patt.loc5_41.2 (constants.%x.patt.cec)]
// CHECK:STDOUT:     %return.param_patt.loc5_49.1: @fn.loc5.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc5_49.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:     %return.patt.loc5_46.1: @fn.loc5.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc5_49.1, %.loc5_49.3 [symbolic = %return.patt.loc5_46.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5_49: %Copy.type = name_ref T, %T.loc5_27.2 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc5_49: type = facet_access_type %T.ref.loc5_49 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_49.3: type = converted %T.ref.loc5_49, %T.as_type.loc5_49 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_49.4: Core.Form = init_form %.loc5_49.3 [symbolic = %.loc5_49.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %.loc5_33: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_27.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.param: @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_43.1: type = splice_block %.loc5_43.2 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc5_43: %Copy.type = name_ref T, %T.loc5_27.2 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc5_43.2: type = facet_access_type %T.ref.loc5_43 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_43.2: type = converted %T.ref.loc5_43, %T.as_type.loc5_43.2 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %generic_id: %fn.type.d4a = alias_binding generic_id, %fn.decl [concrete = constants.%fn.9dd]
// CHECK:STDOUT:   %generic_id.ref.loc12: %fn.type.d4a = name_ref generic_id, %generic_id [concrete = constants.%fn.9dd]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %i32.loc12_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.979 = impl_witness_access constants.%As.impl_witness.e09, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.94f]
// CHECK:STDOUT:   %bound_method.loc12_37.1: <bound method> = bound_method %int_42, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_37.2: <bound method> = bound_method %int_42, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc12_37.2(%int_42) [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc12_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %.loc12_37.2: %i32 = converted %int_42, %.loc12_37.1 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %Copy.facet.loc12_43.1: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc12_43.1: %Copy.type = converted constants.%i32, %Copy.facet.loc12_43.1 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %Copy.facet.loc12_43.2: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc12_43.2: %Copy.type = converted constants.%i32, %Copy.facet.loc12_43.2 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %fn.specific_fn.loc12: <specific function> = specific_function %generic_id.ref.loc12, @fn.loc5(constants.%Copy.facet.95d) [concrete = constants.%fn.specific_fn.d27]
// CHECK:STDOUT:   %fn.call.loc12: init %i32 = call %fn.specific_fn.loc12(%.loc12_37.2)
// CHECK:STDOUT:   %.loc12_43.3: %i32 = value_of_initializer %fn.call.loc12
// CHECK:STDOUT:   %.loc12_43.4: %i32 = converted %fn.call.loc12, %.loc12_43.3
// CHECK:STDOUT:   %i32.loc12_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: %i32 = wrapper_binding c, %.loc12_43.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.d94 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %generic_id.ref.loc13: %fn.type.d4a = name_ref generic_id, %generic_id [concrete = constants.%fn.9dd]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %Copy.facet.loc13_39.1: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5e1) [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %.loc13_39.1: %Copy.type = converted bool, %Copy.facet.loc13_39.1 [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %Copy.facet.loc13_39.2: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5e1) [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %.loc13_39.2: %Copy.type = converted bool, %Copy.facet.loc13_39.2 [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %fn.specific_fn.loc13: <specific function> = specific_function %generic_id.ref.loc13, @fn.loc5(constants.%Copy.facet.7af) [concrete = constants.%fn.specific_fn.8de]
// CHECK:STDOUT:   %fn.call.loc13: init bool = call %fn.specific_fn.loc13(%true)
// CHECK:STDOUT:   %.loc13_39.3: bool = value_of_initializer %fn.call.loc13
// CHECK:STDOUT:   %.loc13_39.4: bool = converted %fn.call.loc13, %.loc13_39.3
// CHECK:STDOUT:   %.loc13_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %d: bool = wrapper_binding d, %.loc13_39.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.831 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @fn.loc5(%T.loc5_27.2: %Copy.type) {
// CHECK:STDOUT:   %T.patt.loc5_27.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_27.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   %T.loc5_27.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc5_43.1: type = facet_access_type %T.loc5_27.1 [symbolic = %T.as_type.loc5_43.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc5_43.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %x.param_patt.loc5_41.2: @fn.loc5.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc5_41.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:   %x.patt.loc5_41.2: @fn.loc5.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern x, %x.param_patt.loc5_41.2 [symbolic = %x.patt.loc5_41.2 (constants.%x.patt.cec)]
// CHECK:STDOUT:   %.loc5_49.2: Core.Form = init_form %T.as_type.loc5_43.1 [symbolic = %.loc5_49.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc5_49.2: @fn.loc5.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc5_49.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc5_46.2: @fn.loc5.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc5_49.2, %T.as_type.loc5_43.1 [symbolic = %return.patt.loc5_46.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc5_43.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %fn.type: type = fn_type @fn.loc6, @fn.loc5(%T.loc5_27.1) [symbolic = %fn.type (constants.%fn.type.7d1)]
// CHECK:STDOUT:   %fn: @fn.loc5.%fn.type (%fn.type.7d1) = struct_value () [symbolic = %fn (constants.%fn.dd0)]
// CHECK:STDOUT:   %fn.specific_fn.loc6_26.2: <specific function> = specific_function %fn, @fn.loc6(%T.loc5_27.1) [symbolic = %fn.specific_fn.loc6_26.2 (constants.%fn.specific_fn.eed)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type)) -> out %return.param: @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %fn.decl: @fn.loc5.%fn.type (%fn.type.7d1) = fn_decl @fn.loc6 [symbolic = @fn.loc5.%fn (constants.%fn.dd0)] {
// CHECK:STDOUT:       %y.param_patt.loc6_16.1: @fn.loc6.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %y.param_patt.loc6_16.2 (constants.%y.param_patt.ec0)]
// CHECK:STDOUT:       %y.patt.loc6_16.1: @fn.loc6.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern y, %y.param_patt.loc6_16.1 [symbolic = %y.patt.loc6_16.2 (constants.%y.patt.d6d)]
// CHECK:STDOUT:       %return.param_patt.loc6_24.1: @fn.loc6.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc6_24.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:       %return.patt.loc6_21.1: @fn.loc6.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc6_24.1, %.loc6_24.4 [symbolic = %return.patt.loc6_21.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc6_24.3: %Copy.type = specific_constant @fn.loc5.%T.loc5_27.2, @fn.loc5(constants.%T.f84) [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.ref.loc6_24: %Copy.type = name_ref T, %.loc6_24.3 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc6_24: type = facet_access_type %T.ref.loc6_24 [symbolic = %T.as_type.loc6_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_24.4: type = converted %T.ref.loc6_24, %T.as_type.loc6_24 [symbolic = %T.as_type.loc6_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_24.5: Core.Form = init_form %.loc6_24.4 [symbolic = %.loc6_24.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:       %y.param: @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc6_18.1: type = splice_block %.loc6_18.3 [symbolic = %T.as_type.loc6_18.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %.loc6_18.2: %Copy.type = specific_constant @fn.loc5.%T.loc5_27.2, @fn.loc5(constants.%T.f84) [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:         %T.ref.loc6_18: %Copy.type = name_ref T, %.loc6_18.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:         %T.as_type.loc6_18.2: type = facet_access_type %T.ref.loc6_18 [symbolic = %T.as_type.loc6_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc6_18.3: type = converted %T.ref.loc6_18, %T.as_type.loc6_18.2 [symbolic = %T.as_type.loc6_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %y: @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type) = wrapper_binding y, %y.param
// CHECK:STDOUT:       %return.param: ref @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.ref: @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc10_8.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %.loc10_8.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %fn.specific_fn.loc6_26.1: <specific function> = specific_function %fn.decl, @fn.loc6(constants.%T.f84) [symbolic = %fn.specific_fn.loc6_26.2 (constants.%fn.specific_fn.eed)]
// CHECK:STDOUT:     %.loc5_49.1: ref @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %fn.call: init @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type) to %.loc5_49.1 = call %fn.specific_fn.loc6_26.1(%x.ref)
// CHECK:STDOUT:     return %fn.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @fn.loc6(@fn.loc5.%T.loc5_27.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc6_18.1: type = facet_access_type %T [symbolic = %T.as_type.loc6_18.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc6_18.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %y.param_patt.loc6_16.2: @fn.loc6.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %y.param_patt.loc6_16.2 (constants.%y.param_patt.ec0)]
// CHECK:STDOUT:   %y.patt.loc6_16.2: @fn.loc6.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern y, %y.param_patt.loc6_16.2 [symbolic = %y.patt.loc6_16.2 (constants.%y.patt.d6d)]
// CHECK:STDOUT:   %.loc6_24.2: Core.Form = init_form %T.as_type.loc6_18.1 [symbolic = %.loc6_24.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc6_24.2: @fn.loc6.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc6_24.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc6_21.2: @fn.loc6.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc6_24.2, %T.as_type.loc6_18.1 [symbolic = %return.patt.loc6_21.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc6_18.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %fn.type: type = fn_type @fn.loc7, @fn.loc6(%T) [symbolic = %fn.type (constants.%fn.type.531)]
// CHECK:STDOUT:   %fn: @fn.loc6.%fn.type (%fn.type.531) = struct_value () [symbolic = %fn (constants.%fn.dc1)]
// CHECK:STDOUT:   %fn.specific_fn.loc7_28.2: <specific function> = specific_function %fn, @fn.loc7(%T) [symbolic = %fn.specific_fn.loc7_28.2 (constants.%fn.specific_fn.6d4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type)) -> out %return.param: @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %fn.decl: @fn.loc6.%fn.type (%fn.type.531) = fn_decl @fn.loc7 [symbolic = @fn.loc6.%fn (constants.%fn.dc1)] {
// CHECK:STDOUT:       %z.param_patt.loc7_18.1: @fn.loc7.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %z.param_patt.loc7_18.2 (constants.%z.param_patt.b42)]
// CHECK:STDOUT:       %z.patt.loc7_18.1: @fn.loc7.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern z, %z.param_patt.loc7_18.1 [symbolic = %z.patt.loc7_18.2 (constants.%z.patt.513)]
// CHECK:STDOUT:       %return.param_patt.loc7_26.1: @fn.loc7.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc7_26.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:       %return.patt.loc7_23.1: @fn.loc7.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc7_26.1, %.loc7_26.4 [symbolic = %return.patt.loc7_23.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc7_26.3: %Copy.type = specific_constant @fn.loc5.%T.loc5_27.2, @fn.loc5(constants.%T.f84) [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.ref.loc7_26: %Copy.type = name_ref T, %.loc7_26.3 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc7_26: type = facet_access_type %T.ref.loc7_26 [symbolic = %T.as_type.loc7_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_26.4: type = converted %T.ref.loc7_26, %T.as_type.loc7_26 [symbolic = %T.as_type.loc7_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_26.5: Core.Form = init_form %.loc7_26.4 [symbolic = %.loc7_26.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:       %z.param: @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc7_20.1: type = splice_block %.loc7_20.3 [symbolic = %T.as_type.loc7_20.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %.loc7_20.2: %Copy.type = specific_constant @fn.loc5.%T.loc5_27.2, @fn.loc5(constants.%T.f84) [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:         %T.ref.loc7_20: %Copy.type = name_ref T, %.loc7_20.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:         %T.as_type.loc7_20.2: type = facet_access_type %T.ref.loc7_20 [symbolic = %T.as_type.loc7_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc7_20.3: type = converted %T.ref.loc7_20, %T.as_type.loc7_20.2 [symbolic = %T.as_type.loc7_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %z: @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) = wrapper_binding z, %z.param
// CHECK:STDOUT:       %return.param: ref @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y.ref: @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type) = name_ref y, %y
// CHECK:STDOUT:     %.loc9_10.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc9_10.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %fn.specific_fn.loc7_28.1: <specific function> = specific_function %fn.decl, @fn.loc7(constants.%T.f84) [symbolic = %fn.specific_fn.loc7_28.2 (constants.%fn.specific_fn.6d4)]
// CHECK:STDOUT:     %.loc6_24.1: ref @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %fn.call: init @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type) to %.loc6_24.1 = call %fn.specific_fn.loc7_28.1(%y.ref)
// CHECK:STDOUT:     return %fn.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @fn.loc7(@fn.loc5.%T.loc5_27.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc7_20.1: type = facet_access_type %T [symbolic = %T.as_type.loc7_20.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_20.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %z.param_patt.loc7_18.2: @fn.loc7.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %z.param_patt.loc7_18.2 (constants.%z.param_patt.b42)]
// CHECK:STDOUT:   %z.patt.loc7_18.2: @fn.loc7.%pattern_type (%pattern_type.2c66b4.2) = wrapper_binding_pattern z, %z.param_patt.loc7_18.2 [symbolic = %z.patt.loc7_18.2 (constants.%z.patt.513)]
// CHECK:STDOUT:   %.loc7_26.2: Core.Form = init_form %T.as_type.loc7_20.1 [symbolic = %.loc7_26.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc7_26.2: @fn.loc7.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc7_26.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc7_23.2: @fn.loc7.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc7_26.2, %T.as_type.loc7_20.1 [symbolic = %return.patt.loc7_23.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_20.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2f9c3.2)]
// CHECK:STDOUT:   %.loc8_16.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc8_16.3 (constants.%.8bf)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc8_16.2: @fn.loc7.%.loc8_16.3 (%.8bf) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_16.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_16.2: <specific function> = specific_impl_function %impl.elem0.loc8_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc8_16.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%z.param: @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type)) -> out %return.param: @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %z.ref: @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) = name_ref z, %z
// CHECK:STDOUT:     %impl.elem0.loc8_16.1: @fn.loc7.%.loc8_16.3 (%.8bf) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc8_16.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:     %bound_method.loc8_16.1: <bound method> = bound_method %z.ref, %impl.elem0.loc8_16.1
// CHECK:STDOUT:     %.loc8_16.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc8_16.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_16.1: <specific function> = specific_impl_function %impl.elem0.loc8_16.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc8_16.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:     %bound_method.loc8_16.2: <bound method> = bound_method %z.ref, %specific_impl_fn.loc8_16.1
// CHECK:STDOUT:     %.loc7_26.1: ref @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) to %.loc7_26.1 = call %bound_method.loc8_16.2(%z.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc5(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc5_27.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_27.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc5_43.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %x.param_patt.loc5_41.2 => constants.%x.param_patt.1f4
// CHECK:STDOUT:   %x.patt.loc5_41.2 => constants.%x.patt.cec
// CHECK:STDOUT:   %.loc5_49.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc5_46.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc6(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc6_18.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %y.param_patt.loc6_16.2 => constants.%y.param_patt.ec0
// CHECK:STDOUT:   %y.patt.loc6_16.2 => constants.%y.patt.d6d
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %return.param_patt.loc6_24.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc6_21.2 => constants.%return.patt.f44
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83
// CHECK:STDOUT:   %fn.type => constants.%fn.type.531
// CHECK:STDOUT:   %fn => constants.%fn.dc1
// CHECK:STDOUT:   %fn.specific_fn.loc7_28.2 => constants.%fn.specific_fn.6d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc7(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc7_20.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %z.param_patt.loc7_18.2 => constants.%z.param_patt.b42
// CHECK:STDOUT:   %z.patt.loc7_18.2 => constants.%z.patt.513
// CHECK:STDOUT:   %.loc7_26.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %return.param_patt.loc7_26.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc7_23.2 => constants.%return.patt.f44
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d2f9c3.2
// CHECK:STDOUT:   %.loc8_16.3 => constants.%.8bf
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.d9c
// CHECK:STDOUT:   %impl.elem0.loc8_16.2 => constants.%impl.elem0.bb5
// CHECK:STDOUT:   %specific_impl_fn.loc8_16.2 => constants.%specific_impl_fn.2ac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc5(constants.%Copy.facet.95d) {
// CHECK:STDOUT:   %T.patt.loc5_27.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_27.1 => constants.%Copy.facet.95d
// CHECK:STDOUT:   %T.as_type.loc5_43.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %x.param_patt.loc5_41.2 => constants.%x.param_patt.051
// CHECK:STDOUT:   %x.patt.loc5_41.2 => constants.%x.patt.0f7
// CHECK:STDOUT:   %.loc5_49.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc5_46.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %fn.type => constants.%fn.type.949
// CHECK:STDOUT:   %fn => constants.%fn.62e
// CHECK:STDOUT:   %fn.specific_fn.loc6_26.2 => constants.%fn.specific_fn.458
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc5(constants.%Copy.facet.7af) {
// CHECK:STDOUT:   %T.patt.loc5_27.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_27.1 => constants.%Copy.facet.7af
// CHECK:STDOUT:   %T.as_type.loc5_43.1 => bool
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.831
// CHECK:STDOUT:   %x.param_patt.loc5_41.2 => constants.%x.param_patt.37d
// CHECK:STDOUT:   %x.patt.loc5_41.2 => constants.%x.patt.e47
// CHECK:STDOUT:   %.loc5_49.2 => constants.%.f34
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt.5a1
// CHECK:STDOUT:   %return.patt.loc5_46.2 => constants.%return.patt.c0c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.127
// CHECK:STDOUT:   %fn.type => constants.%fn.type.319
// CHECK:STDOUT:   %fn => constants.%fn.1af
// CHECK:STDOUT:   %fn.specific_fn.loc6_26.2 => constants.%fn.specific_fn.c2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc6(constants.%Copy.facet.95d) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.95d
// CHECK:STDOUT:   %T.as_type.loc6_18.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %y.param_patt.loc6_16.2 => constants.%y.param_patt.665
// CHECK:STDOUT:   %y.patt.loc6_16.2 => constants.%y.patt.e18
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc6_24.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc6_21.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %fn.type => constants.%fn.type.70b
// CHECK:STDOUT:   %fn => constants.%fn.37d
// CHECK:STDOUT:   %fn.specific_fn.loc7_28.2 => constants.%fn.specific_fn.6d6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc6(constants.%Copy.facet.7af) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.7af
// CHECK:STDOUT:   %T.as_type.loc6_18.1 => bool
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.831
// CHECK:STDOUT:   %y.param_patt.loc6_16.2 => constants.%y.param_patt.077
// CHECK:STDOUT:   %y.patt.loc6_16.2 => constants.%y.patt.252
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.f34
// CHECK:STDOUT:   %return.param_patt.loc6_24.2 => constants.%return.param_patt.5a1
// CHECK:STDOUT:   %return.patt.loc6_21.2 => constants.%return.patt.c0c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.127
// CHECK:STDOUT:   %fn.type => constants.%fn.type.c1a
// CHECK:STDOUT:   %fn => constants.%fn.d5a
// CHECK:STDOUT:   %fn.specific_fn.loc7_28.2 => constants.%fn.specific_fn.8ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc7(constants.%Copy.facet.95d) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.95d
// CHECK:STDOUT:   %T.as_type.loc7_20.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %z.param_patt.loc7_18.2 => constants.%z.param_patt.896
// CHECK:STDOUT:   %z.patt.loc7_18.2 => constants.%z.patt.660
// CHECK:STDOUT:   %.loc7_26.2 => constants.%.0b3
// CHECK:STDOUT:   %return.param_patt.loc7_26.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc7_23.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.f76
// CHECK:STDOUT:   %.loc8_16.3 => constants.%.2b9
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.8af
// CHECK:STDOUT:   %impl.elem0.loc8_16.2 => constants.%Int.as.Copy.impl.Op.cae
// CHECK:STDOUT:   %specific_impl_fn.loc8_16.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc7(constants.%Copy.facet.7af) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.7af
// CHECK:STDOUT:   %T.as_type.loc7_20.1 => bool
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.831
// CHECK:STDOUT:   %z.param_patt.loc7_18.2 => constants.%z.param_patt.a49
// CHECK:STDOUT:   %z.patt.loc7_18.2 => constants.%z.patt.246
// CHECK:STDOUT:   %.loc7_26.2 => constants.%.f34
// CHECK:STDOUT:   %return.param_patt.loc7_26.2 => constants.%return.param_patt.5a1
// CHECK:STDOUT:   %return.patt.loc7_23.2 => constants.%return.patt.c0c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.127
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.c98
// CHECK:STDOUT:   %.loc8_16.3 => constants.%.68a
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.5e1
// CHECK:STDOUT:   %impl.elem0.loc8_16.2 => constants.%bool.as.Copy.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc8_16.2 => constants.%bool.as.Copy.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- control_flow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %fn.type: type = fn_type @fn [concrete]
// CHECK:STDOUT:   %fn: %fn.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestControlFlow() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn.decl: %fn.type = fn_decl @fn [concrete = constants.%fn] {} {}
// CHECK:STDOUT:   %loop_in_lambda: %fn.type = alias_binding loop_in_lambda, %fn.decl [concrete = constants.%fn]
// CHECK:STDOUT:   %loop_in_lambda.ref: %fn.type = name_ref loop_in_lambda, %loop_in_lambda [concrete = constants.%fn]
// CHECK:STDOUT:   %fn.call: init %empty_tuple.type = call %loop_in_lambda.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @fn() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
