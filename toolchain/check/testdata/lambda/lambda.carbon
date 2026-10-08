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

// --- nested.carbon
library "[[@TEST_NAME]]";

fn TestNested() {
  alias outer = fn () -> i32 {
    alias inner = fn (x: i32) -> i32 { return x; };
    return inner(3);
  };
  let _: i32 = outer();
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

fn TestAsGenericArgument() {
  PassToGeneric(id_fn);
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
  // CHECK:STDERR: fail_todo_as_non_constant_template_argument.carbon:[[@LINE+7]]:3: error: unable to monomorphize specific `PassToTemplate(<type of function expression>)` [ResolvingSpecificHere]
  // CHECK:STDERR:   PassToTemplate(id_fn);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_as_non_constant_template_argument.carbon:[[@LINE-8]]:45: note: value of type `<type of function expression>` is not callable [CallToNonCallable]
  // CHECK:STDERR: fn PassToTemplate[template F: type](f: F) { f(42); }
  // CHECK:STDERR:                                             ^~~~~
  // CHECK:STDERR:
  PassToTemplate(id_fn);
}

// --- in_class.carbon
library "[[@TEST_NAME]]";

class InClass {
  var field: i32 = (fn (x: i32) -> i32 { return x; })(1);
}

// --- fail_self.carbon
library "[[@TEST_NAME]]";

class C {
  fn M() {
    // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:17: error: `self` parameter in function expression [SelfParameterInLambda]
    // CHECK:STDERR:     (fn (unused self: Self) {});
    // CHECK:STDERR:                 ^~~~~~~~~~
    // CHECK:STDERR:
    (fn (unused self: Self) {});
    // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:17: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
    // CHECK:STDERR:     (fn [unused self: Self]() {});
    // CHECK:STDERR:                 ^~~~~~~~~~
    // CHECK:STDERR:
    (fn [unused self: Self]() {});
  }
}

fn F() {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:15: error: `self` parameter in function expression [SelfParameterInLambda]
  // CHECK:STDERR:   (fn (unused self: i32) {});
  // CHECK:STDERR:               ^~~~~~~~~
  // CHECK:STDERR:
  (fn (unused self: i32) {});
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:30: error: `self` parameter in function expression [SelfParameterInLambda]
  // CHECK:STDERR:   (fn (unused x: i32, unused self: i32) {});
  // CHECK:STDERR:                              ^~~~~~~~~
  // CHECK:STDERR:
  (fn (unused x: i32, unused self: i32) {});
}

// --- fail_return.carbon
library "[[@TEST_NAME]]";

fn Outer() -> i32 {
  // CHECK:STDERR: fail_return.carbon:[[@LINE+4]]:18: error: missing `return` at end of function with declared return type [MissingReturnStatement]
  // CHECK:STDERR:   (fn () -> i32 {});
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  (fn () -> i32 {});
  // CHECK:STDERR: fail_return.carbon:[[@LINE+7]]:20: error: cannot implicitly convert expression of type `Core.IntLiteral` to `bool` [ConversionFailure]
  // CHECK:STDERR:   (fn () -> bool { return 1; });
  // CHECK:STDERR:                    ^~~~~~~~~
  // CHECK:STDERR: fail_return.carbon:[[@LINE+4]]:20: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   (fn () -> bool { return 1; });
  // CHECK:STDERR:                    ^~~~~~~~~
  // CHECK:STDERR:
  (fn () -> bool { return 1; });
  // CHECK:STDERR: fail_return.carbon:[[@LINE+7]]:12: error: no return expression should be provided in this context [ReturnStatementDisallowExpr]
  // CHECK:STDERR:   (fn () { return 1; });
  // CHECK:STDERR:            ^~~~~~~~~
  // CHECK:STDERR: fail_return.carbon:[[@LINE+4]]:4: note: there was no return type provided [ReturnTypeOmittedNote]
  // CHECK:STDERR:   (fn () { return 1; });
  // CHECK:STDERR:    ^~~~~~~
  // CHECK:STDERR:
  (fn () { return 1; });

  return 0;
}

// --- fail_unused.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_unused.carbon:[[@LINE+7]]:8: warning: binding `x` unused [UnusedBinding]
  // CHECK:STDERR:   (fn (x: i32, unused y: i32) -> i32 {
  // CHECK:STDERR:        ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_unused.carbon:[[@LINE+3]]:23: error: variable `y` marked `unused` but used [UnusedButUsed]
  // CHECK:STDERR:   (fn (x: i32, unused y: i32) -> i32 {
  // CHECK:STDERR:                       ^~~~~~
  (fn (x: i32, unused y: i32) -> i32 {
    let z: i32 = 1;
    // CHECK:STDERR: fail_unused.carbon:[[@LINE+8]]:12: note: usage here [UnusedButUsedHere]
    // CHECK:STDERR:     return y;
    // CHECK:STDERR:            ^
    // CHECK:STDERR:
    // CHECK:STDERR: fail_unused.carbon:[[@LINE-5]]:9: warning: binding `z` unused [UnusedBinding]
    // CHECK:STDERR:     let z: i32 = 1;
    // CHECK:STDERR:         ^~~~~~
    // CHECK:STDERR:
    return y;
  });
}

// --- fail_break_continue.carbon
library "[[@TEST_NAME]]";

fn F() {
  while (true) {
    (fn () {
      // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:7: error: `break` can only be used in a loop [BreakOutsideLoop]
      // CHECK:STDERR:       break;
      // CHECK:STDERR:       ^~~~~
      // CHECK:STDERR:
      break;
    });
    (fn () {
      // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:7: error: `continue` can only be used in a loop [ContinueOutsideLoop]
      // CHECK:STDERR:       continue;
      // CHECK:STDERR:       ^~~~~~~~
      // CHECK:STDERR:
      continue;
    });
    break;
  }
}

// --- fail_default_values.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_default_values.carbon:[[@LINE+7]]:34: error: this pattern is missing a required default value. [RequiredPatternDefaultValueMissing]
  // CHECK:STDERR:   (fn (unused x: i32 = 1, unused y: i32) {});
  // CHECK:STDERR:                                  ^~~~~~
  // CHECK:STDERR: fail_default_values.carbon:[[@LINE+4]]:15: note: all patterns to the right of this first pattern with a default value must also specify a default value. [RequiredPatternDefaultValueFirstDefault]
  // CHECK:STDERR:   (fn (unused x: i32 = 1, unused y: i32) {});
  // CHECK:STDERR:               ^~~~~~
  // CHECK:STDERR:
  (fn (unused x: i32 = 1, unused y: i32) {});
}

// --- fail_incomplete_type.carbon
library "[[@TEST_NAME]]";

class Incomplete;

fn F() {
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE+7]]:15: error: parameter has incomplete type `Incomplete` in function definition [IncompleteTypeInFunctionParam]
  // CHECK:STDERR:   (fn (unused x: Incomplete) {});
  // CHECK:STDERR:               ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE-6]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (fn (unused x: Incomplete) {});
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE+11]]:10: error: function returns incomplete type `Incomplete` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   (fn () -> Incomplete {});
  // CHECK:STDERR:          ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE-14]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Incomplete;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_incomplete_type.carbon:[[@LINE+4]]:25: error: missing `return` at end of function with declared return type [MissingReturnStatement]
  // CHECK:STDERR:   (fn () -> Incomplete {});
  // CHECK:STDERR:                         ^
  // CHECK:STDERR:
  (fn () -> Incomplete {});
}

// --- fail_conversion.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_conversion.carbon:[[@LINE+7]]:16: error: cannot implicitly convert expression of type `<type of function expression>` to `i32` [ConversionFailure]
  // CHECK:STDERR:   let _: i32 = fn () {};
  // CHECK:STDERR:                ^~~~~~~
  // CHECK:STDERR: fail_conversion.carbon:[[@LINE+4]]:16: note: type `<type of function expression>` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: i32 = fn () {};
  // CHECK:STDERR:                ^~~~~~~
  // CHECK:STDERR:
  let _: i32 = fn () {};
}

// --- fail_todo_positional_params.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_todo_positional_params.carbon:[[@LINE+4]]:4: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
  // CHECK:STDERR:   (fn {});
  // CHECK:STDERR:    ^~~~
  // CHECK:STDERR:
  (fn {});
  // CHECK:STDERR: fail_todo_positional_params.carbon:[[@LINE+4]]:4: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
  // CHECK:STDERR:   (fn -> i32 { return 0; });
  // CHECK:STDERR:    ^~~~~~~~~~~
  // CHECK:STDERR:
  (fn -> i32 { return 0; });
}

// --- fail_todo_terse_body.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_todo_terse_body.carbon:[[@LINE+4]]:4: error: semantics TODO: `terse lambda body` [SemanticsTodo]
  // CHECK:STDERR:   (fn (x: i32) => x);
  // CHECK:STDERR:    ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (fn (x: i32) => x);
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
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %fn.type.4ee: type = fn_type @fn.loc5 [concrete]
// CHECK:STDOUT:   %fn.eaf: %fn.type.4ee = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf3: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.098: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.ac4: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %fn.type.ee2: type = fn_type @fn.loc7 [concrete]
// CHECK:STDOUT:   %fn.eb8: %fn.type.ee2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcb: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.1b6: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.6bf: %i32 = int_value 10 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestIdentity() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn.decl.loc5: %fn.type.4ee = fn_decl @fn.loc5 [concrete = constants.%fn.eaf] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc5_33 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5_33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5: Core.Form = init_form %i32.loc5_33 [concrete = constants.%.795f]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc5_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %id_i32: %fn.type.4ee = alias_binding id_i32, %fn.decl.loc5 [concrete = constants.%fn.eaf]
// CHECK:STDOUT:   %id_i32.ref: %fn.type.4ee = name_ref id_i32, %id_i32 [concrete = constants.%fn.eaf]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc6: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc6_30.1: <bound method> = bound_method %int_42, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf3]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_30.2: <bound method> = bound_method %int_42, %specific_fn.loc6 [concrete = constants.%bound_method.098]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_30.2(%int_42) [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc6_30.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc6_30.2: %i32 = converted %int_42, %.loc6_30.1 [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %fn.call.loc6: init %i32 = call %id_i32.ref(%.loc6_30.2)
// CHECK:STDOUT:   %.loc6_32.1: %i32 = value_of_initializer %fn.call.loc6
// CHECK:STDOUT:   %.loc6_32.2: %i32 = converted %fn.call.loc6, %.loc6_32.1
// CHECK:STDOUT:   %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc6_32.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %fn.decl.loc7: %fn.type.ee2 = fn_decl @fn.loc7 [concrete = constants.%fn.eb8] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc7_39 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc7_39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7: Core.Form = init_form %i32.loc7_39 [concrete = constants.%.795f]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc7_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc7: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc7_58.1: <bound method> = bound_method %int_10, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcb]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_58.2: <bound method> = bound_method %int_10, %specific_fn.loc7 [concrete = constants.%bound_method.1b6]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_58.2(%int_10) [concrete = constants.%int_10.6bf]
// CHECK:STDOUT:   %.loc7_58.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_10.6bf]
// CHECK:STDOUT:   %.loc7_58.2: %i32 = converted %int_10, %.loc7_58.1 [concrete = constants.%int_10.6bf]
// CHECK:STDOUT:   %fn.call.loc7: init %i32 = call %fn.decl.loc7(%.loc7_58.2)
// CHECK:STDOUT:   %.loc7_60.1: %i32 = value_of_initializer %fn.call.loc7
// CHECK:STDOUT:   %.loc7_60.2: %i32 = converted %fn.call.loc7, %.loc7_60.1
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %.loc7_60.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @fn.loc5(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
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
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc7_52.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_52.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_52.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
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
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.6b6 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.155: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e7f: <witness> = impl_witness imports.%As.impl_witness_table.d22, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e79: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8a4: %Core.IntLiteral.as.As.impl.Convert.type.e79 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.155 = facet_value Core.IntLiteral, (%As.impl_witness.e7f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d02: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.453: type = fn_type_with_self_type %As.WithSelf.Convert.type.d02, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.8a4 [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.8a4, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.ac4: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.1e1: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %x.param_patt.cee: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e3b: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt.cee [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.2d9: <specific function> = specific_function %fn, @fn(%Copy.facet.1e1) [concrete]
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
// CHECK:STDOUT:   %fn.specific_fn.4f4: <specific function> = specific_function %fn, @fn(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.1e1) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet.1e1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %complete_type.127: <witness> = complete_type_witness bool [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d02: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %.e28: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d02, %Copy.facet.7af [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.d22 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %impl.elem0: %.453 = impl_witness_access constants.%As.impl_witness.e7f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.8a4]
// CHECK:STDOUT:   %bound_method.loc6_37.1: <bound method> = bound_method %int_42, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_37.2: <bound method> = bound_method %int_42, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc6_37.2(%int_42) [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc6_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc6_37.2: %i32 = converted %int_42, %.loc6_37.1 [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %Copy.facet.loc6_43.1: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.b51) [concrete = constants.%Copy.facet.1e1]
// CHECK:STDOUT:   %.loc6_43.1: %Copy.type = converted constants.%i32, %Copy.facet.loc6_43.1 [concrete = constants.%Copy.facet.1e1]
// CHECK:STDOUT:   %Copy.facet.loc6_43.2: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.b51) [concrete = constants.%Copy.facet.1e1]
// CHECK:STDOUT:   %.loc6_43.2: %Copy.type = converted constants.%i32, %Copy.facet.loc6_43.2 [concrete = constants.%Copy.facet.1e1]
// CHECK:STDOUT:   %fn.specific_fn.loc6: <specific function> = specific_function %generic_id.ref.loc6, @fn(constants.%Copy.facet.1e1) [concrete = constants.%fn.specific_fn.2d9]
// CHECK:STDOUT:   %fn.call.loc6: init %i32 = call %fn.specific_fn.loc6(%.loc6_37.2)
// CHECK:STDOUT:   %.loc6_43.3: %i32 = value_of_initializer %fn.call.loc6
// CHECK:STDOUT:   %.loc6_43.4: %i32 = converted %fn.call.loc6, %.loc6_43.3
// CHECK:STDOUT:   %i32.loc6_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: %i32 = wrapper_binding c, %.loc6_43.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.6b6 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %generic_id.ref.loc7: %fn.type = name_ref generic_id, %generic_id [concrete = constants.%fn]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %Copy.facet.loc7_39.1: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5e1) [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %.loc7_39.1: %Copy.type = converted bool, %Copy.facet.loc7_39.1 [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %Copy.facet.loc7_39.2: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5e1) [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %.loc7_39.2: %Copy.type = converted bool, %Copy.facet.loc7_39.2 [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %fn.specific_fn.loc7: <specific function> = specific_function %generic_id.ref.loc7, @fn(constants.%Copy.facet.7af) [concrete = constants.%fn.specific_fn.4f4]
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
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc5_27.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc5_60.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc5_27.1 [symbolic = %.loc5_60.3 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc5_27.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc5_60.2: @fn.%.loc5_60.3 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc5_60.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc5_60.2: <specific function> = specific_impl_function %impl.elem0.loc5_60.2, @Copy.WithSelf.Op(%T.loc5_27.1) [symbolic = %specific_impl_fn.loc5_60.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @fn.%T.as_type.loc5_43.1 (%T.as_type)) -> out %return.param: @fn.%T.as_type.loc5_43.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @fn.%T.as_type.loc5_43.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc5_60.1: @fn.%.loc5_60.3 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc5_60.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc5_60.1: <bound method> = bound_method %x.ref, %impl.elem0.loc5_60.1
// CHECK:STDOUT:     %.loc5_60.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %.loc5_60.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc5_27.1 (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc5_60.1: <specific function> = specific_impl_function %impl.elem0.loc5_60.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc5_60.2 (constants.%specific_impl_fn.e2a)]
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
// CHECK:STDOUT: specific @fn(constants.%Copy.facet.1e1) {
// CHECK:STDOUT:   %T.patt.loc5_27.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_27.1 => constants.%Copy.facet.1e1
// CHECK:STDOUT:   %T.as_type.loc5_43.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_41.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_41.2 => constants.%x.patt.e3b
// CHECK:STDOUT:   %.loc5_49.2 => constants.%.795
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc5_46.2 => constants.%return.patt.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.381
// CHECK:STDOUT:   %.loc5_60.3 => constants.%.737
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.b51
// CHECK:STDOUT:   %impl.elem0.loc5_60.2 => constants.%Int.as.Copy.impl.Op.4f6
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
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d02
// CHECK:STDOUT:   %.loc5_60.3 => constants.%.e28
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
// CHECK:STDOUT:   %fn.type.c16: type = fn_type @fn.loc5 [concrete]
// CHECK:STDOUT:   %fn.415: %fn.type.c16 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %y.param_patt.ec0: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.d6d: %pattern_type.2c66b4.2 = wrapper_binding_pattern y, %y.param_patt.ec0 [symbolic]
// CHECK:STDOUT:   %fn.type.d07: type = fn_type @fn.loc6, @fn.loc5(%T.f84) [symbolic]
// CHECK:STDOUT:   %fn.328: %fn.type.d07 = struct_value () [symbolic]
// CHECK:STDOUT:   %z.param_patt.b42: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %z.patt.513: %pattern_type.2c66b4.2 = wrapper_binding_pattern z, %z.param_patt.b42 [symbolic]
// CHECK:STDOUT:   %fn.type.f76: type = fn_type @fn.loc7, @fn.loc6(%T.f84) [symbolic]
// CHECK:STDOUT:   %fn.04b: %fn.type.f76 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %fn.specific_fn.76d: <specific function> = specific_function %fn.04b, @fn.loc7(%T.f84) [symbolic]
// CHECK:STDOUT:   %fn.specific_fn.f79: <specific function> = specific_function %fn.328, @fn.loc6(%T.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.6b6 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.155: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e7f: <witness> = impl_witness imports.%As.impl_witness_table.d22, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e79: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8a4: %Core.IntLiteral.as.As.impl.Convert.type.e79 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.155 = facet_value Core.IntLiteral, (%As.impl_witness.e7f) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d02: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.453: type = fn_type_with_self_type %As.WithSelf.Convert.type.d02, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.8a4 [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.8a4, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.ac4: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.1e1: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %x.param_patt.cee: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e3b: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt.cee [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.eb8: <specific function> = specific_function %fn.415, @fn.loc5(%Copy.facet.1e1) [concrete]
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
// CHECK:STDOUT:   %fn.specific_fn.1a6: <specific function> = specific_function %fn.415, @fn.loc5(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %fn.type.a2a: type = fn_type @fn.loc6, @fn.loc5(%Copy.facet.1e1) [concrete]
// CHECK:STDOUT:   %fn.62c: %fn.type.a2a = struct_value () [concrete]
// CHECK:STDOUT:   %y.param_patt.673: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.547: %pattern_type.6b6 = wrapper_binding_pattern y, %y.param_patt.673 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.b67: <specific function> = specific_function %fn.62c, @fn.loc6(%Copy.facet.1e1) [concrete]
// CHECK:STDOUT:   %complete_type.127: <witness> = complete_type_witness bool [concrete]
// CHECK:STDOUT:   %fn.type.356: type = fn_type @fn.loc6, @fn.loc5(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %fn.025: %fn.type.356 = struct_value () [concrete]
// CHECK:STDOUT:   %y.param_patt.077: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.252: %pattern_type.831 = wrapper_binding_pattern y, %y.param_patt.077 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.7c8: <specific function> = specific_function %fn.025, @fn.loc6(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %fn.type.ec9: type = fn_type @fn.loc7, @fn.loc6(%Copy.facet.1e1) [concrete]
// CHECK:STDOUT:   %fn.084: %fn.type.ec9 = struct_value () [concrete]
// CHECK:STDOUT:   %z.param_patt.9ae: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %z.patt.45e: %pattern_type.6b6 = wrapper_binding_pattern z, %z.param_patt.9ae [concrete]
// CHECK:STDOUT:   %fn.specific_fn.89f: <specific function> = specific_function %fn.084, @fn.loc7(%Copy.facet.1e1) [concrete]
// CHECK:STDOUT:   %fn.type.48c: type = fn_type @fn.loc7, @fn.loc6(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %fn.129: %fn.type.48c = struct_value () [concrete]
// CHECK:STDOUT:   %z.param_patt.a49: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %z.patt.246: %pattern_type.831 = wrapper_binding_pattern z, %z.param_patt.a49 [concrete]
// CHECK:STDOUT:   %fn.specific_fn.813: <specific function> = specific_function %fn.129, @fn.loc7(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.1e1) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet.1e1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d02: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %.e28: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d02, %Copy.facet.7af [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.d22 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestGenericNested() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn.decl: %fn.type.c16 = fn_decl @fn.loc5 [concrete = constants.%fn.415] {
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
// CHECK:STDOUT:   %generic_id: %fn.type.c16 = alias_binding generic_id, %fn.decl [concrete = constants.%fn.415]
// CHECK:STDOUT:   %generic_id.ref.loc12: %fn.type.c16 = name_ref generic_id, %generic_id [concrete = constants.%fn.415]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %i32.loc12_40: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.453 = impl_witness_access constants.%As.impl_witness.e7f, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.8a4]
// CHECK:STDOUT:   %bound_method.loc12_37.1: <bound method> = bound_method %int_42, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_37.2: <bound method> = bound_method %int_42, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc12_37.2(%int_42) [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc12_37.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc12_37.2: %i32 = converted %int_42, %.loc12_37.1 [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %Copy.facet.loc12_43.1: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.b51) [concrete = constants.%Copy.facet.1e1]
// CHECK:STDOUT:   %.loc12_43.1: %Copy.type = converted constants.%i32, %Copy.facet.loc12_43.1 [concrete = constants.%Copy.facet.1e1]
// CHECK:STDOUT:   %Copy.facet.loc12_43.2: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.b51) [concrete = constants.%Copy.facet.1e1]
// CHECK:STDOUT:   %.loc12_43.2: %Copy.type = converted constants.%i32, %Copy.facet.loc12_43.2 [concrete = constants.%Copy.facet.1e1]
// CHECK:STDOUT:   %fn.specific_fn.loc12: <specific function> = specific_function %generic_id.ref.loc12, @fn.loc5(constants.%Copy.facet.1e1) [concrete = constants.%fn.specific_fn.eb8]
// CHECK:STDOUT:   %fn.call.loc12: init %i32 = call %fn.specific_fn.loc12(%.loc12_37.2)
// CHECK:STDOUT:   %.loc12_43.3: %i32 = value_of_initializer %fn.call.loc12
// CHECK:STDOUT:   %.loc12_43.4: %i32 = converted %fn.call.loc12, %.loc12_43.3
// CHECK:STDOUT:   %i32.loc12_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: %i32 = wrapper_binding c, %.loc12_43.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.6b6 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %generic_id.ref.loc13: %fn.type.c16 = name_ref generic_id, %generic_id [concrete = constants.%fn.415]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %Copy.facet.loc13_39.1: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5e1) [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %.loc13_39.1: %Copy.type = converted bool, %Copy.facet.loc13_39.1 [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %Copy.facet.loc13_39.2: %Copy.type = facet_value bool, (constants.%Copy.impl_witness.5e1) [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %.loc13_39.2: %Copy.type = converted bool, %Copy.facet.loc13_39.2 [concrete = constants.%Copy.facet.7af]
// CHECK:STDOUT:   %fn.specific_fn.loc13: <specific function> = specific_function %generic_id.ref.loc13, @fn.loc5(constants.%Copy.facet.7af) [concrete = constants.%fn.specific_fn.1a6]
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
// CHECK:STDOUT:   %fn.type: type = fn_type @fn.loc6, @fn.loc5(%T.loc5_27.1) [symbolic = %fn.type (constants.%fn.type.d07)]
// CHECK:STDOUT:   %fn: @fn.loc5.%fn.type (%fn.type.d07) = struct_value () [symbolic = %fn (constants.%fn.328)]
// CHECK:STDOUT:   %fn.specific_fn.loc6_26.2: <specific function> = specific_function %fn, @fn.loc6(%T.loc5_27.1) [symbolic = %fn.specific_fn.loc6_26.2 (constants.%fn.specific_fn.f79)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type)) -> out %return.param: @fn.loc5.%T.as_type.loc5_43.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %fn.decl: @fn.loc5.%fn.type (%fn.type.d07) = fn_decl @fn.loc6 [symbolic = @fn.loc5.%fn (constants.%fn.328)] {
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
// CHECK:STDOUT:     %fn.specific_fn.loc6_26.1: <specific function> = specific_function %fn.decl, @fn.loc6(constants.%T.f84) [symbolic = %fn.specific_fn.loc6_26.2 (constants.%fn.specific_fn.f79)]
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
// CHECK:STDOUT:   %fn.type: type = fn_type @fn.loc7, @fn.loc6(%T) [symbolic = %fn.type (constants.%fn.type.f76)]
// CHECK:STDOUT:   %fn: @fn.loc6.%fn.type (%fn.type.f76) = struct_value () [symbolic = %fn (constants.%fn.04b)]
// CHECK:STDOUT:   %fn.specific_fn.loc7_28.2: <specific function> = specific_function %fn, @fn.loc7(%T) [symbolic = %fn.specific_fn.loc7_28.2 (constants.%fn.specific_fn.76d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type)) -> out %return.param: @fn.loc6.%T.as_type.loc6_18.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %fn.decl: @fn.loc6.%fn.type (%fn.type.f76) = fn_decl @fn.loc7 [symbolic = @fn.loc6.%fn (constants.%fn.04b)] {
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
// CHECK:STDOUT:     %fn.specific_fn.loc7_28.1: <specific function> = specific_function %fn.decl, @fn.loc7(constants.%T.f84) [symbolic = %fn.specific_fn.loc7_28.2 (constants.%fn.specific_fn.76d)]
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
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc8_16.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc8_16.3 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc8_16.2: @fn.loc7.%.loc8_16.3 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_16.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_16.2: <specific function> = specific_impl_function %impl.elem0.loc8_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc8_16.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%z.param: @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type)) -> out %return.param: @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %z.ref: @fn.loc7.%T.as_type.loc7_20.1 (%T.as_type) = name_ref z, %z
// CHECK:STDOUT:     %impl.elem0.loc8_16.1: @fn.loc7.%.loc8_16.3 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc8_16.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc8_16.1: <bound method> = bound_method %z.ref, %impl.elem0.loc8_16.1
// CHECK:STDOUT:     %.loc8_16.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc8_16.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_16.1: <specific function> = specific_impl_function %impl.elem0.loc8_16.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc8_16.2 (constants.%specific_impl_fn.e2a)]
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
// CHECK:STDOUT:   %fn.type => constants.%fn.type.f76
// CHECK:STDOUT:   %fn => constants.%fn.04b
// CHECK:STDOUT:   %fn.specific_fn.loc7_28.2 => constants.%fn.specific_fn.76d
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
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.c85f21.2
// CHECK:STDOUT:   %.loc8_16.3 => constants.%.c29
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.d9c
// CHECK:STDOUT:   %impl.elem0.loc8_16.2 => constants.%impl.elem0.bd2
// CHECK:STDOUT:   %specific_impl_fn.loc8_16.2 => constants.%specific_impl_fn.e2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc5(constants.%Copy.facet.1e1) {
// CHECK:STDOUT:   %T.patt.loc5_27.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_27.1 => constants.%Copy.facet.1e1
// CHECK:STDOUT:   %T.as_type.loc5_43.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_41.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_41.2 => constants.%x.patt.e3b
// CHECK:STDOUT:   %.loc5_49.2 => constants.%.795
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc5_46.2 => constants.%return.patt.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %fn.type => constants.%fn.type.a2a
// CHECK:STDOUT:   %fn => constants.%fn.62c
// CHECK:STDOUT:   %fn.specific_fn.loc6_26.2 => constants.%fn.specific_fn.b67
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
// CHECK:STDOUT:   %fn.type => constants.%fn.type.356
// CHECK:STDOUT:   %fn => constants.%fn.025
// CHECK:STDOUT:   %fn.specific_fn.loc6_26.2 => constants.%fn.specific_fn.7c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc6(constants.%Copy.facet.1e1) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.1e1
// CHECK:STDOUT:   %T.as_type.loc6_18.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %y.param_patt.loc6_16.2 => constants.%y.param_patt.673
// CHECK:STDOUT:   %y.patt.loc6_16.2 => constants.%y.patt.547
// CHECK:STDOUT:   %.loc6_24.2 => constants.%.795
// CHECK:STDOUT:   %return.param_patt.loc6_24.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc6_21.2 => constants.%return.patt.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %fn.type => constants.%fn.type.ec9
// CHECK:STDOUT:   %fn => constants.%fn.084
// CHECK:STDOUT:   %fn.specific_fn.loc7_28.2 => constants.%fn.specific_fn.89f
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
// CHECK:STDOUT:   %fn.type => constants.%fn.type.48c
// CHECK:STDOUT:   %fn => constants.%fn.129
// CHECK:STDOUT:   %fn.specific_fn.loc7_28.2 => constants.%fn.specific_fn.813
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @fn.loc7(constants.%Copy.facet.1e1) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.1e1
// CHECK:STDOUT:   %T.as_type.loc7_20.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %z.param_patt.loc7_18.2 => constants.%z.param_patt.9ae
// CHECK:STDOUT:   %z.patt.loc7_18.2 => constants.%z.patt.45e
// CHECK:STDOUT:   %.loc7_26.2 => constants.%.795
// CHECK:STDOUT:   %return.param_patt.loc7_26.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc7_23.2 => constants.%return.patt.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.381
// CHECK:STDOUT:   %.loc8_16.3 => constants.%.737
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.b51
// CHECK:STDOUT:   %impl.elem0.loc8_16.2 => constants.%Int.as.Copy.impl.Op.4f6
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
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d02
// CHECK:STDOUT:   %.loc8_16.3 => constants.%.e28
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
