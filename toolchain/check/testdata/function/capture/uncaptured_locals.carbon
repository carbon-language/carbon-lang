// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/capture/uncaptured_locals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/capture/uncaptured_locals.carbon

// --- fail_reference_uncaptured.carbon
library "[[@TEST_NAME]]";

class X {}

fn Test() {
  let x: X = {};
  fn Inner() {
    // CHECK:STDERR: fail_reference_uncaptured.carbon:[[@LINE+3]]:23: error: cannot reference local `x` declared in enclosing scope [UncapturedLocal]
    // CHECK:STDERR: fail_reference_uncaptured.carbon:[[@LINE-3]]:7-10: note: declared here [UncapturedLocalDecl]
    // CHECK:STDERR:
    let unused y: X = x;
  }
}

// --- fail_reference_uncaptured_var_and_param.carbon
library "[[@TEST_NAME]]";

fn F(p: i32) {
  var n: i32;
  fn G() -> i32 {
    // CHECK:STDERR: fail_reference_uncaptured_var_and_param.carbon:[[@LINE+3]]:12: error: cannot reference local `n` declared in enclosing scope [UncapturedLocal]
    // CHECK:STDERR: fail_reference_uncaptured_var_and_param.carbon:[[@LINE-3]]:7-12: note: declared here [UncapturedLocalDecl]
    // CHECK:STDERR:
    return n;
  }
  fn H() -> i32 {
    // CHECK:STDERR: fail_reference_uncaptured_var_and_param.carbon:[[@LINE+3]]:12: error: cannot reference local `p` declared in enclosing scope [UncapturedLocal]
    // CHECK:STDERR: fail_reference_uncaptured_var_and_param.carbon:[[@LINE-10]]:6-11: note: declared here [UncapturedLocalDecl]
    // CHECK:STDERR:
    return p;
  }
}

// --- fail_reference_uncaptured_lambda.carbon
library "[[@TEST_NAME]]";

class X {}

fn Test() {
  let x: X = {};
  (fn() {
    // CHECK:STDERR: fail_reference_uncaptured_lambda.carbon:[[@LINE+3]]:23: error: cannot reference local `x` declared in enclosing scope [UncapturedLocal]
    // CHECK:STDERR: fail_reference_uncaptured_lambda.carbon:[[@LINE-3]]:7-10: note: declared here [UncapturedLocalDecl]
    // CHECK:STDERR:
    let unused y: X = x;
  });
}

// --- fail_reference_uncaptured_local_class.carbon
library "[[@TEST_NAME]]";

fn F() {
  var n: i32 = 0;
  let m: i32 = 1;
  class Local {
    // CHECK:STDERR: fail_reference_uncaptured_local_class.carbon:[[@LINE+3]]:22: error: cannot reference local `n` declared in enclosing scope [UncapturedLocal]
    // CHECK:STDERR: fail_reference_uncaptured_local_class.carbon:[[@LINE-4]]:7-12: note: declared here [UncapturedLocalDecl]
    // CHECK:STDERR:
    var field: i32 = n;
    // CHECK:STDERR: fail_reference_uncaptured_local_class.carbon:[[@LINE+3]]:27: error: cannot reference local `m` declared in enclosing scope [UncapturedLocal]
    // CHECK:STDERR: fail_reference_uncaptured_local_class.carbon:[[@LINE-7]]:7-12: note: declared here [UncapturedLocalDecl]
    // CHECK:STDERR:
    let member_let: i32 = m;
    fn Method() -> i32 {
      // CHECK:STDERR: fail_reference_uncaptured_local_class.carbon:[[@LINE+3]]:14: error: cannot reference local `n` declared in enclosing scope [UncapturedLocal]
      // CHECK:STDERR: fail_reference_uncaptured_local_class.carbon:[[@LINE-13]]:7-12: note: declared here [UncapturedLocalDecl]
      // CHECK:STDERR:
      return n;
    }
  }
}

// --- fail_reference_uncaptured_local_interface_and_impl.carbon
library "[[@TEST_NAME]]";

class C {}

fn F() {
  var n: i32 = 0;
  interface LocalIface {
    // CHECK:STDERR: fail_reference_uncaptured_local_interface_and_impl.carbon:[[@LINE+3]]:18: error: cannot reference local `n` declared in enclosing scope [UncapturedLocal]
    // CHECK:STDERR: fail_reference_uncaptured_local_interface_and_impl.carbon:[[@LINE-3]]:7-12: note: declared here [UncapturedLocalDecl]
    // CHECK:STDERR:
    let X: i32 = n;
  }
  interface OtherIface {
    let Y: i32;
    fn Get(self) -> i32;
  }
  // CHECK:STDERR: fail_reference_uncaptured_local_interface_and_impl.carbon:[[@LINE+3]]:3-24: error: associated constant Y not given a value in impl of interface OtherIface [ImplAssociatedConstantNeedsValue]
  // CHECK:STDERR: fail_reference_uncaptured_local_interface_and_impl.carbon:[[@LINE-4]]:9-14: note: associated constant declared here [AssociatedConstantHere]
  // CHECK:STDERR:
  impl C as OtherIface {
    // CHECK:STDERR: fail_reference_uncaptured_local_interface_and_impl.carbon:[[@LINE+3]]:18: error: cannot reference local `n` declared in enclosing scope [UncapturedLocal]
    // CHECK:STDERR: fail_reference_uncaptured_local_interface_and_impl.carbon:[[@LINE-16]]:7-12: note: declared here [UncapturedLocalDecl]
    // CHECK:STDERR:
    let Y: i32 = n;
    fn Get(unused self) -> i32 {
      // CHECK:STDERR: fail_reference_uncaptured_local_interface_and_impl.carbon:[[@LINE+3]]:14: error: cannot reference local `n` declared in enclosing scope [UncapturedLocal]
      // CHECK:STDERR: fail_reference_uncaptured_local_interface_and_impl.carbon:[[@LINE-21]]:7-12: note: declared here [UncapturedLocalDecl]
      // CHECK:STDERR:
      return n;
    }
  }
}

// --- fail_reference_uncaptured_qualified_alias.carbon
library "[[@TEST_NAME]]";

fn F() {
  var x: i32 = 0;
  // CHECK:STDERR: fail_reference_uncaptured_qualified_alias.carbon:[[@LINE+2]]:3-14: error: `namespace` declaration not at top level [NamespaceDeclNotAtTopLevel]
  // CHECK:STDERR:
  namespace N;
  // CHECK:STDERR: fail_reference_uncaptured_qualified_alias.carbon:[[@LINE+3]]:15: error: cannot reference local `x` declared in enclosing scope [UncapturedLocal]
  // CHECK:STDERR: fail_reference_uncaptured_qualified_alias.carbon:[[@LINE-5]]:7-12: note: declared here [UncapturedLocalDecl]
  // CHECK:STDERR:
  alias N.a = x;
}

// --- fail_unqualified_alias_still_constant_error.carbon
library "[[@TEST_NAME]]";

fn F() -> i32 {
  var x: i32 = 0;
  // CHECK:STDERR: fail_unqualified_alias_still_constant_error.carbon:[[@LINE+2]]:13: error: alias refers to a runtime value [AliasRequiresConstantValue]
  // CHECK:STDERR:
  alias a = x;
  return a;
}

// --- fail_typeof_nested_lambda.carbon
library "[[@TEST_NAME]]";

fn F() {
  var n: i32 = 0;
  // CHECK:STDERR: fail_typeof_nested_lambda.carbon:[[@LINE+3]]:46: error: cannot reference local `n` declared in enclosing scope [UncapturedLocal]
  // CHECK:STDERR: fail_typeof_nested_lambda.carbon:[[@LINE-2]]:7-12: note: declared here [UncapturedLocalDecl]
  // CHECK:STDERR:
  let unused x: typeof((fn() -> i32 { return n; })()) = 0;
}

// --- allowed_scopes.carbon
library "[[@TEST_NAME]]";

var global_var: i32 = 1;
let global_let: i32 = 2;

fn Outer(generic T: type, p: i32) -> i32 {
  var local_var: i32 = p;
  let local_let: i32 = global_let;
  alias const_alias = 3;

  if (true) {
    local_var = local_let;
  }

  while (false) {
    local_var = p;
  }

  if (false) {
    let inner_block: i32 = local_var;
    local_var = inner_block;
  }

  class LocalClass {}
  fn LocalHelper() -> i32 {
    return global_var;
  }

  //@dump-sem-ir-begin
  fn Nested() -> typeof(local_var) {
    let unused t_type: type = T;
    var unused c_val: LocalClass = {};
    let unused a: typeof(local_let) = LocalHelper();
    let b: typeof(p) = global_let;
    let unused d: i32 = const_alias;
    return b;
  }
  //@dump-sem-ir-end

  return Nested();
}

// CHECK:STDOUT: --- allowed_scopes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type type [concrete]
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
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
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
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %LocalClass: type = class_type @LocalClass, @LocalClass(%T.67d) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %LocalHelper.type: type = fn_type @LocalHelper, @Outer(%T.67d) [symbolic]
// CHECK:STDOUT:   %LocalHelper: %LocalHelper.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Nested.type: type = fn_type @Nested, @Outer(%T.67d) [symbolic]
// CHECK:STDOUT:   %Nested: %Nested.type = struct_value () [symbolic]
// CHECK:STDOUT:   %t_type.patt: %pattern_type.9a587c.1 = value_binding_pattern t_type [concrete]
// CHECK:STDOUT:   %require_complete.9cd: <witness> = require_complete_type %LocalClass [symbolic]
// CHECK:STDOUT:   %pattern_type.16b: type = pattern_type %LocalClass [symbolic]
// CHECK:STDOUT:   %c_val.patt: %pattern_type.16b = ref_binding_pattern c_val [symbolic]
// CHECK:STDOUT:   %c_val.var_patt: %pattern_type.16b = var_pattern %c_val.patt [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %LocalClass.val: %LocalClass = struct_value () [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %LocalHelper.specific_fn: <specific function> = specific_function %LocalHelper, @LocalHelper(%T.67d) [symbolic]
// CHECK:STDOUT:   %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.d94 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.5c4: <witness> = lookup_impl_witness %LocalClass, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.91b: <witness> = lookup_impl_witness %LocalClass, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.274: %SubobjectDestroy.type = facet_value %LocalClass, (%SubobjectDestroy.lookup_impl_witness.91b) [symbolic]
// CHECK:STDOUT:   %.88a: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.274) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.50c: %Destroy.type = facet_value %LocalClass, (%DestroyImpl.lookup_impl_witness.5c4, %SubobjectDestroy.lookup_impl_witness.91b) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.b99: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.50c) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer(%T.loc6_19.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Nested.type: type = fn_type @Nested, @Outer(%T.loc6_19.1) [symbolic = %Nested.type (constants.%Nested.type)]
// CHECK:STDOUT:   %Nested: @Outer.%Nested.type (%Nested.type) = struct_value () [symbolic = %Nested (constants.%Nested)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then.loc11:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else.loc11:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !while.cond:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !while.body:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !while.done:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.then.loc19:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !if.else.loc19:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Nested.decl: @Outer.%Nested.type (%Nested.type) = fn_decl @Nested [symbolic = @Outer.%Nested (constants.%Nested)] {
// CHECK:STDOUT:       %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:       %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %.loc30_34.1 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc30_34.1: type = type_of %local_var.ref in [concrete = constants.%i32] {
// CHECK:STDOUT:         %local_var.ref: ref %i32 = name_ref local_var, @Outer.%local_var
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %.loc30_34.2: Core.Form = init_form %.loc30_34.1 [concrete = constants.%.0b3]
// CHECK:STDOUT:       %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:       %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Nested(@Outer.%T.loc6_19.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %LocalClass: type = class_type @LocalClass, @LocalClass(%T) [symbolic = %LocalClass (constants.%LocalClass)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %LocalClass [symbolic = %require_complete (constants.%require_complete.9cd)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %LocalClass [symbolic = %pattern_type (constants.%pattern_type.16b)]
// CHECK:STDOUT:   %c_val.patt.loc32_21.2: @Nested.%pattern_type (%pattern_type.16b) = ref_binding_pattern c_val [symbolic = %c_val.patt.loc32_21.2 (constants.%c_val.patt)]
// CHECK:STDOUT:   %c_val.var_patt.loc32_5.2: @Nested.%pattern_type (%pattern_type.16b) = var_pattern %c_val.patt.loc32_21.2 [symbolic = %c_val.var_patt.loc32_5.2 (constants.%c_val.var_patt)]
// CHECK:STDOUT:   %LocalClass.val: @Nested.%LocalClass (%LocalClass) = struct_value () [symbolic = %LocalClass.val (constants.%LocalClass.val)]
// CHECK:STDOUT:   %LocalHelper.type: type = fn_type @LocalHelper, @Outer(%T) [symbolic = %LocalHelper.type (constants.%LocalHelper.type)]
// CHECK:STDOUT:   %LocalHelper: @Nested.%LocalHelper.type (%LocalHelper.type) = struct_value () [symbolic = %LocalHelper (constants.%LocalHelper)]
// CHECK:STDOUT:   %LocalHelper.specific_fn.loc33_39.2: <specific function> = specific_function %LocalHelper, @LocalHelper(%T) [symbolic = %LocalHelper.specific_fn.loc33_39.2 (constants.%LocalHelper.specific_fn)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %LocalClass, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.91b)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %LocalClass, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.274)]
// CHECK:STDOUT:   %.loc32_5.4: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc32_5.4 (constants.%.88a)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %LocalClass, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.5c4)]
// CHECK:STDOUT:   %Destroy.facet.loc32_5.3: %Destroy.type = facet_value %LocalClass, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc32_5.3 (constants.%Destroy.facet.50c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc31_31: type = specific_constant @Outer.%T.loc6_19.2, @Outer(constants.%T.67d) [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %.loc31_31 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:     %.loc31_24: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %t_type: type = wrapper_binding t_type, %T.ref
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %t_type.patt: %pattern_type.9a587c.1 = value_binding_pattern t_type [concrete = constants.%t_type.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c_val.var: ref @Nested.%LocalClass (%LocalClass) = var_storage %c_val.var_patt.loc32_5.1
// CHECK:STDOUT:     %.loc32_37.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc32_37.2: init @Nested.%LocalClass (%LocalClass) to %c_val.var = class_init () [symbolic = %LocalClass.val (constants.%LocalClass.val)]
// CHECK:STDOUT:     %.loc32_5.1: init @Nested.%LocalClass (%LocalClass) = converted %.loc32_37.1, %.loc32_37.2 [symbolic = %LocalClass.val (constants.%LocalClass.val)]
// CHECK:STDOUT:     assign %c_val.var, %.loc32_5.1
// CHECK:STDOUT:     %.loc32_23.1: type = splice_block %LocalClass.ref [symbolic = %LocalClass (constants.%LocalClass)] {
// CHECK:STDOUT:       %.loc32_23.2: type = specific_constant @Outer.%LocalClass.decl, @Outer(constants.%T.67d) [symbolic = %LocalClass (constants.%LocalClass)]
// CHECK:STDOUT:       %LocalClass.ref: type = name_ref LocalClass, %.loc32_23.2 [symbolic = %LocalClass (constants.%LocalClass)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c_val: ref @Nested.%LocalClass (%LocalClass) = wrapper_binding c_val, %c_val.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c_val.patt.loc32_21.1: @Nested.%pattern_type (%pattern_type.16b) = ref_binding_pattern c_val [symbolic = %c_val.patt.loc32_21.2 (constants.%c_val.patt)]
// CHECK:STDOUT:       %c_val.var_patt.loc32_5.1: @Nested.%pattern_type (%pattern_type.16b) = var_pattern %c_val.patt.loc32_21.1 [symbolic = %c_val.var_patt.loc32_5.2 (constants.%c_val.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc33_39: @Nested.%LocalHelper.type (%LocalHelper.type) = specific_constant @Outer.%LocalHelper.decl, @Outer(constants.%T.67d) [symbolic = %LocalHelper (constants.%LocalHelper)]
// CHECK:STDOUT:     %LocalHelper.ref: @Nested.%LocalHelper.type (%LocalHelper.type) = name_ref LocalHelper, %.loc33_39 [symbolic = %LocalHelper (constants.%LocalHelper)]
// CHECK:STDOUT:     %LocalHelper.specific_fn.loc33_39.1: <specific function> = specific_function %LocalHelper.ref, @LocalHelper(constants.%T.67d) [symbolic = %LocalHelper.specific_fn.loc33_39.2 (constants.%LocalHelper.specific_fn)]
// CHECK:STDOUT:     %LocalHelper.call: init %i32 = call %LocalHelper.specific_fn.loc33_39.1()
// CHECK:STDOUT:     %.loc33_51.1: %i32 = value_of_initializer %LocalHelper.call
// CHECK:STDOUT:     %.loc33_51.2: %i32 = converted %LocalHelper.call, %.loc33_51.1
// CHECK:STDOUT:     %.loc33_35: type = type_of %local_let.ref in [concrete = constants.%i32] {
// CHECK:STDOUT:       %local_let.ref: %i32 = name_ref local_let, @Outer.%local_let
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = wrapper_binding a, %.loc33_51.2
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %global_let.global_ref: %i32 = global_ref file.%global_let
// CHECK:STDOUT:     %global_let.ref: %i32 = name_ref global_let, %global_let.global_ref
// CHECK:STDOUT:     %.loc34: type = type_of %p.ref in [concrete = constants.%i32] {
// CHECK:STDOUT:       %p.ref: %i32 = name_ref p, @Outer.%p
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = wrapper_binding b, %global_let.ref
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %const_alias.ref: Core.IntLiteral = name_ref const_alias, @Outer.%const_alias [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0.loc35: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc35_25.1: <bound method> = bound_method %const_alias.ref, %impl.elem0.loc35 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:     %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc35_25.2: <bound method> = bound_method %const_alias.ref, %specific_fn.loc35 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc35_25.2(%const_alias.ref) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %.loc35_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %.loc35_25.2: %i32 = converted %const_alias.ref, %.loc35_25.1 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %d: %i32 = wrapper_binding d, %.loc35_25.2
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %d.patt: %pattern_type.d94 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:     %impl.elem0.loc36: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:     %bound_method.loc36_12.1: <bound method> = bound_method %b.ref, %impl.elem0.loc36
// CHECK:STDOUT:     %specific_fn.loc36: <specific function> = specific_function %impl.elem0.loc36, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc36_12.2: <bound method> = bound_method %b.ref, %specific_fn.loc36
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc36_12.2(%b.ref)
// CHECK:STDOUT:     %Destroy.facet.loc32_5.1: %Destroy.type = facet_value constants.%LocalClass, (constants.%DestroyImpl.lookup_impl_witness.5c4, constants.%SubobjectDestroy.lookup_impl_witness.91b) [symbolic = %Destroy.facet.loc32_5.3 (constants.%Destroy.facet.50c)]
// CHECK:STDOUT:     %.loc32_5.2: %Destroy.type = converted constants.%LocalClass, %Destroy.facet.loc32_5.1 [symbolic = %Destroy.facet.loc32_5.3 (constants.%Destroy.facet.50c)]
// CHECK:STDOUT:     %Destroy.facet.loc32_5.2: %Destroy.type = facet_value constants.%LocalClass, (constants.%DestroyImpl.lookup_impl_witness.5c4, constants.%SubobjectDestroy.lookup_impl_witness.91b) [symbolic = %Destroy.facet.loc32_5.3 (constants.%Destroy.facet.50c)]
// CHECK:STDOUT:     %.loc32_5.3: %Destroy.type = converted constants.%LocalClass, %Destroy.facet.loc32_5.2 [symbolic = %Destroy.facet.loc32_5.3 (constants.%Destroy.facet.50c)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%c_val.var)
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc32(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc6_19.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc6_19.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Nested(constants.%T.67d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %LocalClass => constants.%LocalClass
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9cd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.16b
// CHECK:STDOUT:   %c_val.patt.loc32_21.2 => constants.%c_val.patt
// CHECK:STDOUT:   %c_val.var_patt.loc32_5.2 => constants.%c_val.var_patt
// CHECK:STDOUT:   %LocalClass.val => constants.%LocalClass.val
// CHECK:STDOUT:   %LocalHelper.type => constants.%LocalHelper.type
// CHECK:STDOUT:   %LocalHelper => constants.%LocalHelper
// CHECK:STDOUT:   %LocalHelper.specific_fn.loc33_39.2 => constants.%LocalHelper.specific_fn
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness => constants.%SubobjectDestroy.lookup_impl_witness.91b
// CHECK:STDOUT:   %SubobjectDestroy.facet => constants.%SubobjectDestroy.facet.274
// CHECK:STDOUT:   %.loc32_5.4 => constants.%.88a
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness => constants.%DestroyImpl.lookup_impl_witness.5c4
// CHECK:STDOUT:   %Destroy.facet.loc32_5.3 => constants.%Destroy.facet.50c
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.b99
// CHECK:STDOUT: }
// CHECK:STDOUT:
