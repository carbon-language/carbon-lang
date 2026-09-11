// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/destroy_control_flow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/destroy_control_flow.carbon

// --- destructible.carbon

library "[[@TEST_NAME]]";

class Destructible {
  // TODO: Implement Core.Destructor instead.
  impl as Core.Destroy {
    fn Op(self);
  }
};

fn Use(ref d: Destructible);

// --- if_then.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn IfThen(cond: bool) {
  //@dump-sem-ir-begin
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
  }
  Use(ref a);
  //@dump-sem-ir-end
}

// --- if_else.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn IfElse(cond: bool) {
  //@dump-sem-ir-begin
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
  } else {
    var c: Destructible = {};
    Use(ref c);
  }
  Use(ref a);
  //@dump-sem-ir-end
}

// --- if_return.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn IfReturn(cond: bool) {
  //@dump-sem-ir-begin
  var a: Destructible = {};
  Use(ref a);
  if (cond) {
    var b: Destructible = {};
    Use(ref b);
    return;
  }
  //@dump-sem-ir-end
}

// --- while_break.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn WhileBreak(cond: bool) {
  //@dump-sem-ir-begin
  var a: Destructible = {};
  Use(ref a);
  while (cond) {
    Use(ref a);
    var b: Destructible = {};
    Use(ref b);
    if (cond) {
      var c: Destructible = {};
      Use(ref c);
      break;
    }
  }
  Use(ref a);
  //@dump-sem-ir-end
}

// --- while_continue.carbon

library "[[@TEST_NAME]]";
import library "destructible";

fn WhileContinue(cond: bool) {
  //@dump-sem-ir-begin
  var a: Destructible = {};
  Use(ref a);
  while (cond) {
    Use(ref a);
    var b: Destructible = {};
    Use(ref b);
    if (cond) {
      var c: Destructible = {};
      Use(ref c);
      continue;
    }
  }
  Use(ref a);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- if_then.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destructible: type = class_type @Destructible [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.397: type = pattern_type %Destructible [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.val: %Destructible = struct_value () [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.type.a1d3ab.2: type = fn_type @Destroy.SubobjectDestroy.loc10_5.2 [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.ab82fa.2: %Destroy.SubobjectDestroy.type.a1d3ab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.type.facet: %type = facet_value %Destructible, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.94b: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destructible.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.162: %Destroy.WithSelf.SelfDestruct.type.94b = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_5.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.923: <witness> = custom_witness (%Destroy.Op.1a2547.2, %Destroy.SubobjectDestroy.ab82fa.2, %Destroy.WithSelf.SelfDestruct.162), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.fcf: %Destroy.type = facet_value %Destructible, (%custom_witness.923) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Destructible: type = import_ref Main//destructible, Destructible, loaded [concrete = constants.%Destructible]
// CHECK:STDOUT:   %Main.Use: %Use.type = import_ref Main//destructible, Use, loaded [concrete = constants.%Use]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IfThen(%cond.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Destructible = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc7_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_26.2: init %Destructible to %a.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc7_3.1: init %Destructible = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_3.1
// CHECK:STDOUT:   %Destructible.ref.loc7: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %a: ref %Destructible = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc8: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc8: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc8: %Destructible = ref_tag %a.ref.loc8
// CHECK:STDOUT:   %Use.call.loc8: init %empty_tuple.type = call %Use.ref.loc8(%a.ref.loc8)
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %b.var: ref %Destructible = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc10_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_28.2: init %Destructible to %b.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc10_5.1: init %Destructible = converted %.loc10_28.1, %.loc10_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %b.var, %.loc10_5.1
// CHECK:STDOUT:   %Destructible.ref.loc10: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %b: ref %Destructible = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc11: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %b.ref: ref %Destructible = name_ref b, %b
// CHECK:STDOUT:   %.loc11: %Destructible = ref_tag %b.ref
// CHECK:STDOUT:   %Use.call.loc11: init %empty_tuple.type = call %Use.ref.loc11(%b.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc10_5.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc10_5.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc10_5.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc10_5.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc10_5.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc10_5.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %b.var, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%b.var)
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %Use.ref.loc13: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc13: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc13: %Destructible = ref_tag %a.ref.loc13
// CHECK:STDOUT:   %Use.call.loc13: init %empty_tuple.type = call %Use.ref.loc13(%a.ref.loc13)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %a.var, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7: init %empty_tuple.type = call %bound_method.loc7(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc10_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc10_5.2(%self.param: ref %Destructible) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_5.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- if_else.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destructible: type = class_type @Destructible [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.397: type = pattern_type %Destructible [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.val: %Destructible = struct_value () [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.type.a1d3ab.2: type = fn_type @Destroy.SubobjectDestroy.loc10_5.2 [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.ab82fa.2: %Destroy.SubobjectDestroy.type.a1d3ab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.type.facet: %type = facet_value %Destructible, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.94b: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destructible.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.162: %Destroy.WithSelf.SelfDestruct.type.94b = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_5.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.923: <witness> = custom_witness (%Destroy.Op.1a2547.2, %Destroy.SubobjectDestroy.ab82fa.2, %Destroy.WithSelf.SelfDestruct.162), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.fcf: %Destroy.type = facet_value %Destructible, (%custom_witness.923) [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.397 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.397 = var_pattern %c.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Destructible: type = import_ref Main//destructible, Destructible, loaded [concrete = constants.%Destructible]
// CHECK:STDOUT:   %Main.Use: %Use.type = import_ref Main//destructible, Use, loaded [concrete = constants.%Use]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IfElse(%cond.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Destructible = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc7_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_26.2: init %Destructible to %a.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc7_3.1: init %Destructible = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_3.1
// CHECK:STDOUT:   %Destructible.ref.loc7: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %a: ref %Destructible = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc8: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc8: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc8: %Destructible = ref_tag %a.ref.loc8
// CHECK:STDOUT:   %Use.call.loc8: init %empty_tuple.type = call %Use.ref.loc8(%a.ref.loc8)
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %b.var: ref %Destructible = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc10_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_28.2: init %Destructible to %b.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc10_5.1: init %Destructible = converted %.loc10_28.1, %.loc10_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %b.var, %.loc10_5.1
// CHECK:STDOUT:   %Destructible.ref.loc10: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %b: ref %Destructible = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc11: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %b.ref: ref %Destructible = name_ref b, %b
// CHECK:STDOUT:   %.loc11: %Destructible = ref_tag %b.ref
// CHECK:STDOUT:   %Use.call.loc11: init %empty_tuple.type = call %Use.ref.loc11(%b.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc10_5.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc10_5.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc10_5.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc10_5.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc10_5.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc10_5.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %b.var, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%b.var)
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %c.var: ref %Destructible = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc13_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc13_28.2: init %Destructible to %c.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc13_5.1: init %Destructible = converted %.loc13_28.1, %.loc13_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %c.var, %.loc13_5.1
// CHECK:STDOUT:   %Destructible.ref.loc13: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %c: ref %Destructible = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.397 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.397 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc14: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %c.ref: ref %Destructible = name_ref c, %c
// CHECK:STDOUT:   %.loc14: %Destructible = ref_tag %c.ref
// CHECK:STDOUT:   %Use.call.loc14: init %empty_tuple.type = call %Use.ref.loc14(%c.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc13: <bound method> = bound_method %c.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc13_5.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc13_5.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc13_5.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc13_5.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc13_5.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc13_5.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %c.var, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc13: init %empty_tuple.type = call %bound_method.loc13(%c.var)
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   %Use.ref.loc16: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc16: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc16: %Destructible = ref_tag %a.ref.loc16
// CHECK:STDOUT:   %Use.call.loc16: init %empty_tuple.type = call %Use.ref.loc16(%a.ref.loc16)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %a.var, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7: init %empty_tuple.type = call %bound_method.loc7(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc10_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc10_5.2(%self.param: ref %Destructible) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_5.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- if_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destructible: type = class_type @Destructible [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.397: type = pattern_type %Destructible [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.val: %Destructible = struct_value () [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.type.a1d3ab.2: type = fn_type @Destroy.SubobjectDestroy.loc10_5.2 [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.ab82fa.2: %Destroy.SubobjectDestroy.type.a1d3ab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.type.facet: %type = facet_value %Destructible, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.94b: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destructible.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.162: %Destroy.WithSelf.SelfDestruct.type.94b = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_5.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.923: <witness> = custom_witness (%Destroy.Op.1a2547.2, %Destroy.SubobjectDestroy.ab82fa.2, %Destroy.WithSelf.SelfDestruct.162), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.fcf: %Destroy.type = facet_value %Destructible, (%custom_witness.923) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Destructible: type = import_ref Main//destructible, Destructible, loaded [concrete = constants.%Destructible]
// CHECK:STDOUT:   %Main.Use: %Use.type = import_ref Main//destructible, Use, loaded [concrete = constants.%Use]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IfReturn(%cond.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Destructible = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc7_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_26.2: init %Destructible to %a.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc7_3.1: init %Destructible = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_3.1
// CHECK:STDOUT:   %Destructible.ref.loc7: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %a: ref %Destructible = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc8: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc8: %Destructible = ref_tag %a.ref
// CHECK:STDOUT:   %Use.call.loc8: init %empty_tuple.type = call %Use.ref.loc8(%a.ref)
// CHECK:STDOUT:   %cond.ref: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %b.var: ref %Destructible = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc10_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_28.2: init %Destructible to %b.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc10_5.1: init %Destructible = converted %.loc10_28.1, %.loc10_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %b.var, %.loc10_5.1
// CHECK:STDOUT:   %Destructible.ref.loc10: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %b: ref %Destructible = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc11: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %b.ref: ref %Destructible = name_ref b, %b
// CHECK:STDOUT:   %.loc11: %Destructible = ref_tag %b.ref
// CHECK:STDOUT:   %Use.call.loc11: init %empty_tuple.type = call %Use.ref.loc11(%b.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc10: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc10_5.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc10_5.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc10_5.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc10_5.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc10_5.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc10_5.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %b.var, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10: init %empty_tuple.type = call %bound_method.loc10(%b.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7_3.1: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc7_3.1: <bound method> = bound_method %a.var, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7_3.1: init %empty_tuple.type = call %bound_method.loc7_3.1(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7_3.2: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc7_3.3: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.4: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.3 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.4: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.5: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.4 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc7_3.2: <bound method> = bound_method %a.var, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7_3.2: init %empty_tuple.type = call %bound_method.loc7_3.2(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc10_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc10_5.2(%self.param: ref %Destructible) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_5.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- while_break.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destructible: type = class_type @Destructible [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.397: type = pattern_type %Destructible [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.val: %Destructible = struct_value () [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.397 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.397 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.type.a1d3ab.2: type = fn_type @Destroy.SubobjectDestroy.loc14_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.ab82fa.2: %Destroy.SubobjectDestroy.type.a1d3ab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.type.facet: %type = facet_value %Destructible, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.94b: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destructible.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.162: %Destroy.WithSelf.SelfDestruct.type.94b = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc14_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.923: <witness> = custom_witness (%Destroy.Op.1a2547.2, %Destroy.SubobjectDestroy.ab82fa.2, %Destroy.WithSelf.SelfDestruct.162), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.fcf: %Destroy.type = facet_value %Destructible, (%custom_witness.923) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Destructible: type = import_ref Main//destructible, Destructible, loaded [concrete = constants.%Destructible]
// CHECK:STDOUT:   %Main.Use: %Use.type = import_ref Main//destructible, Use, loaded [concrete = constants.%Use]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WhileBreak(%cond.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Destructible = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc7_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_26.2: init %Destructible to %a.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc7_3.1: init %Destructible = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_3.1
// CHECK:STDOUT:   %Destructible.ref.loc7: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %a: ref %Destructible = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc8: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc8: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc8: %Destructible = ref_tag %a.ref.loc8
// CHECK:STDOUT:   %Use.call.loc8: init %empty_tuple.type = call %Use.ref.loc8(%a.ref.loc8)
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %cond.ref.loc9: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref.loc9 br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   %Use.ref.loc10: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc10: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc10: %Destructible = ref_tag %a.ref.loc10
// CHECK:STDOUT:   %Use.call.loc10: init %empty_tuple.type = call %Use.ref.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %b.var: ref %Destructible = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc11_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_28.2: init %Destructible to %b.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc11_5.1: init %Destructible = converted %.loc11_28.1, %.loc11_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %b.var, %.loc11_5.1
// CHECK:STDOUT:   %Destructible.ref.loc11: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %b: ref %Destructible = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc12: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %b.ref: ref %Destructible = name_ref b, %b
// CHECK:STDOUT:   %.loc12: %Destructible = ref_tag %b.ref
// CHECK:STDOUT:   %Use.call.loc12: init %empty_tuple.type = call %Use.ref.loc12(%b.ref)
// CHECK:STDOUT:   %cond.ref.loc13: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref.loc13 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %c.var: ref %Destructible = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc14_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_30.2: init %Destructible to %c.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc14_7.1: init %Destructible = converted %.loc14_30.1, %.loc14_30.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %c.var, %.loc14_7.1
// CHECK:STDOUT:   %Destructible.ref.loc14: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %c: ref %Destructible = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.397 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.397 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc15: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %c.ref: ref %Destructible = name_ref c, %c
// CHECK:STDOUT:   %.loc15: %Destructible = ref_tag %c.ref
// CHECK:STDOUT:   %Use.call.loc15: init %empty_tuple.type = call %Use.ref.loc15(%c.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc14: <bound method> = bound_method %c.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc14_7.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc14_7.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc14_7.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc14_7.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc14_7.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc14_7.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %c.var, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc14: init %empty_tuple.type = call %bound_method.loc14(%c.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc11_5.1: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc11_5.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc11_5.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc11_5.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc11_5.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc11_5.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc11_5.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.var, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11_5.1: init %empty_tuple.type = call %bound_method.loc11_5.1(%b.var)
// CHECK:STDOUT:   br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc11_5.2: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc11_5.3: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc11_5.4: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc11_5.3 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc11_5.4: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc11_5.5: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc11_5.4 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.var, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11_5.2: init %empty_tuple.type = call %bound_method.loc11_5.2(%b.var)
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %Use.ref.loc19: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc19: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc19: %Destructible = ref_tag %a.ref.loc19
// CHECK:STDOUT:   %Use.call.loc19: init %empty_tuple.type = call %Use.ref.loc19(%a.ref.loc19)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %a.var, %Destroy.WithSelf.SelfDestruct.specific_fn.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7: init %empty_tuple.type = call %bound_method.loc7(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc14_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc14_7.2(%self.param: ref %Destructible) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_7.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- while_continue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Destructible: type = class_type @Destructible [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.397: type = pattern_type %Destructible [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.val: %Destructible = struct_value () [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.397 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.397 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.type.a1d3ab.2: type = fn_type @Destroy.SubobjectDestroy.loc14_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.SubobjectDestroy.ab82fa.2: %Destroy.SubobjectDestroy.type.a1d3ab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destructible.type.facet: %type = facet_value %Destructible, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.94b: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destructible.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.162: %Destroy.WithSelf.SelfDestruct.type.94b = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc14_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.923: <witness> = custom_witness (%Destroy.Op.1a2547.2, %Destroy.SubobjectDestroy.ab82fa.2, %Destroy.WithSelf.SelfDestruct.162), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.fcf: %Destroy.type = facet_value %Destructible, (%custom_witness.923) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Destructible: type = import_ref Main//destructible, Destructible, loaded [concrete = constants.%Destructible]
// CHECK:STDOUT:   %Main.Use: %Use.type = import_ref Main//destructible, Use, loaded [concrete = constants.%Use]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WhileContinue(%cond.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Destructible = var_storage %a.var_patt
// CHECK:STDOUT:   %.loc7_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_26.2: init %Destructible to %a.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc7_3.1: init %Destructible = converted %.loc7_26.1, %.loc7_26.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %a.var, %.loc7_3.1
// CHECK:STDOUT:   %Destructible.ref.loc7: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %a: ref %Destructible = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.397 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.397 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc8: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc8: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc8: %Destructible = ref_tag %a.ref.loc8
// CHECK:STDOUT:   %Use.call.loc8: init %empty_tuple.type = call %Use.ref.loc8(%a.ref.loc8)
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %cond.ref.loc9: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref.loc9 br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   %Use.ref.loc10: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc10: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc10: %Destructible = ref_tag %a.ref.loc10
// CHECK:STDOUT:   %Use.call.loc10: init %empty_tuple.type = call %Use.ref.loc10(%a.ref.loc10)
// CHECK:STDOUT:   %b.var: ref %Destructible = var_storage %b.var_patt
// CHECK:STDOUT:   %.loc11_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_28.2: init %Destructible to %b.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc11_5.1: init %Destructible = converted %.loc11_28.1, %.loc11_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %b.var, %.loc11_5.1
// CHECK:STDOUT:   %Destructible.ref.loc11: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %b: ref %Destructible = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc12: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %b.ref: ref %Destructible = name_ref b, %b
// CHECK:STDOUT:   %.loc12: %Destructible = ref_tag %b.ref
// CHECK:STDOUT:   %Use.call.loc12: init %empty_tuple.type = call %Use.ref.loc12(%b.ref)
// CHECK:STDOUT:   %cond.ref.loc13: bool = name_ref cond, %cond
// CHECK:STDOUT:   if %cond.ref.loc13 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %c.var: ref %Destructible = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc14_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_30.2: init %Destructible to %c.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc14_7.1: init %Destructible = converted %.loc14_30.1, %.loc14_30.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %c.var, %.loc14_7.1
// CHECK:STDOUT:   %Destructible.ref.loc14: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %c: ref %Destructible = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.397 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.397 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc15: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %c.ref: ref %Destructible = name_ref c, %c
// CHECK:STDOUT:   %.loc15: %Destructible = ref_tag %c.ref
// CHECK:STDOUT:   %Use.call.loc15: init %empty_tuple.type = call %Use.ref.loc15(%c.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc14: <bound method> = bound_method %c.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc14_7.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc14_7.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc14_7.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc14_7.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc14_7.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc14_7.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %c.var, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc14: init %empty_tuple.type = call %bound_method.loc14(%c.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc11_5.1: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc11_5.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc11_5.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc11_5.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc11_5.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc11_5.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc11_5.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11_5.1: <bound method> = bound_method %b.var, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11_5.1: init %empty_tuple.type = call %bound_method.loc11_5.1(%b.var)
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc11_5.2: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc11_5.3: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc11_5.4: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc11_5.3 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc11_5.4: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc11_5.5: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc11_5.4 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11_5.2: <bound method> = bound_method %b.var, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11_5.2: init %empty_tuple.type = call %bound_method.loc11_5.2(%b.var)
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %Use.ref.loc19: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc19: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc19: %Destructible = ref_tag %a.ref.loc19
// CHECK:STDOUT:   %Use.call.loc19: init %empty_tuple.type = call %Use.ref.loc19(%a.ref.loc19)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc7: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.162
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%Destructible, (constants.%custom_witness.923) [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%Destructible, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %a.var, %Destroy.WithSelf.SelfDestruct.specific_fn.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7: init %empty_tuple.type = call %bound_method.loc7(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc14_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.SubobjectDestroy.loc14_7.2(%self.param: ref %Destructible) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_7.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
