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
  impl as Core.DestroyImpl {
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc24_5.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Destructible, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
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
// CHECK:STDOUT:   %.loc24_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_28.2: init %Destructible to %b.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc24_5.1: init %Destructible = converted %.loc24_28.1, %.loc24_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %b.var, %.loc24_5.1
// CHECK:STDOUT:   %Destructible.ref.loc24: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %b: ref %Destructible = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc25: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %b.ref: ref %Destructible = name_ref b, %b
// CHECK:STDOUT:   %.loc25: %Destructible = ref_tag %b.ref
// CHECK:STDOUT:   %Use.call.loc25: init %empty_tuple.type = call %Use.ref.loc25(%b.ref)
// CHECK:STDOUT:   %facet_value.loc24_5.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc24_5.2: %facet_type = converted constants.%Destructible, %facet_value.loc24_5.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc24_5.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc24_5.3: %facet_type = converted constants.%Destructible, %facet_value.loc24_5.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%b.var)
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %Use.ref.loc27: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc27: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc27: %Destructible = ref_tag %a.ref.loc27
// CHECK:STDOUT:   %Use.call.loc27: init %empty_tuple.type = call %Use.ref.loc27(%a.ref.loc27)
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_5.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_5.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- if_else.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc17_5.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Destructible, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
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
// CHECK:STDOUT:   %.loc17_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_28.2: init %Destructible to %b.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc17_5.1: init %Destructible = converted %.loc17_28.1, %.loc17_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %b.var, %.loc17_5.1
// CHECK:STDOUT:   %Destructible.ref.loc17: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %b: ref %Destructible = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc18: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %b.ref: ref %Destructible = name_ref b, %b
// CHECK:STDOUT:   %.loc18: %Destructible = ref_tag %b.ref
// CHECK:STDOUT:   %Use.call.loc18: init %empty_tuple.type = call %Use.ref.loc18(%b.ref)
// CHECK:STDOUT:   %facet_value.loc17_5.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc17_5.2: %facet_type = converted constants.%Destructible, %facet_value.loc17_5.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc17_5.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc17_5.3: %facet_type = converted constants.%Destructible, %facet_value.loc17_5.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc17: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%b.var)
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %c.var: ref %Destructible = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc34_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc34_28.2: init %Destructible to %c.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc34_5.1: init %Destructible = converted %.loc34_28.1, %.loc34_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %c.var, %.loc34_5.1
// CHECK:STDOUT:   %Destructible.ref.loc34: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %c: ref %Destructible = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.397 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.397 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc35: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %c.ref: ref %Destructible = name_ref c, %c
// CHECK:STDOUT:   %.loc35: %Destructible = ref_tag %c.ref
// CHECK:STDOUT:   %Use.call.loc35: init %empty_tuple.type = call %Use.ref.loc35(%c.ref)
// CHECK:STDOUT:   %facet_value.loc34_5.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc34_5.2: %facet_type = converted constants.%Destructible, %facet_value.loc34_5.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc34_5.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc34_5.3: %facet_type = converted constants.%Destructible, %facet_value.loc34_5.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc34: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%c.var)
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   %Use.ref.loc37: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc37: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc37: %Destructible = ref_tag %a.ref.loc37
// CHECK:STDOUT:   %Use.call.loc37: init %empty_tuple.type = call %Use.ref.loc37(%a.ref.loc37)
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc17_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc17_5.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc17_5.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- if_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc38_5.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc38_5.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Destructible, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
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
// CHECK:STDOUT:   %.loc38_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc38_28.2: init %Destructible to %b.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc38_5.1: init %Destructible = converted %.loc38_28.1, %.loc38_28.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %b.var, %.loc38_5.1
// CHECK:STDOUT:   %Destructible.ref.loc38: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %b: ref %Destructible = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.397 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.397 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc39: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %b.ref: ref %Destructible = name_ref b, %b
// CHECK:STDOUT:   %.loc39: %Destructible = ref_tag %b.ref
// CHECK:STDOUT:   %Use.call.loc39: init %empty_tuple.type = call %Use.ref.loc39(%b.ref)
// CHECK:STDOUT:   %facet_value.loc38_5.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc38_5.2: %facet_type = converted constants.%Destructible, %facet_value.loc38_5.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc38_5.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc38_5.3: %facet_type = converted constants.%Destructible, %facet_value.loc38_5.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%b.var)
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %facet_value.loc7_3.3: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.4: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.3 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc7_3.4: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.5: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.4 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc38_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc38_5.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_5.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_5.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- while_break.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc49_7.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc49_7.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Destructible, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
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
// CHECK:STDOUT:   %.loc49_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc49_30.2: init %Destructible to %c.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc49_7.1: init %Destructible = converted %.loc49_30.1, %.loc49_30.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %c.var, %.loc49_7.1
// CHECK:STDOUT:   %Destructible.ref.loc49: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %c: ref %Destructible = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.397 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.397 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc50: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %c.ref: ref %Destructible = name_ref c, %c
// CHECK:STDOUT:   %.loc50: %Destructible = ref_tag %c.ref
// CHECK:STDOUT:   %Use.call.loc50: init %empty_tuple.type = call %Use.ref.loc50(%c.ref)
// CHECK:STDOUT:   %facet_value.loc49_7.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc49_7.2: %facet_type = converted constants.%Destructible, %facet_value.loc49_7.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc49_7.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc49_7.3: %facet_type = converted constants.%Destructible, %facet_value.loc49_7.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%c.var)
// CHECK:STDOUT:   %facet_value.loc11_5.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.2: %facet_type = converted constants.%Destructible, %facet_value.loc11_5.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc11_5.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.3: %facet_type = converted constants.%Destructible, %facet_value.loc11_5.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%b.var)
// CHECK:STDOUT:   br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %facet_value.loc11_5.3: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.4: %facet_type = converted constants.%Destructible, %facet_value.loc11_5.3 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc11_5.4: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.5: %facet_type = converted constants.%Destructible, %facet_value.loc11_5.4 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%b.var)
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %Use.ref.loc54: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc54: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc54: %Destructible = ref_tag %a.ref.loc54
// CHECK:STDOUT:   %Use.call.loc54: init %empty_tuple.type = call %Use.ref.loc54(%a.ref.loc54)
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc49_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc49_7.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc49_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc49_7.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- while_continue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc49_7.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc49_7.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Destructible, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
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
// CHECK:STDOUT:   %.loc49_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc49_30.2: init %Destructible to %c.var = class_init () [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   %.loc49_7.1: init %Destructible = converted %.loc49_30.1, %.loc49_30.2 [concrete = constants.%Destructible.val]
// CHECK:STDOUT:   assign %c.var, %.loc49_7.1
// CHECK:STDOUT:   %Destructible.ref.loc49: type = name_ref Destructible, imports.%Main.Destructible [concrete = constants.%Destructible]
// CHECK:STDOUT:   %c: ref %Destructible = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.397 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.397 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Use.ref.loc50: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %c.ref: ref %Destructible = name_ref c, %c
// CHECK:STDOUT:   %.loc50: %Destructible = ref_tag %c.ref
// CHECK:STDOUT:   %Use.call.loc50: init %empty_tuple.type = call %Use.ref.loc50(%c.ref)
// CHECK:STDOUT:   %facet_value.loc49_7.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc49_7.2: %facet_type = converted constants.%Destructible, %facet_value.loc49_7.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc49_7.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc49_7.3: %facet_type = converted constants.%Destructible, %facet_value.loc49_7.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%c.var)
// CHECK:STDOUT:   %facet_value.loc11_5.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.2: %facet_type = converted constants.%Destructible, %facet_value.loc11_5.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc11_5.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.3: %facet_type = converted constants.%Destructible, %facet_value.loc11_5.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_5.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%b.var)
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %facet_value.loc11_5.3: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.4: %facet_type = converted constants.%Destructible, %facet_value.loc11_5.3 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc11_5.4: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.5: %facet_type = converted constants.%Destructible, %facet_value.loc11_5.4 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11_5.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%b.var)
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   %Use.ref.loc54: %Use.type = name_ref Use, imports.%Main.Use [concrete = constants.%Use]
// CHECK:STDOUT:   %a.ref.loc54: ref %Destructible = name_ref a, %a
// CHECK:STDOUT:   %.loc54: %Destructible = ref_tag %a.ref.loc54
// CHECK:STDOUT:   %Use.call.loc54: init %empty_tuple.type = call %Use.ref.loc54(%a.ref.loc54)
// CHECK:STDOUT:   %facet_value.loc7_3.1: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.2: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc7_3.2: %facet_type = facet_value constants.%Destructible, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc7_3.3: %facet_type = converted constants.%Destructible, %facet_value.loc7_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc49_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc49_7.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc49_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc49_7.2(%self.param: ref %Destructible) = "no_op";
// CHECK:STDOUT:
