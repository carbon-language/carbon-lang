// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import --phase=check --dump-sem-ir --dump-raw-sem-ir %s
//
// Check that we can combine textual IR and raw IR dumping in one compile.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/raw_and_textual_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/raw_and_textual_ir.carbon

fn Foo(n: ()) -> ((), ()) {
  return (n, ());
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        raw_and_textual_ir.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<invalid>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, parent_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst+19}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope<invalid>, index: comp_time_bind<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, return_slot_pattern: inst+15, return_slot: inst+18, body: [block9]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     typeTypeType:    {kind: copy, type: typeTypeType}
// CHECK:STDOUT:     typeError:       {kind: copy, type: typeError}
// CHECK:STDOUT:     'type(instNamespaceType)': {kind: copy, type: type(instNamespaceType)}
// CHECK:STDOUT:     'type(inst+20)':   {kind: none, type: type(inst+1)}
// CHECK:STDOUT:     'type(inst+1)':    {kind: none, type: type(inst+1)}
// CHECK:STDOUT:     'type(inst+9)':    {kind: pointer, type: type(inst+22)}
// CHECK:STDOUT:     'type(inst+22)':   {kind: copy, type: type(inst+22)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:     type_block1:
// CHECK:STDOUT:       0:               type(inst+1)
// CHECK:STDOUT:       1:               type(inst+1)
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type(instNamespaceType)}
// CHECK:STDOUT:     'inst+1':          {kind: TupleType, arg0: type_block0, type: typeTypeType}
// CHECK:STDOUT:     'inst+2':          {kind: TupleLiteral, arg0: empty, type: type(inst+1)}
// CHECK:STDOUT:     'inst+3':          {kind: Converted, arg0: inst+2, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+4':          {kind: BindName, arg0: entity_name0, arg1: inst+16, type: type(inst+1)}
// CHECK:STDOUT:     'inst+5':          {kind: BindingPattern, arg0: entity_name0, type: type(inst+1)}
// CHECK:STDOUT:     'inst+6':          {kind: ValueParamPattern, arg0: inst+5, arg1: runtime_param0, type: type(inst+1)}
// CHECK:STDOUT:     'inst+7':          {kind: TupleLiteral, arg0: empty, type: type(inst+1)}
// CHECK:STDOUT:     'inst+8':          {kind: TupleLiteral, arg0: empty, type: type(inst+1)}
// CHECK:STDOUT:     'inst+9':          {kind: TupleType, arg0: type_block1, type: typeTypeType}
// CHECK:STDOUT:     'inst+10':         {kind: TupleLiteral, arg0: block5, type: type(inst+9)}
// CHECK:STDOUT:     'inst+11':         {kind: Converted, arg0: inst+7, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+12':         {kind: Converted, arg0: inst+8, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+13':         {kind: Converted, arg0: inst+10, arg1: inst+9, type: typeTypeType}
// CHECK:STDOUT:     'inst+14':         {kind: ReturnSlotPattern, arg0: inst+10, type: type(inst+9)}
// CHECK:STDOUT:     'inst+15':         {kind: OutParamPattern, arg0: inst+14, arg1: runtime_param1, type: type(inst+9)}
// CHECK:STDOUT:     'inst+16':         {kind: ValueParam, arg0: runtime_param0, arg1: name1, type: type(inst+1)}
// CHECK:STDOUT:     'inst+17':         {kind: OutParam, arg0: runtime_param1, arg1: nameReturnSlot, type: type(inst+9)}
// CHECK:STDOUT:     'inst+18':         {kind: ReturnSlot, arg0: inst+10, arg1: inst+17, type: type(inst+9)}
// CHECK:STDOUT:     'inst+19':         {kind: FunctionDecl, arg0: function0, arg1: block8, type: type(inst+20)}
// CHECK:STDOUT:     'inst+20':         {kind: FunctionType, arg0: function0, arg1: specific<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+21':         {kind: StructValue, arg0: empty, type: type(inst+20)}
// CHECK:STDOUT:     'inst+22':         {kind: PointerType, arg0: type(inst+9), type: typeTypeType}
// CHECK:STDOUT:     'inst+23':         {kind: NameRef, arg0: name1, arg1: inst+4, type: type(inst+1)}
// CHECK:STDOUT:     'inst+24':         {kind: TupleLiteral, arg0: empty, type: type(inst+1)}
// CHECK:STDOUT:     'inst+25':         {kind: TupleLiteral, arg0: block10, type: type(inst+9)}
// CHECK:STDOUT:     'inst+26':         {kind: TupleAccess, arg0: inst+18, arg1: element0, type: type(inst+1)}
// CHECK:STDOUT:     'inst+27':         {kind: TupleInit, arg0: block11, arg1: inst+26, type: type(inst+1)}
// CHECK:STDOUT:     'inst+28':         {kind: TupleValue, arg0: empty, type: type(inst+1)}
// CHECK:STDOUT:     'inst+29':         {kind: Converted, arg0: inst+23, arg1: inst+27, type: type(inst+1)}
// CHECK:STDOUT:     'inst+30':         {kind: TupleAccess, arg0: inst+18, arg1: element1, type: type(inst+1)}
// CHECK:STDOUT:     'inst+31':         {kind: TupleInit, arg0: empty, arg1: inst+30, type: type(inst+1)}
// CHECK:STDOUT:     'inst+32':         {kind: Converted, arg0: inst+24, arg1: inst+31, type: type(inst+1)}
// CHECK:STDOUT:     'inst+33':         {kind: TupleInit, arg0: block12, arg1: inst+18, type: type(inst+9)}
// CHECK:STDOUT:     'inst+34':         {kind: TupleValue, arg0: block13, type: type(inst+9)}
// CHECK:STDOUT:     'inst+35':         {kind: Converted, arg0: inst+25, arg1: inst+33, type: type(inst+9)}
// CHECK:STDOUT:     'inst+36':         {kind: ReturnExpr, arg0: inst+35, arg1: inst+18}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          templateConstant(inst+0)
// CHECK:STDOUT:     'inst+1':          templateConstant(inst+1)
// CHECK:STDOUT:     'inst+3':          templateConstant(inst+1)
// CHECK:STDOUT:     'inst+9':          templateConstant(inst+9)
// CHECK:STDOUT:     'inst+11':         templateConstant(inst+1)
// CHECK:STDOUT:     'inst+12':         templateConstant(inst+1)
// CHECK:STDOUT:     'inst+13':         templateConstant(inst+9)
// CHECK:STDOUT:     'inst+19':         templateConstant(inst+21)
// CHECK:STDOUT:     'inst+20':         templateConstant(inst+20)
// CHECK:STDOUT:     'inst+21':         templateConstant(inst+21)
// CHECK:STDOUT:     'inst+22':         templateConstant(inst+22)
// CHECK:STDOUT:     'inst+27':         templateConstant(inst+28)
// CHECK:STDOUT:     'inst+28':         templateConstant(inst+28)
// CHECK:STDOUT:     'inst+29':         templateConstant(inst+28)
// CHECK:STDOUT:     'inst+31':         templateConstant(inst+28)
// CHECK:STDOUT:     'inst+32':         templateConstant(inst+28)
// CHECK:STDOUT:     'inst+33':         templateConstant(inst+34)
// CHECK:STDOUT:     'inst+34':         templateConstant(inst+34)
// CHECK:STDOUT:     'inst+35':         templateConstant(inst+34)
// CHECK:STDOUT:   symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+19
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+6
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+7
// CHECK:STDOUT:       1:               inst+8
// CHECK:STDOUT:     block6:
// CHECK:STDOUT:       0:               inst+4
// CHECK:STDOUT:     block7:
// CHECK:STDOUT:       0:               inst+5
// CHECK:STDOUT:       1:               inst+6
// CHECK:STDOUT:       2:               inst+14
// CHECK:STDOUT:       3:               inst+15
// CHECK:STDOUT:     block8:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:       1:               inst+3
// CHECK:STDOUT:       2:               inst+7
// CHECK:STDOUT:       3:               inst+8
// CHECK:STDOUT:       4:               inst+10
// CHECK:STDOUT:       5:               inst+11
// CHECK:STDOUT:       6:               inst+12
// CHECK:STDOUT:       7:               inst+13
// CHECK:STDOUT:       8:               inst+16
// CHECK:STDOUT:       9:               inst+4
// CHECK:STDOUT:       10:              inst+17
// CHECK:STDOUT:       11:              inst+18
// CHECK:STDOUT:     block9:
// CHECK:STDOUT:       0:               inst+23
// CHECK:STDOUT:       1:               inst+24
// CHECK:STDOUT:       2:               inst+25
// CHECK:STDOUT:       3:               inst+26
// CHECK:STDOUT:       4:               inst+27
// CHECK:STDOUT:       5:               inst+29
// CHECK:STDOUT:       6:               inst+30
// CHECK:STDOUT:       7:               inst+31
// CHECK:STDOUT:       8:               inst+32
// CHECK:STDOUT:       9:               inst+33
// CHECK:STDOUT:       10:              inst+35
// CHECK:STDOUT:       11:              inst+36
// CHECK:STDOUT:     block10:
// CHECK:STDOUT:       0:               inst+23
// CHECK:STDOUT:       1:               inst+24
// CHECK:STDOUT:     block11:         {}
// CHECK:STDOUT:     block12:
// CHECK:STDOUT:       0:               inst+29
// CHECK:STDOUT:       1:               inst+32
// CHECK:STDOUT:     block13:
// CHECK:STDOUT:       0:               inst+28
// CHECK:STDOUT:       1:               inst+28
// CHECK:STDOUT:     block14:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+19
// CHECK:STDOUT: ...
// CHECK:STDOUT:
// CHECK:STDOUT: --- raw_and_textual_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %n.patt: %empty_tuple.type = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %empty_tuple.type = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_12.2: type = converted %.loc15_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc15_20: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_24: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_25.1: %tuple.type = tuple_literal (%.loc15_20, %.loc15_24)
// CHECK:STDOUT:     %.loc15_25.2: type = converted %.loc15_20, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc15_25.3: type = converted %.loc15_24, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc15_25.4: type = converted %.loc15_25.1, constants.%tuple.type [template = constants.%tuple.type]
// CHECK:STDOUT:     %n.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %n: %empty_tuple.type = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %tuple.type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n.param_patt: %empty_tuple.type) -> %return: %tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %empty_tuple.type = name_ref n, %n
// CHECK:STDOUT:   %.loc16_15.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc16_16.1: %tuple.type = tuple_literal (%n.ref, %.loc16_15.1)
// CHECK:STDOUT:   %.loc16_16.2: ref %empty_tuple.type = tuple_access %return, element0
// CHECK:STDOUT:   %.loc16_11: init %empty_tuple.type = tuple_init () to %.loc16_16.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_16.3: init %empty_tuple.type = converted %n.ref, %.loc16_11 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_16.4: ref %empty_tuple.type = tuple_access %return, element1
// CHECK:STDOUT:   %.loc16_15.2: init %empty_tuple.type = tuple_init () to %.loc16_16.4 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_16.5: init %empty_tuple.type = converted %.loc16_15.1, %.loc16_15.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_16.6: init %tuple.type = tuple_init (%.loc16_16.3, %.loc16_16.5) to %return [template = constants.%tuple]
// CHECK:STDOUT:   %.loc16_17: init %tuple.type = converted %.loc16_16.1, %.loc16_16.6 [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc16_17 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
