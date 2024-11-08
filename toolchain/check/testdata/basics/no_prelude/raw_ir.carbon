// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import --phase=check --dump-raw-sem-ir %s
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/raw_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/raw_ir.carbon

fn Foo[T:! type](n: T) -> (T, ()) {
  return (n, ());
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        raw_ir.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<invalid>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, parent_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst+24}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope<invalid>, index: comp_time_bind0}
// CHECK:STDOUT:     entity_name1:    {name: name2, parent_scope: name_scope<invalid>, index: comp_time_bind<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, return_slot_pattern: inst+19, return_slot: inst+23, body: [block15]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst+24, bindings: block11}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: block13}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     typeTypeType:    {kind: copy, type: typeTypeType}
// CHECK:STDOUT:     typeError:       {kind: copy, type: typeError}
// CHECK:STDOUT:     'type(instNamespaceType)': {kind: copy, type: type(instNamespaceType)}
// CHECK:STDOUT:     'type(inst+28)':   {kind: none, type: type(inst+11)}
// CHECK:STDOUT:     'type(inst+11)':   {kind: none, type: type(inst+11)}
// CHECK:STDOUT:     'type(symbolicConstant0)': {kind: copy, type: type(symbolicConstant0)}
// CHECK:STDOUT:     'type(symbolicConstant2)': {kind: pointer, type: type(symbolicConstant6)}
// CHECK:STDOUT:     'type(symbolicConstant6)': {kind: copy, type: type(symbolicConstant6)}
// CHECK:STDOUT:     'type(symbolicConstant3)': {kind: copy, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'type(symbolicConstant5)': {kind: pointer, type: type(symbolicConstant6)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:     type_block1:
// CHECK:STDOUT:       0:               typeTypeType
// CHECK:STDOUT:       1:               type(inst+11)
// CHECK:STDOUT:     type_block2:
// CHECK:STDOUT:       0:               type(symbolicConstant0)
// CHECK:STDOUT:       1:               type(inst+11)
// CHECK:STDOUT:     type_block3:
// CHECK:STDOUT:       0:               type(symbolicConstant3)
// CHECK:STDOUT:       1:               type(inst+11)
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type(instNamespaceType)}
// CHECK:STDOUT:     'inst+1':          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst+20, type: typeTypeType}
// CHECK:STDOUT:     'inst+2':          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+3':          {kind: SymbolicBindingPattern, arg0: entity_name0, type: typeTypeType}
// CHECK:STDOUT:     'inst+4':          {kind: SymbolicBindingPattern, arg0: entity_name0, type: typeTypeType}
// CHECK:STDOUT:     'inst+5':          {kind: ValueParamPattern, arg0: inst+3, arg1: runtime_param<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+6':          {kind: NameRef, arg0: name1, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+7':          {kind: BindName, arg0: entity_name1, arg1: inst+21, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+8':          {kind: BindingPattern, arg0: entity_name1, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+9':          {kind: ValueParamPattern, arg0: inst+8, arg1: runtime_param0, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+10':         {kind: NameRef, arg0: name1, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+11':         {kind: TupleType, arg0: type_block0, type: typeTypeType}
// CHECK:STDOUT:     'inst+12':         {kind: TupleLiteral, arg0: empty, type: type(inst+11)}
// CHECK:STDOUT:     'inst+13':         {kind: TupleType, arg0: type_block1, type: typeTypeType}
// CHECK:STDOUT:     'inst+14':         {kind: TupleLiteral, arg0: block6, type: type(inst+13)}
// CHECK:STDOUT:     'inst+15':         {kind: Converted, arg0: inst+12, arg1: inst+11, type: typeTypeType}
// CHECK:STDOUT:     'inst+16':         {kind: TupleType, arg0: type_block2, type: typeTypeType}
// CHECK:STDOUT:     'inst+17':         {kind: Converted, arg0: inst+14, arg1: inst+16, type: typeTypeType}
// CHECK:STDOUT:     'inst+18':         {kind: ReturnSlotPattern, arg0: inst+14, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+19':         {kind: OutParamPattern, arg0: inst+18, arg1: runtime_param1, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+20':         {kind: ValueParam, arg0: runtime_param<invalid>, arg1: name1, type: typeTypeType}
// CHECK:STDOUT:     'inst+21':         {kind: ValueParam, arg0: runtime_param0, arg1: name2, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+22':         {kind: OutParam, arg0: runtime_param1, arg1: nameReturnSlot, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+23':         {kind: ReturnSlot, arg0: inst+14, arg1: inst+22, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+24':         {kind: FunctionDecl, arg0: function0, arg1: block10, type: type(inst+28)}
// CHECK:STDOUT:     'inst+25':         {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+26':         {kind: SymbolicBindingPattern, arg0: entity_name0, type: typeTypeType}
// CHECK:STDOUT:     'inst+27':         {kind: TupleType, arg0: type_block3, type: typeTypeType}
// CHECK:STDOUT:     'inst+28':         {kind: FunctionType, arg0: function0, arg1: specific<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+29':         {kind: StructValue, arg0: empty, type: type(inst+28)}
// CHECK:STDOUT:     'inst+30':         {kind: PointerType, arg0: type(symbolicConstant2), type: typeTypeType}
// CHECK:STDOUT:     'inst+31':         {kind: NameRef, arg0: name2, arg1: inst+7, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+32':         {kind: TupleLiteral, arg0: empty, type: type(inst+11)}
// CHECK:STDOUT:     'inst+33':         {kind: TupleLiteral, arg0: block16, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+34':         {kind: TupleAccess, arg0: inst+23, arg1: element0, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+35':         {kind: InitializeFrom, arg0: inst+31, arg1: inst+34, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+36':         {kind: TupleAccess, arg0: inst+23, arg1: element1, type: type(inst+11)}
// CHECK:STDOUT:     'inst+37':         {kind: TupleInit, arg0: empty, arg1: inst+36, type: type(inst+11)}
// CHECK:STDOUT:     'inst+38':         {kind: TupleValue, arg0: empty, type: type(inst+11)}
// CHECK:STDOUT:     'inst+39':         {kind: Converted, arg0: inst+32, arg1: inst+37, type: type(inst+11)}
// CHECK:STDOUT:     'inst+40':         {kind: TupleInit, arg0: block17, arg1: inst+23, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+41':         {kind: Converted, arg0: inst+33, arg1: inst+40, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+42':         {kind: ReturnExpr, arg0: inst+41, arg1: inst+23}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          templateConstant(inst+0)
// CHECK:STDOUT:     'inst+1':          symbolicConstant3
// CHECK:STDOUT:     'inst+2':          symbolicConstant0
// CHECK:STDOUT:     'inst+3':          symbolicConstant4
// CHECK:STDOUT:     'inst+4':          symbolicConstant1
// CHECK:STDOUT:     'inst+5':          symbolicConstant4
// CHECK:STDOUT:     'inst+6':          symbolicConstant3
// CHECK:STDOUT:     'inst+10':         symbolicConstant3
// CHECK:STDOUT:     'inst+11':         templateConstant(inst+11)
// CHECK:STDOUT:     'inst+13':         templateConstant(inst+13)
// CHECK:STDOUT:     'inst+15':         templateConstant(inst+11)
// CHECK:STDOUT:     'inst+16':         symbolicConstant2
// CHECK:STDOUT:     'inst+17':         symbolicConstant5
// CHECK:STDOUT:     'inst+24':         templateConstant(inst+29)
// CHECK:STDOUT:     'inst+25':         symbolicConstant3
// CHECK:STDOUT:     'inst+26':         symbolicConstant4
// CHECK:STDOUT:     'inst+27':         symbolicConstant5
// CHECK:STDOUT:     'inst+28':         templateConstant(inst+28)
// CHECK:STDOUT:     'inst+29':         templateConstant(inst+29)
// CHECK:STDOUT:     'inst+30':         symbolicConstant6
// CHECK:STDOUT:     'inst+37':         templateConstant(inst+38)
// CHECK:STDOUT:     'inst+38':         templateConstant(inst+38)
// CHECK:STDOUT:     'inst+39':         templateConstant(inst+38)
// CHECK:STDOUT:   symbolic_constants:
// CHECK:STDOUT:     symbolicConstant0: {inst: inst+2, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:     symbolicConstant1: {inst: inst+4, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:     symbolicConstant2: {inst: inst+16, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:     symbolicConstant3: {inst: inst+2, generic: generic0, index: genericInstInDecl0}
// CHECK:STDOUT:     symbolicConstant4: {inst: inst+4, generic: generic0, index: genericInstInDecl1}
// CHECK:STDOUT:     symbolicConstant5: {inst: inst+16, generic: generic0, index: genericInstInDecl2}
// CHECK:STDOUT:     symbolicConstant6: {inst: inst+30, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+24
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+5
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+9
// CHECK:STDOUT:     block6:
// CHECK:STDOUT:       0:               inst+10
// CHECK:STDOUT:       1:               inst+12
// CHECK:STDOUT:     block7:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:     block8:
// CHECK:STDOUT:       0:               inst+7
// CHECK:STDOUT:     block9:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:       1:               inst+5
// CHECK:STDOUT:       2:               inst+8
// CHECK:STDOUT:       3:               inst+9
// CHECK:STDOUT:       4:               inst+18
// CHECK:STDOUT:       5:               inst+19
// CHECK:STDOUT:     block10:
// CHECK:STDOUT:       0:               inst+6
// CHECK:STDOUT:       1:               inst+10
// CHECK:STDOUT:       2:               inst+12
// CHECK:STDOUT:       3:               inst+14
// CHECK:STDOUT:       4:               inst+15
// CHECK:STDOUT:       5:               inst+17
// CHECK:STDOUT:       6:               inst+20
// CHECK:STDOUT:       7:               inst+1
// CHECK:STDOUT:       8:               inst+21
// CHECK:STDOUT:       9:               inst+7
// CHECK:STDOUT:       10:              inst+22
// CHECK:STDOUT:       11:              inst+23
// CHECK:STDOUT:     block11:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:     block12:
// CHECK:STDOUT:       0:               inst+25
// CHECK:STDOUT:       1:               inst+26
// CHECK:STDOUT:       2:               inst+27
// CHECK:STDOUT:     block13:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:     block14:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:       1:               inst+2
// CHECK:STDOUT:       2:               inst+16
// CHECK:STDOUT:     block15:
// CHECK:STDOUT:       0:               inst+31
// CHECK:STDOUT:       1:               inst+32
// CHECK:STDOUT:       2:               inst+33
// CHECK:STDOUT:       3:               inst+34
// CHECK:STDOUT:       4:               inst+35
// CHECK:STDOUT:       5:               inst+36
// CHECK:STDOUT:       6:               inst+37
// CHECK:STDOUT:       7:               inst+39
// CHECK:STDOUT:       8:               inst+40
// CHECK:STDOUT:       9:               inst+41
// CHECK:STDOUT:       10:              inst+42
// CHECK:STDOUT:     block16:
// CHECK:STDOUT:       0:               inst+31
// CHECK:STDOUT:       1:               inst+32
// CHECK:STDOUT:     block17:
// CHECK:STDOUT:       0:               inst+35
// CHECK:STDOUT:       1:               inst+39
// CHECK:STDOUT:     block18:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+24
// CHECK:STDOUT: ...
