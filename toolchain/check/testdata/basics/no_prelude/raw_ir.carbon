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
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, parent_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst+16}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope<invalid>, index: comp_time_bind0}
// CHECK:STDOUT:     entity_name1:    {name: name2, parent_scope: name_scope<invalid>, index: comp_time_bind<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, return_storage: inst+15, body: [block12]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst+16, bindings: block8}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: block10}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     typeTypeType:    {kind: copy, type: typeTypeType}
// CHECK:STDOUT:     typeError:       {kind: copy, type: typeError}
// CHECK:STDOUT:     'type(instNamespaceType)': {kind: copy, type: type(instNamespaceType)}
// CHECK:STDOUT:     'type(inst+19)':   {kind: none, type: type(inst+8)}
// CHECK:STDOUT:     'type(inst+8)':    {kind: none, type: type(inst+8)}
// CHECK:STDOUT:     'type(symbolicConstant0)': {kind: copy, type: type(symbolicConstant0)}
// CHECK:STDOUT:     'type(symbolicConstant1)': {kind: pointer, type: type(symbolicConstant4)}
// CHECK:STDOUT:     'type(symbolicConstant4)': {kind: copy, type: type(symbolicConstant4)}
// CHECK:STDOUT:     'type(symbolicConstant2)': {kind: copy, type: type(symbolicConstant2)}
// CHECK:STDOUT:     'type(symbolicConstant3)': {kind: pointer, type: type(symbolicConstant4)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:     type_block1:
// CHECK:STDOUT:       0:               typeTypeType
// CHECK:STDOUT:       1:               type(inst+8)
// CHECK:STDOUT:     type_block2:
// CHECK:STDOUT:       0:               type(symbolicConstant0)
// CHECK:STDOUT:       1:               type(inst+8)
// CHECK:STDOUT:     type_block3:
// CHECK:STDOUT:       0:               type(symbolicConstant2)
// CHECK:STDOUT:       1:               type(inst+8)
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type(instNamespaceType)}
// CHECK:STDOUT:     'inst+1':          {kind: Param, arg0: name1, type: typeTypeType}
// CHECK:STDOUT:     'inst+2':          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+3':          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+4':          {kind: NameRef, arg0: name1, arg1: inst+2, type: typeTypeType}
// CHECK:STDOUT:     'inst+5':          {kind: Param, arg0: name2, type: type(symbolicConstant2)}
// CHECK:STDOUT:     'inst+6':          {kind: BindName, arg0: entity_name1, arg1: inst+5, type: type(symbolicConstant2)}
// CHECK:STDOUT:     'inst+7':          {kind: NameRef, arg0: name1, arg1: inst+2, type: typeTypeType}
// CHECK:STDOUT:     'inst+8':          {kind: TupleType, arg0: type_block0, type: typeTypeType}
// CHECK:STDOUT:     'inst+9':          {kind: TupleLiteral, arg0: empty, type: type(inst+8)}
// CHECK:STDOUT:     'inst+10':         {kind: TupleType, arg0: type_block1, type: typeTypeType}
// CHECK:STDOUT:     'inst+11':         {kind: TupleLiteral, arg0: block6, type: type(inst+10)}
// CHECK:STDOUT:     'inst+12':         {kind: Converted, arg0: inst+9, arg1: inst+8, type: typeTypeType}
// CHECK:STDOUT:     'inst+13':         {kind: TupleType, arg0: type_block2, type: typeTypeType}
// CHECK:STDOUT:     'inst+14':         {kind: Converted, arg0: inst+11, arg1: inst+13, type: typeTypeType}
// CHECK:STDOUT:     'inst+15':         {kind: VarStorage, arg0: nameReturnSlot, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+16':         {kind: FunctionDecl, arg0: function0, arg1: block7, type: type(inst+19)}
// CHECK:STDOUT:     'inst+17':         {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+18':         {kind: TupleType, arg0: type_block3, type: typeTypeType}
// CHECK:STDOUT:     'inst+19':         {kind: FunctionType, arg0: function0, arg1: specific<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+20':         {kind: StructValue, arg0: empty, type: type(inst+19)}
// CHECK:STDOUT:     'inst+21':         {kind: PointerType, arg0: type(symbolicConstant1), type: typeTypeType}
// CHECK:STDOUT:     'inst+22':         {kind: NameRef, arg0: name2, arg1: inst+6, type: type(symbolicConstant2)}
// CHECK:STDOUT:     'inst+23':         {kind: TupleLiteral, arg0: empty, type: type(inst+8)}
// CHECK:STDOUT:     'inst+24':         {kind: TupleLiteral, arg0: block13, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+25':         {kind: TupleAccess, arg0: inst+15, arg1: element0, type: type(symbolicConstant2)}
// CHECK:STDOUT:     'inst+26':         {kind: InitializeFrom, arg0: inst+22, arg1: inst+25, type: type(symbolicConstant2)}
// CHECK:STDOUT:     'inst+27':         {kind: TupleAccess, arg0: inst+15, arg1: element1, type: type(inst+8)}
// CHECK:STDOUT:     'inst+28':         {kind: TupleInit, arg0: empty, arg1: inst+27, type: type(inst+8)}
// CHECK:STDOUT:     'inst+29':         {kind: TupleValue, arg0: block15, type: type(inst+8)}
// CHECK:STDOUT:     'inst+30':         {kind: Converted, arg0: inst+23, arg1: inst+28, type: type(inst+8)}
// CHECK:STDOUT:     'inst+31':         {kind: TupleInit, arg0: block14, arg1: inst+15, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+32':         {kind: Converted, arg0: inst+24, arg1: inst+31, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+33':         {kind: ReturnExpr, arg0: inst+32, arg1: inst+15}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          templateConstant(inst+0)
// CHECK:STDOUT:     'inst+2':          symbolicConstant2
// CHECK:STDOUT:     'inst+3':          symbolicConstant0
// CHECK:STDOUT:     'inst+4':          symbolicConstant2
// CHECK:STDOUT:     'inst+7':          symbolicConstant2
// CHECK:STDOUT:     'inst+8':          templateConstant(inst+8)
// CHECK:STDOUT:     'inst+10':         templateConstant(inst+10)
// CHECK:STDOUT:     'inst+12':         templateConstant(inst+8)
// CHECK:STDOUT:     'inst+13':         symbolicConstant1
// CHECK:STDOUT:     'inst+14':         symbolicConstant3
// CHECK:STDOUT:     'inst+16':         templateConstant(inst+20)
// CHECK:STDOUT:     'inst+17':         symbolicConstant2
// CHECK:STDOUT:     'inst+18':         symbolicConstant3
// CHECK:STDOUT:     'inst+19':         templateConstant(inst+19)
// CHECK:STDOUT:     'inst+20':         templateConstant(inst+20)
// CHECK:STDOUT:     'inst+21':         symbolicConstant4
// CHECK:STDOUT:     'inst+28':         templateConstant(inst+29)
// CHECK:STDOUT:     'inst+29':         templateConstant(inst+29)
// CHECK:STDOUT:     'inst+30':         templateConstant(inst+29)
// CHECK:STDOUT:   symbolic_constants:
// CHECK:STDOUT:     symbolicConstant0: {inst: inst+3, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:     symbolicConstant1: {inst: inst+13, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:     symbolicConstant2: {inst: inst+3, generic: generic0, index: genericInstInDecl0}
// CHECK:STDOUT:     symbolicConstant3: {inst: inst+13, generic: generic0, index: genericInstInDecl1}
// CHECK:STDOUT:     symbolicConstant4: {inst: inst+21, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+16
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+6
// CHECK:STDOUT:     block6:
// CHECK:STDOUT:       0:               inst+7
// CHECK:STDOUT:       1:               inst+9
// CHECK:STDOUT:     block7:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:       1:               inst+2
// CHECK:STDOUT:       2:               inst+4
// CHECK:STDOUT:       3:               inst+5
// CHECK:STDOUT:       4:               inst+6
// CHECK:STDOUT:       5:               inst+7
// CHECK:STDOUT:       6:               inst+9
// CHECK:STDOUT:       7:               inst+11
// CHECK:STDOUT:       8:               inst+12
// CHECK:STDOUT:       9:               inst+14
// CHECK:STDOUT:       10:              inst+15
// CHECK:STDOUT:     block8:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:     block9:
// CHECK:STDOUT:       0:               inst+17
// CHECK:STDOUT:       1:               inst+18
// CHECK:STDOUT:     block10:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:     block11:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:       1:               inst+13
// CHECK:STDOUT:     block12:
// CHECK:STDOUT:       0:               inst+22
// CHECK:STDOUT:       1:               inst+23
// CHECK:STDOUT:       2:               inst+24
// CHECK:STDOUT:       3:               inst+25
// CHECK:STDOUT:       4:               inst+26
// CHECK:STDOUT:       5:               inst+27
// CHECK:STDOUT:       6:               inst+28
// CHECK:STDOUT:       7:               inst+30
// CHECK:STDOUT:       8:               inst+31
// CHECK:STDOUT:       9:               inst+32
// CHECK:STDOUT:       10:              inst+33
// CHECK:STDOUT:     block13:
// CHECK:STDOUT:       0:               inst+22
// CHECK:STDOUT:       1:               inst+23
// CHECK:STDOUT:     block14:
// CHECK:STDOUT:       0:               inst+26
// CHECK:STDOUT:       1:               inst+30
// CHECK:STDOUT:     block15:         {}
// CHECK:STDOUT:     block16:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+16
// CHECK:STDOUT: ...
