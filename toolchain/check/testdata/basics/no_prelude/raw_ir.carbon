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
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, parent_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst+21}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope<invalid>, index: comp_time_bind0}
// CHECK:STDOUT:     entity_name1:    {name: name2, parent_scope: name_scope<invalid>, index: comp_time_bind<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, return_storage: inst+20, body: [block15]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst+21, bindings: block11}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: block13}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     typeTypeType:    {kind: copy, type: typeTypeType}
// CHECK:STDOUT:     typeError:       {kind: copy, type: typeError}
// CHECK:STDOUT:     'type(instNamespaceType)': {kind: copy, type: type(instNamespaceType)}
// CHECK:STDOUT:     'type(inst+25)':   {kind: none, type: type(inst+13)}
// CHECK:STDOUT:     'type(inst+13)':   {kind: none, type: type(inst+13)}
// CHECK:STDOUT:     'type(symbolicConstant0)': {kind: copy, type: type(symbolicConstant0)}
// CHECK:STDOUT:     'type(symbolicConstant2)': {kind: pointer, type: type(symbolicConstant6)}
// CHECK:STDOUT:     'type(symbolicConstant6)': {kind: copy, type: type(symbolicConstant6)}
// CHECK:STDOUT:     'type(symbolicConstant3)': {kind: copy, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'type(symbolicConstant5)': {kind: pointer, type: type(symbolicConstant6)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:     type_block1:
// CHECK:STDOUT:       0:               typeTypeType
// CHECK:STDOUT:       1:               type(inst+13)
// CHECK:STDOUT:     type_block2:
// CHECK:STDOUT:       0:               type(symbolicConstant0)
// CHECK:STDOUT:       1:               type(inst+13)
// CHECK:STDOUT:     type_block3:
// CHECK:STDOUT:       0:               type(symbolicConstant3)
// CHECK:STDOUT:       1:               type(inst+13)
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type(instNamespaceType)}
// CHECK:STDOUT:     'inst+1':          {kind: Param, arg0: runtime_param<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+2':          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+3':          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+4':          {kind: SymbolicBindingPattern, arg0: entity_name0, type: typeTypeType}
// CHECK:STDOUT:     'inst+5':          {kind: SymbolicBindingPattern, arg0: entity_name0, type: typeTypeType}
// CHECK:STDOUT:     'inst+6':          {kind: ParamPattern, arg0: inst+4, arg1: runtime_param<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+7':          {kind: NameRef, arg0: name1, arg1: inst+2, type: typeTypeType}
// CHECK:STDOUT:     'inst+8':          {kind: Param, arg0: runtime_param0, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+9':          {kind: BindName, arg0: entity_name1, arg1: inst+8, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+10':         {kind: BindingPattern, arg0: entity_name1, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+11':         {kind: ParamPattern, arg0: inst+10, arg1: runtime_param0, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+12':         {kind: NameRef, arg0: name1, arg1: inst+2, type: typeTypeType}
// CHECK:STDOUT:     'inst+13':         {kind: TupleType, arg0: type_block0, type: typeTypeType}
// CHECK:STDOUT:     'inst+14':         {kind: TupleLiteral, arg0: empty, type: type(inst+13)}
// CHECK:STDOUT:     'inst+15':         {kind: TupleType, arg0: type_block1, type: typeTypeType}
// CHECK:STDOUT:     'inst+16':         {kind: TupleLiteral, arg0: block8, type: type(inst+15)}
// CHECK:STDOUT:     'inst+17':         {kind: Converted, arg0: inst+14, arg1: inst+13, type: typeTypeType}
// CHECK:STDOUT:     'inst+18':         {kind: TupleType, arg0: type_block2, type: typeTypeType}
// CHECK:STDOUT:     'inst+19':         {kind: Converted, arg0: inst+16, arg1: inst+18, type: typeTypeType}
// CHECK:STDOUT:     'inst+20':         {kind: VarStorage, arg0: nameReturnSlot, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+21':         {kind: FunctionDecl, arg0: function0, arg1: block10, type: type(inst+25)}
// CHECK:STDOUT:     'inst+22':         {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+23':         {kind: SymbolicBindingPattern, arg0: entity_name0, type: typeTypeType}
// CHECK:STDOUT:     'inst+24':         {kind: TupleType, arg0: type_block3, type: typeTypeType}
// CHECK:STDOUT:     'inst+25':         {kind: FunctionType, arg0: function0, arg1: specific<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+26':         {kind: StructValue, arg0: empty, type: type(inst+25)}
// CHECK:STDOUT:     'inst+27':         {kind: PointerType, arg0: type(symbolicConstant2), type: typeTypeType}
// CHECK:STDOUT:     'inst+28':         {kind: NameRef, arg0: name2, arg1: inst+9, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+29':         {kind: TupleLiteral, arg0: empty, type: type(inst+13)}
// CHECK:STDOUT:     'inst+30':         {kind: TupleLiteral, arg0: block16, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+31':         {kind: TupleAccess, arg0: inst+20, arg1: element0, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+32':         {kind: InitializeFrom, arg0: inst+28, arg1: inst+31, type: type(symbolicConstant3)}
// CHECK:STDOUT:     'inst+33':         {kind: TupleAccess, arg0: inst+20, arg1: element1, type: type(inst+13)}
// CHECK:STDOUT:     'inst+34':         {kind: TupleInit, arg0: empty, arg1: inst+33, type: type(inst+13)}
// CHECK:STDOUT:     'inst+35':         {kind: TupleValue, arg0: empty, type: type(inst+13)}
// CHECK:STDOUT:     'inst+36':         {kind: Converted, arg0: inst+29, arg1: inst+34, type: type(inst+13)}
// CHECK:STDOUT:     'inst+37':         {kind: TupleInit, arg0: block17, arg1: inst+20, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+38':         {kind: Converted, arg0: inst+30, arg1: inst+37, type: type(symbolicConstant5)}
// CHECK:STDOUT:     'inst+39':         {kind: ReturnExpr, arg0: inst+38, arg1: inst+20}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          templateConstant(inst+0)
// CHECK:STDOUT:     'inst+2':          symbolicConstant3
// CHECK:STDOUT:     'inst+3':          symbolicConstant0
// CHECK:STDOUT:     'inst+4':          symbolicConstant4
// CHECK:STDOUT:     'inst+5':          symbolicConstant1
// CHECK:STDOUT:     'inst+6':          symbolicConstant4
// CHECK:STDOUT:     'inst+7':          symbolicConstant3
// CHECK:STDOUT:     'inst+12':         symbolicConstant3
// CHECK:STDOUT:     'inst+13':         templateConstant(inst+13)
// CHECK:STDOUT:     'inst+15':         templateConstant(inst+15)
// CHECK:STDOUT:     'inst+17':         templateConstant(inst+13)
// CHECK:STDOUT:     'inst+18':         symbolicConstant2
// CHECK:STDOUT:     'inst+19':         symbolicConstant5
// CHECK:STDOUT:     'inst+21':         templateConstant(inst+26)
// CHECK:STDOUT:     'inst+22':         symbolicConstant3
// CHECK:STDOUT:     'inst+23':         symbolicConstant4
// CHECK:STDOUT:     'inst+24':         symbolicConstant5
// CHECK:STDOUT:     'inst+25':         templateConstant(inst+25)
// CHECK:STDOUT:     'inst+26':         templateConstant(inst+26)
// CHECK:STDOUT:     'inst+27':         symbolicConstant6
// CHECK:STDOUT:     'inst+34':         templateConstant(inst+35)
// CHECK:STDOUT:     'inst+35':         templateConstant(inst+35)
// CHECK:STDOUT:     'inst+36':         templateConstant(inst+35)
// CHECK:STDOUT:   symbolic_constants:
// CHECK:STDOUT:     symbolicConstant0: {inst: inst+3, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:     symbolicConstant1: {inst: inst+5, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:     symbolicConstant2: {inst: inst+18, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:     symbolicConstant3: {inst: inst+3, generic: generic0, index: genericInstInDecl0}
// CHECK:STDOUT:     symbolicConstant4: {inst: inst+5, generic: generic0, index: genericInstInDecl1}
// CHECK:STDOUT:     symbolicConstant5: {inst: inst+18, generic: generic0, index: genericInstInDecl2}
// CHECK:STDOUT:     symbolicConstant6: {inst: inst+27, generic: generic<invalid>, index: genericInst<invalid>}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+21
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+6
// CHECK:STDOUT:     block6:
// CHECK:STDOUT:       0:               inst+9
// CHECK:STDOUT:     block7:
// CHECK:STDOUT:       0:               inst+11
// CHECK:STDOUT:     block8:
// CHECK:STDOUT:       0:               inst+12
// CHECK:STDOUT:       1:               inst+14
// CHECK:STDOUT:     block9:
// CHECK:STDOUT:       0:               inst+4
// CHECK:STDOUT:       1:               inst+6
// CHECK:STDOUT:       2:               inst+10
// CHECK:STDOUT:       3:               inst+11
// CHECK:STDOUT:     block10:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:       1:               inst+2
// CHECK:STDOUT:       2:               inst+7
// CHECK:STDOUT:       3:               inst+8
// CHECK:STDOUT:       4:               inst+9
// CHECK:STDOUT:       5:               inst+12
// CHECK:STDOUT:       6:               inst+14
// CHECK:STDOUT:       7:               inst+16
// CHECK:STDOUT:       8:               inst+17
// CHECK:STDOUT:       9:               inst+19
// CHECK:STDOUT:       10:              inst+20
// CHECK:STDOUT:     block11:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:     block12:
// CHECK:STDOUT:       0:               inst+22
// CHECK:STDOUT:       1:               inst+23
// CHECK:STDOUT:       2:               inst+24
// CHECK:STDOUT:     block13:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:     block14:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:       1:               inst+3
// CHECK:STDOUT:       2:               inst+18
// CHECK:STDOUT:     block15:
// CHECK:STDOUT:       0:               inst+28
// CHECK:STDOUT:       1:               inst+29
// CHECK:STDOUT:       2:               inst+30
// CHECK:STDOUT:       3:               inst+31
// CHECK:STDOUT:       4:               inst+32
// CHECK:STDOUT:       5:               inst+33
// CHECK:STDOUT:       6:               inst+34
// CHECK:STDOUT:       7:               inst+36
// CHECK:STDOUT:       8:               inst+37
// CHECK:STDOUT:       9:               inst+38
// CHECK:STDOUT:       10:              inst+39
// CHECK:STDOUT:     block16:
// CHECK:STDOUT:       0:               inst+28
// CHECK:STDOUT:       1:               inst+29
// CHECK:STDOUT:     block17:
// CHECK:STDOUT:       0:               inst+32
// CHECK:STDOUT:       1:               inst+36
// CHECK:STDOUT:     block18:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+21
// CHECK:STDOUT: ...
