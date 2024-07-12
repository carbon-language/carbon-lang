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
// CHECK:STDOUT:   bind_names:
// CHECK:STDOUT:     bind_name0:      {name: name1, parent_scope: name_scope<invalid>, index: comp_time_bind0}
// CHECK:STDOUT:     bind_name1:      {name: name2, parent_scope: name_scope<invalid>, index: comp_time_bind<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, param_refs: block5, return_storage: inst+15, return_slot: present, body: [block11]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst+16, bindings: block8}
// CHECK:STDOUT:   generic_instances:
// CHECK:STDOUT:     genericInstance0: {generic: generic0, args: block10}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:     type1:           {constant: symbolic 0, value_rep: {kind: copy, type: type1}}
// CHECK:STDOUT:     type2:           {constant: template inst+8, value_rep: {kind: none, type: type2}}
// CHECK:STDOUT:     type3:           {constant: template inst+10, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:     type4:           {constant: symbolic 1, value_rep: {kind: pointer, type: type8}}
// CHECK:STDOUT:     type5:           {constant: symbolic 2, value_rep: {kind: copy, type: type5}}
// CHECK:STDOUT:     type6:           {constant: symbolic 3, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:     type7:           {constant: template inst+17, value_rep: {kind: none, type: type2}}
// CHECK:STDOUT:     type8:           {constant: symbolic 4, value_rep: {kind: copy, type: type8}}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:     type_block1:
// CHECK:STDOUT:       0:               typeTypeType
// CHECK:STDOUT:       1:               type2
// CHECK:STDOUT:     type_block2:
// CHECK:STDOUT:       0:               type1
// CHECK:STDOUT:       1:               type2
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:     'inst+1':          {kind: Param, arg0: name1, type: typeTypeType}
// CHECK:STDOUT:     'inst+2':          {kind: BindSymbolicName, arg0: bind_name0, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+3':          {kind: BindSymbolicName, arg0: bind_name0, arg1: inst<invalid>, type: typeTypeType}
// CHECK:STDOUT:     'inst+4':          {kind: NameRef, arg0: name1, arg1: inst+2, type: typeTypeType}
// CHECK:STDOUT:     'inst+5':          {kind: Param, arg0: name2, type: type5}
// CHECK:STDOUT:     'inst+6':          {kind: BindName, arg0: bind_name1, arg1: inst+5, type: type5}
// CHECK:STDOUT:     'inst+7':          {kind: NameRef, arg0: name1, arg1: inst+2, type: typeTypeType}
// CHECK:STDOUT:     'inst+8':          {kind: TupleType, arg0: type_block0, type: typeTypeType}
// CHECK:STDOUT:     'inst+9':          {kind: TupleLiteral, arg0: empty, type: type2}
// CHECK:STDOUT:     'inst+10':         {kind: TupleType, arg0: type_block1, type: typeTypeType}
// CHECK:STDOUT:     'inst+11':         {kind: TupleLiteral, arg0: block6, type: type3}
// CHECK:STDOUT:     'inst+12':         {kind: Converted, arg0: inst+9, arg1: inst+8, type: typeTypeType}
// CHECK:STDOUT:     'inst+13':         {kind: TupleType, arg0: type_block2, type: typeTypeType}
// CHECK:STDOUT:     'inst+14':         {kind: Converted, arg0: inst+11, arg1: inst+13, type: typeTypeType}
// CHECK:STDOUT:     'inst+15':         {kind: VarStorage, arg0: nameReturnSlot, type: type4}
// CHECK:STDOUT:     'inst+16':         {kind: FunctionDecl, arg0: function0, arg1: block7, type: type7}
// CHECK:STDOUT:     'inst+17':         {kind: FunctionType, arg0: function0, type: typeTypeType}
// CHECK:STDOUT:     'inst+18':         {kind: StructValue, arg0: empty, type: type7}
// CHECK:STDOUT:     'inst+19':         {kind: PointerType, arg0: type4, type: typeTypeType}
// CHECK:STDOUT:     'inst+20':         {kind: NameRef, arg0: name2, arg1: inst+6, type: type1}
// CHECK:STDOUT:     'inst+21':         {kind: TupleLiteral, arg0: empty, type: type2}
// CHECK:STDOUT:     'inst+22':         {kind: TupleLiteral, arg0: block12, type: type4}
// CHECK:STDOUT:     'inst+23':         {kind: TupleAccess, arg0: inst+15, arg1: element0, type: type1}
// CHECK:STDOUT:     'inst+24':         {kind: InitializeFrom, arg0: inst+20, arg1: inst+23, type: type1}
// CHECK:STDOUT:     'inst+25':         {kind: TupleAccess, arg0: inst+15, arg1: element1, type: type2}
// CHECK:STDOUT:     'inst+26':         {kind: TupleInit, arg0: empty, arg1: inst+25, type: type2}
// CHECK:STDOUT:     'inst+27':         {kind: TupleValue, arg0: block14, type: type2}
// CHECK:STDOUT:     'inst+28':         {kind: Converted, arg0: inst+21, arg1: inst+26, type: type2}
// CHECK:STDOUT:     'inst+29':         {kind: TupleInit, arg0: block13, arg1: inst+15, type: type4}
// CHECK:STDOUT:     'inst+30':         {kind: Converted, arg0: inst+22, arg1: inst+29, type: type4}
// CHECK:STDOUT:     'inst+31':         {kind: ReturnExpr, arg0: inst+30, arg1: inst+15}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          template inst+0
// CHECK:STDOUT:     'inst+2':          symbolic 2
// CHECK:STDOUT:     'inst+3':          symbolic 0
// CHECK:STDOUT:     'inst+4':          symbolic 2
// CHECK:STDOUT:     'inst+7':          symbolic 2
// CHECK:STDOUT:     'inst+8':          template inst+8
// CHECK:STDOUT:     'inst+10':         template inst+10
// CHECK:STDOUT:     'inst+12':         template inst+8
// CHECK:STDOUT:     'inst+13':         symbolic 1
// CHECK:STDOUT:     'inst+14':         symbolic 3
// CHECK:STDOUT:     'inst+16':         template inst+18
// CHECK:STDOUT:     'inst+17':         template inst+17
// CHECK:STDOUT:     'inst+18':         template inst+18
// CHECK:STDOUT:     'inst+19':         symbolic 4
// CHECK:STDOUT:     'inst+26':         template inst+27
// CHECK:STDOUT:     'inst+27':         template inst+27
// CHECK:STDOUT:     'inst+28':         template inst+27
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
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:       1:               inst+14
// CHECK:STDOUT:     block10:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:     block11:
// CHECK:STDOUT:       0:               inst+20
// CHECK:STDOUT:       1:               inst+21
// CHECK:STDOUT:       2:               inst+22
// CHECK:STDOUT:       3:               inst+23
// CHECK:STDOUT:       4:               inst+24
// CHECK:STDOUT:       5:               inst+25
// CHECK:STDOUT:       6:               inst+26
// CHECK:STDOUT:       7:               inst+28
// CHECK:STDOUT:       8:               inst+29
// CHECK:STDOUT:       9:               inst+30
// CHECK:STDOUT:       10:              inst+31
// CHECK:STDOUT:     block12:
// CHECK:STDOUT:       0:               inst+20
// CHECK:STDOUT:       1:               inst+21
// CHECK:STDOUT:     block13:
// CHECK:STDOUT:       0:               inst+24
// CHECK:STDOUT:       1:               inst+28
// CHECK:STDOUT:     block14:         {}
// CHECK:STDOUT:     block15:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+16
// CHECK:STDOUT: ...
