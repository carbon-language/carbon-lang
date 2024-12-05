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
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, return_slot_pattern: inst+19, body: [inst_block14]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst+24, bindings: inst_block10}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: inst_block12}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':  {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':     {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))': {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst+28)':   {kind: none, type: type(inst+11)}
// CHECK:STDOUT:     'type(inst+11)':   {kind: none, type: type(inst+11)}
// CHECK:STDOUT:     'type(symbolic_constant0)': {kind: copy, type: type(symbolic_constant0)}
// CHECK:STDOUT:     'type(symbolic_constant2)': {kind: pointer, type: type(symbolic_constant6)}
// CHECK:STDOUT:     'type(symbolic_constant6)': {kind: copy, type: type(symbolic_constant6)}
// CHECK:STDOUT:     'type(symbolic_constant3)': {kind: copy, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'type(symbolic_constant5)': {kind: pointer, type: type(symbolic_constant6)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:     type_block1:
// CHECK:STDOUT:       0:               type(TypeType)
// CHECK:STDOUT:       1:               type(inst+11)
// CHECK:STDOUT:     type_block2:
// CHECK:STDOUT:       0:               type(symbolic_constant0)
// CHECK:STDOUT:       1:               type(inst+11)
// CHECK:STDOUT:     type_block3:
// CHECK:STDOUT:       0:               type(symbolic_constant3)
// CHECK:STDOUT:       1:               type(inst+11)
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'inst+1':          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst+20, type: type(TypeType)}
// CHECK:STDOUT:     'inst+2':          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: type(TypeType)}
// CHECK:STDOUT:     'inst+3':          {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(TypeType)}
// CHECK:STDOUT:     'inst+4':          {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(TypeType)}
// CHECK:STDOUT:     'inst+5':          {kind: ValueParamPattern, arg0: inst+3, arg1: runtime_param<invalid>, type: type(TypeType)}
// CHECK:STDOUT:     'inst+6':          {kind: NameRef, arg0: name1, arg1: inst+1, type: type(TypeType)}
// CHECK:STDOUT:     'inst+7':          {kind: BindName, arg0: entity_name1, arg1: inst+21, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'inst+8':          {kind: BindingPattern, arg0: entity_name1, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'inst+9':          {kind: ValueParamPattern, arg0: inst+8, arg1: runtime_param0, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'inst+10':         {kind: NameRef, arg0: name1, arg1: inst+1, type: type(TypeType)}
// CHECK:STDOUT:     'inst+11':         {kind: TupleType, arg0: type_block0, type: type(TypeType)}
// CHECK:STDOUT:     'inst+12':         {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst+11)}
// CHECK:STDOUT:     'inst+13':         {kind: TupleType, arg0: type_block1, type: type(TypeType)}
// CHECK:STDOUT:     'inst+14':         {kind: TupleLiteral, arg0: inst_block6, type: type(inst+13)}
// CHECK:STDOUT:     'inst+15':         {kind: Converted, arg0: inst+12, arg1: inst+11, type: type(TypeType)}
// CHECK:STDOUT:     'inst+16':         {kind: TupleType, arg0: type_block2, type: type(TypeType)}
// CHECK:STDOUT:     'inst+17':         {kind: Converted, arg0: inst+14, arg1: inst+16, type: type(TypeType)}
// CHECK:STDOUT:     'inst+18':         {kind: ReturnSlotPattern, arg0: inst+14, type: type(symbolic_constant5)}
// CHECK:STDOUT:     'inst+19':         {kind: OutParamPattern, arg0: inst+18, arg1: runtime_param1, type: type(symbolic_constant5)}
// CHECK:STDOUT:     'inst+20':         {kind: ValueParam, arg0: runtime_param<invalid>, arg1: name1, type: type(TypeType)}
// CHECK:STDOUT:     'inst+21':         {kind: ValueParam, arg0: runtime_param0, arg1: name2, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'inst+22':         {kind: OutParam, arg0: runtime_param1, arg1: name(ReturnSlot), type: type(symbolic_constant5)}
// CHECK:STDOUT:     'inst+23':         {kind: ReturnSlot, arg0: inst+14, arg1: inst+22, type: type(symbolic_constant5)}
// CHECK:STDOUT:     'inst+24':         {kind: FunctionDecl, arg0: function0, arg1: inst_block9, type: type(inst+28)}
// CHECK:STDOUT:     'inst+25':         {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: type(TypeType)}
// CHECK:STDOUT:     'inst+26':         {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(TypeType)}
// CHECK:STDOUT:     'inst+27':         {kind: TupleType, arg0: type_block3, type: type(TypeType)}
// CHECK:STDOUT:     'inst+28':         {kind: FunctionType, arg0: function0, arg1: specific<invalid>, type: type(TypeType)}
// CHECK:STDOUT:     'inst+29':         {kind: StructValue, arg0: inst_block_empty, type: type(inst+28)}
// CHECK:STDOUT:     'inst+30':         {kind: PointerType, arg0: type(symbolic_constant2), type: type(TypeType)}
// CHECK:STDOUT:     'inst+31':         {kind: NameRef, arg0: name2, arg1: inst+7, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'inst+32':         {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst+11)}
// CHECK:STDOUT:     'inst+33':         {kind: TupleLiteral, arg0: inst_block15, type: type(symbolic_constant5)}
// CHECK:STDOUT:     'inst+34':         {kind: TupleAccess, arg0: inst+23, arg1: element0, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'inst+35':         {kind: InitializeFrom, arg0: inst+31, arg1: inst+34, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'inst+36':         {kind: TupleAccess, arg0: inst+23, arg1: element1, type: type(inst+11)}
// CHECK:STDOUT:     'inst+37':         {kind: TupleInit, arg0: inst_block_empty, arg1: inst+36, type: type(inst+11)}
// CHECK:STDOUT:     'inst+38':         {kind: TupleValue, arg0: inst_block_empty, type: type(inst+11)}
// CHECK:STDOUT:     'inst+39':         {kind: Converted, arg0: inst+32, arg1: inst+37, type: type(inst+11)}
// CHECK:STDOUT:     'inst+40':         {kind: TupleInit, arg0: inst_block16, arg1: inst+23, type: type(symbolic_constant5)}
// CHECK:STDOUT:     'inst+41':         {kind: Converted, arg0: inst+33, arg1: inst+40, type: type(symbolic_constant5)}
// CHECK:STDOUT:     'inst+42':         {kind: ReturnExpr, arg0: inst+41, arg1: inst+23}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          template_constant(inst+0)
// CHECK:STDOUT:     'inst+1':          symbolic_constant3
// CHECK:STDOUT:     'inst+2':          symbolic_constant0
// CHECK:STDOUT:     'inst+3':          symbolic_constant4
// CHECK:STDOUT:     'inst+4':          symbolic_constant1
// CHECK:STDOUT:     'inst+5':          symbolic_constant4
// CHECK:STDOUT:     'inst+6':          symbolic_constant3
// CHECK:STDOUT:     'inst+10':         symbolic_constant3
// CHECK:STDOUT:     'inst+11':         template_constant(inst+11)
// CHECK:STDOUT:     'inst+13':         template_constant(inst+13)
// CHECK:STDOUT:     'inst+15':         template_constant(inst+11)
// CHECK:STDOUT:     'inst+16':         symbolic_constant2
// CHECK:STDOUT:     'inst+17':         symbolic_constant5
// CHECK:STDOUT:     'inst+24':         template_constant(inst+29)
// CHECK:STDOUT:     'inst+25':         symbolic_constant3
// CHECK:STDOUT:     'inst+26':         symbolic_constant4
// CHECK:STDOUT:     'inst+27':         symbolic_constant5
// CHECK:STDOUT:     'inst+28':         template_constant(inst+28)
// CHECK:STDOUT:     'inst+29':         template_constant(inst+29)
// CHECK:STDOUT:     'inst+30':         symbolic_constant6
// CHECK:STDOUT:     'inst+37':         template_constant(inst+38)
// CHECK:STDOUT:     'inst+38':         template_constant(inst+38)
// CHECK:STDOUT:     'inst+39':         template_constant(inst+38)
// CHECK:STDOUT:   symbolic_constants:
// CHECK:STDOUT:     symbolic_constant0: {inst: inst+2, generic: generic<invalid>, index: generic_inst<invalid>, .Self: false}
// CHECK:STDOUT:     symbolic_constant1: {inst: inst+4, generic: generic<invalid>, index: generic_inst<invalid>, .Self: false}
// CHECK:STDOUT:     symbolic_constant2: {inst: inst+16, generic: generic<invalid>, index: generic_inst<invalid>, .Self: false}
// CHECK:STDOUT:     symbolic_constant3: {inst: inst+2, generic: generic0, index: generic_inst_in_decl0, .Self: false}
// CHECK:STDOUT:     symbolic_constant4: {inst: inst+4, generic: generic0, index: generic_inst_in_decl1, .Self: false}
// CHECK:STDOUT:     symbolic_constant5: {inst: inst+16, generic: generic0, index: generic_inst_in_decl2, .Self: false}
// CHECK:STDOUT:     symbolic_constant6: {inst: inst+30, generic: generic<invalid>, index: generic_inst<invalid>, .Self: false}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+24
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst+5
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst+9
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst+10
// CHECK:STDOUT:       1:               inst+12
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst+21
// CHECK:STDOUT:       1:               inst+22
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst+3
// CHECK:STDOUT:       1:               inst+5
// CHECK:STDOUT:       2:               inst+8
// CHECK:STDOUT:       3:               inst+9
// CHECK:STDOUT:       4:               inst+18
// CHECK:STDOUT:       5:               inst+19
// CHECK:STDOUT:     inst_block9:
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
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst+25
// CHECK:STDOUT:       1:               inst+26
// CHECK:STDOUT:       2:               inst+27
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:       1:               inst+2
// CHECK:STDOUT:       2:               inst+16
// CHECK:STDOUT:     inst_block14:
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
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst+31
// CHECK:STDOUT:       1:               inst+32
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst+35
// CHECK:STDOUT:       1:               inst+39
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+24
// CHECK:STDOUT: ...
