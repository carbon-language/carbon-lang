// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-raw-sem-ir %s
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE

fn Foo(n: i32) -> (i32, i32, f64) {
  return (n, 2, 3.4);
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        raw_ir.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 1
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst+9}}
// CHECK:STDOUT:   bind_names:
// CHECK:STDOUT:     bindName0:       {name: name1, enclosing_scope: name_scope<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, enclosing_scope: name_scope0, param_refs: block2, return_type: type4, return_slot: inst+7, body: [block5]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:     type1:           {constant: template instIntType, value_rep: {kind: copy, type: type1}}
// CHECK:STDOUT:     type2:           {constant: template inst+3, value_rep: {kind: unknown, type: type<invalid>}}
// CHECK:STDOUT:     type3:           {constant: template instFloatType, value_rep: {kind: copy, type: type3}}
// CHECK:STDOUT:     type4:           {constant: template inst+5, value_rep: {kind: pointer, type: type5}}
// CHECK:STDOUT:     type5:           {constant: template inst+8, value_rep: {kind: copy, type: type5}}
// CHECK:STDOUT:     type6:           {constant: template instFunctionType, value_rep: {kind: copy, type: type6}}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     typeBlock0:
// CHECK:STDOUT:       0:               typeTypeType
// CHECK:STDOUT:       1:               typeTypeType
// CHECK:STDOUT:       2:               typeTypeType
// CHECK:STDOUT:     typeBlock1:
// CHECK:STDOUT:       0:               type1
// CHECK:STDOUT:       1:               type1
// CHECK:STDOUT:       2:               type3
// CHECK:STDOUT:     typeBlock2:
// CHECK:STDOUT:       0:               type1
// CHECK:STDOUT:       1:               type1
// CHECK:STDOUT:       2:               type3
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst+0:          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:     inst+1:          {kind: Param, arg0: name1, type: type1}
// CHECK:STDOUT:     inst+2:          {kind: BindName, arg0: bindName0, arg1: inst+1, type: type1}
// CHECK:STDOUT:     inst+3:          {kind: TupleType, arg0: typeBlock0, type: typeTypeType}
// CHECK:STDOUT:     inst+4:          {kind: TupleLiteral, arg0: block3, type: type2}
// CHECK:STDOUT:     inst+5:          {kind: TupleType, arg0: typeBlock1, type: typeTypeType}
// CHECK:STDOUT:     inst+6:          {kind: Converted, arg0: inst+4, arg1: inst+5, type: typeTypeType}
// CHECK:STDOUT:     inst+7:          {kind: VarStorage, arg0: nameReturnSlot, type: type4}
// CHECK:STDOUT:     inst+8:          {kind: PointerType, arg0: type4, type: typeTypeType}
// CHECK:STDOUT:     inst+9:          {kind: FunctionDecl, arg0: function0, type: type6}
// CHECK:STDOUT:     inst+10:         {kind: NameRef, arg0: name1, arg1: inst+2, type: type1}
// CHECK:STDOUT:     inst+11:         {kind: IntLiteral, arg0: int4, type: type1}
// CHECK:STDOUT:     inst+12:         {kind: IntLiteral, arg0: int4, type: type1}
// CHECK:STDOUT:     inst+13:         {kind: RealLiteral, arg0: real0, type: type3}
// CHECK:STDOUT:     inst+14:         {kind: RealLiteral, arg0: real0, type: type3}
// CHECK:STDOUT:     inst+15:         {kind: TupleLiteral, arg0: block6, type: type4}
// CHECK:STDOUT:     inst+16:         {kind: TupleAccess, arg0: inst+7, arg1: element0, type: type1}
// CHECK:STDOUT:     inst+17:         {kind: InitializeFrom, arg0: inst+10, arg1: inst+16, type: type1}
// CHECK:STDOUT:     inst+18:         {kind: TupleAccess, arg0: inst+7, arg1: element1, type: type1}
// CHECK:STDOUT:     inst+19:         {kind: InitializeFrom, arg0: inst+11, arg1: inst+18, type: type1}
// CHECK:STDOUT:     inst+20:         {kind: TupleAccess, arg0: inst+7, arg1: element2, type: type3}
// CHECK:STDOUT:     inst+21:         {kind: InitializeFrom, arg0: inst+13, arg1: inst+20, type: type3}
// CHECK:STDOUT:     inst+22:         {kind: TupleInit, arg0: block7, arg1: inst+7, type: type4}
// CHECK:STDOUT:     inst+23:         {kind: Converted, arg0: inst+15, arg1: inst+22, type: type4}
// CHECK:STDOUT:     inst+24:         {kind: ReturnExpr, arg0: inst+23}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     inst+0:          template inst+0
// CHECK:STDOUT:     inst+3:          template inst+3
// CHECK:STDOUT:     inst+5:          template inst+5
// CHECK:STDOUT:     inst+6:          template inst+5
// CHECK:STDOUT:     inst+8:          template inst+8
// CHECK:STDOUT:     inst+9:          template inst+9
// CHECK:STDOUT:     inst+11:         template inst+12
// CHECK:STDOUT:     inst+12:         template inst+12
// CHECK:STDOUT:     inst+13:         template inst+14
// CHECK:STDOUT:     inst+14:         template inst+14
// CHECK:STDOUT:     inst+19:         template inst+12
// CHECK:STDOUT:     inst+21:         template inst+14
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+9
// CHECK:STDOUT:     block2:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:     block3:
// CHECK:STDOUT:       0:               instIntType
// CHECK:STDOUT:       1:               instIntType
// CHECK:STDOUT:       2:               instFloatType
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+1
// CHECK:STDOUT:       1:               inst+2
// CHECK:STDOUT:       2:               inst+4
// CHECK:STDOUT:       3:               inst+6
// CHECK:STDOUT:       4:               inst+7
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+10
// CHECK:STDOUT:       1:               inst+11
// CHECK:STDOUT:       2:               inst+13
// CHECK:STDOUT:       3:               inst+15
// CHECK:STDOUT:       4:               inst+16
// CHECK:STDOUT:       5:               inst+17
// CHECK:STDOUT:       6:               inst+18
// CHECK:STDOUT:       7:               inst+19
// CHECK:STDOUT:       8:               inst+20
// CHECK:STDOUT:       9:               inst+21
// CHECK:STDOUT:       10:              inst+22
// CHECK:STDOUT:       11:              inst+23
// CHECK:STDOUT:       12:              inst+24
// CHECK:STDOUT:     block6:
// CHECK:STDOUT:       0:               inst+10
// CHECK:STDOUT:       1:               inst+11
// CHECK:STDOUT:       2:               inst+13
// CHECK:STDOUT:     block7:
// CHECK:STDOUT:       0:               inst+17
// CHECK:STDOUT:       1:               inst+19
// CHECK:STDOUT:       2:               inst+21
// CHECK:STDOUT:     block8:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+9
// CHECK:STDOUT: ...
