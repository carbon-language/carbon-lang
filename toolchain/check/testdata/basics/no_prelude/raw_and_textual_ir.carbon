// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import --phase=check --dump-sem-ir --dump-raw-sem-ir %s
//
// Check that we can combine textual IR and raw IR dumping in one compile.
//
// AUTOUPDATE

fn Foo(n: ()) -> ((), ()) {
  return (n, ());
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        raw_and_textual_ir.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs_size: 1
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst+0, enclosing_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst+14}}
// CHECK:STDOUT:   bind_names:
// CHECK:STDOUT:     bindName0:       {name: name1, enclosing_scope: name_scope<invalid>, index: compTimeBind<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, enclosing_scope: name_scope0, param_refs: block3, return_type: type2, return_storage: inst+13, return_slot: present, body: [block6]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     type0:           {constant: template instNamespaceType, value_rep: {kind: copy, type: type0}}
// CHECK:STDOUT:     type1:           {constant: template inst+1, value_rep: {kind: none, type: type1}}
// CHECK:STDOUT:     type2:           {constant: template inst+8, value_rep: {kind: pointer, type: type4}}
// CHECK:STDOUT:     type3:           {constant: template inst+15, value_rep: {kind: none, type: type1}}
// CHECK:STDOUT:     type4:           {constant: template inst+17, value_rep: {kind: copy, type: type4}}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     typeBlock0:      {}
// CHECK:STDOUT:     typeBlock1:      {}
// CHECK:STDOUT:     typeBlock2:      {}
// CHECK:STDOUT:     typeBlock3:      {}
// CHECK:STDOUT:     typeBlock4:
// CHECK:STDOUT:       0:               type1
// CHECK:STDOUT:       1:               type1
// CHECK:STDOUT:     typeBlock5:      {}
// CHECK:STDOUT:     typeBlock6:      {}
// CHECK:STDOUT:     typeBlock7:
// CHECK:STDOUT:       0:               type1
// CHECK:STDOUT:       1:               type1
// CHECK:STDOUT:     typeBlock8:      {}
// CHECK:STDOUT:     typeBlock9:      {}
// CHECK:STDOUT:     typeBlock10:     {}
// CHECK:STDOUT:     typeBlock11:
// CHECK:STDOUT:       0:               type1
// CHECK:STDOUT:       1:               type1
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst+0':          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type0}
// CHECK:STDOUT:     'inst+1':          {kind: TupleType, arg0: typeBlock0, type: typeTypeType}
// CHECK:STDOUT:     'inst+2':          {kind: TupleLiteral, arg0: empty, type: type1}
// CHECK:STDOUT:     'inst+3':          {kind: Converted, arg0: inst+2, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+4':          {kind: Param, arg0: name1, type: type1}
// CHECK:STDOUT:     'inst+5':          {kind: BindName, arg0: bindName0, arg1: inst+4, type: type1}
// CHECK:STDOUT:     'inst+6':          {kind: TupleLiteral, arg0: empty, type: type1}
// CHECK:STDOUT:     'inst+7':          {kind: TupleLiteral, arg0: empty, type: type1}
// CHECK:STDOUT:     'inst+8':          {kind: TupleType, arg0: typeBlock4, type: typeTypeType}
// CHECK:STDOUT:     'inst+9':          {kind: TupleLiteral, arg0: block4, type: type2}
// CHECK:STDOUT:     'inst+10':         {kind: Converted, arg0: inst+6, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+11':         {kind: Converted, arg0: inst+7, arg1: inst+1, type: typeTypeType}
// CHECK:STDOUT:     'inst+12':         {kind: Converted, arg0: inst+9, arg1: inst+8, type: typeTypeType}
// CHECK:STDOUT:     'inst+13':         {kind: VarStorage, arg0: nameReturnSlot, type: type2}
// CHECK:STDOUT:     'inst+14':         {kind: FunctionDecl, arg0: function0, arg1: block5, type: type3}
// CHECK:STDOUT:     'inst+15':         {kind: FunctionType, arg0: function0, type: typeTypeType}
// CHECK:STDOUT:     'inst+16':         {kind: StructValue, arg0: empty, type: type3}
// CHECK:STDOUT:     'inst+17':         {kind: PointerType, arg0: type2, type: typeTypeType}
// CHECK:STDOUT:     'inst+18':         {kind: NameRef, arg0: name1, arg1: inst+5, type: type1}
// CHECK:STDOUT:     'inst+19':         {kind: TupleLiteral, arg0: empty, type: type1}
// CHECK:STDOUT:     'inst+20':         {kind: TupleLiteral, arg0: block7, type: type2}
// CHECK:STDOUT:     'inst+21':         {kind: TupleAccess, arg0: inst+13, arg1: element0, type: type1}
// CHECK:STDOUT:     'inst+22':         {kind: TupleInit, arg0: block8, arg1: inst+21, type: type1}
// CHECK:STDOUT:     'inst+23':         {kind: TupleValue, arg0: block9, type: type1}
// CHECK:STDOUT:     'inst+24':         {kind: Converted, arg0: inst+18, arg1: inst+22, type: type1}
// CHECK:STDOUT:     'inst+25':         {kind: TupleAccess, arg0: inst+13, arg1: element1, type: type1}
// CHECK:STDOUT:     'inst+26':         {kind: TupleInit, arg0: empty, arg1: inst+25, type: type1}
// CHECK:STDOUT:     'inst+27':         {kind: Converted, arg0: inst+19, arg1: inst+26, type: type1}
// CHECK:STDOUT:     'inst+28':         {kind: TupleInit, arg0: block10, arg1: inst+13, type: type2}
// CHECK:STDOUT:     'inst+29':         {kind: TupleValue, arg0: block12, type: type2}
// CHECK:STDOUT:     'inst+30':         {kind: Converted, arg0: inst+20, arg1: inst+28, type: type2}
// CHECK:STDOUT:     'inst+31':         {kind: ReturnExpr, arg0: inst+30, arg1: inst+13}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     'inst+0':          template inst+0
// CHECK:STDOUT:     'inst+1':          template inst+1
// CHECK:STDOUT:     'inst+3':          template inst+1
// CHECK:STDOUT:     'inst+8':          template inst+8
// CHECK:STDOUT:     'inst+10':         template inst+1
// CHECK:STDOUT:     'inst+11':         template inst+1
// CHECK:STDOUT:     'inst+12':         template inst+8
// CHECK:STDOUT:     'inst+14':         template inst+16
// CHECK:STDOUT:     'inst+15':         template inst+15
// CHECK:STDOUT:     'inst+16':         template inst+16
// CHECK:STDOUT:     'inst+17':         template inst+17
// CHECK:STDOUT:     'inst+22':         template inst+23
// CHECK:STDOUT:     'inst+23':         template inst+23
// CHECK:STDOUT:     'inst+24':         template inst+23
// CHECK:STDOUT:     'inst+26':         template inst+23
// CHECK:STDOUT:     'inst+27':         template inst+23
// CHECK:STDOUT:     'inst+28':         template inst+29
// CHECK:STDOUT:     'inst+29':         template inst+29
// CHECK:STDOUT:     'inst+30':         template inst+29
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     empty:           {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst+14
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     block3:
// CHECK:STDOUT:       0:               inst+5
// CHECK:STDOUT:     block4:
// CHECK:STDOUT:       0:               inst+6
// CHECK:STDOUT:       1:               inst+7
// CHECK:STDOUT:     block5:
// CHECK:STDOUT:       0:               inst+2
// CHECK:STDOUT:       1:               inst+3
// CHECK:STDOUT:       2:               inst+4
// CHECK:STDOUT:       3:               inst+5
// CHECK:STDOUT:       4:               inst+6
// CHECK:STDOUT:       5:               inst+7
// CHECK:STDOUT:       6:               inst+9
// CHECK:STDOUT:       7:               inst+10
// CHECK:STDOUT:       8:               inst+11
// CHECK:STDOUT:       9:               inst+12
// CHECK:STDOUT:       10:              inst+13
// CHECK:STDOUT:     block6:
// CHECK:STDOUT:       0:               inst+18
// CHECK:STDOUT:       1:               inst+19
// CHECK:STDOUT:       2:               inst+20
// CHECK:STDOUT:       3:               inst+21
// CHECK:STDOUT:       4:               inst+22
// CHECK:STDOUT:       5:               inst+24
// CHECK:STDOUT:       6:               inst+25
// CHECK:STDOUT:       7:               inst+26
// CHECK:STDOUT:       8:               inst+27
// CHECK:STDOUT:       9:               inst+28
// CHECK:STDOUT:       10:              inst+30
// CHECK:STDOUT:       11:              inst+31
// CHECK:STDOUT:     block7:
// CHECK:STDOUT:       0:               inst+18
// CHECK:STDOUT:       1:               inst+19
// CHECK:STDOUT:     block8:          {}
// CHECK:STDOUT:     block9:          {}
// CHECK:STDOUT:     block10:
// CHECK:STDOUT:       0:               inst+24
// CHECK:STDOUT:       1:               inst+27
// CHECK:STDOUT:     block11:         {}
// CHECK:STDOUT:     block12:
// CHECK:STDOUT:       0:               inst+23
// CHECK:STDOUT:       1:               inst+23
// CHECK:STDOUT:     block13:
// CHECK:STDOUT:       0:               inst+0
// CHECK:STDOUT:       1:               inst+14
// CHECK:STDOUT: ...
// CHECK:STDOUT:
// CHECK:STDOUT: --- raw_and_textual_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = tuple_type ((), ()) [template]
// CHECK:STDOUT:   %Foo: type = fn_type @Foo [template]
// CHECK:STDOUT:   %struct: Foo = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type ((), ()) [template]
// CHECK:STDOUT:   %tuple.1: () = tuple_value () [template]
// CHECK:STDOUT:   %tuple.2: ((), ()) = tuple_value (%tuple.1, %tuple.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: Foo = fn_decl @Foo [template = constants.%struct] {
// CHECK:STDOUT:     %.loc11_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_12.2: type = converted %.loc11_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %n.loc11_8.1: () = param n
// CHECK:STDOUT:     @Foo.%n: () = bind_name n, %n.loc11_8.1
// CHECK:STDOUT:     %.loc11_20: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_24: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_25.1: ((), ()) = tuple_literal (%.loc11_20, %.loc11_24)
// CHECK:STDOUT:     %.loc11_25.2: type = converted %.loc11_20, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_25.3: type = converted %.loc11_24, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_25.4: type = converted %.loc11_25.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     @Foo.%return: ref ((), ()) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n: ()) -> %return: ((), ()) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: () = name_ref n, %n
// CHECK:STDOUT:   %.loc12_15.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_16.1: ((), ()) = tuple_literal (%n.ref, %.loc12_15.1)
// CHECK:STDOUT:   %.loc12_16.2: ref () = tuple_access %return, element0
// CHECK:STDOUT:   %.loc12_11: init () = tuple_init () to %.loc12_16.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc12_16.3: init () = converted %n.ref, %.loc12_11 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc12_16.4: ref () = tuple_access %return, element1
// CHECK:STDOUT:   %.loc12_15.2: init () = tuple_init () to %.loc12_16.4 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc12_16.5: init () = converted %.loc12_15.1, %.loc12_15.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc12_16.6: init ((), ()) = tuple_init (%.loc12_16.3, %.loc12_16.5) to %return [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc12_17: init ((), ()) = converted %.loc12_16.1, %.loc12_16.6 [template = constants.%tuple.2]
// CHECK:STDOUT:   return %.loc12_17 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
