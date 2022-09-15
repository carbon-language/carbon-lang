// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: 1
// CHECK: 2
// CHECK: 3
// CHECK: 4
// CHECK: 1
// CHECK: 2
// CHECK: 3
// CHECK: 4
// CHECK: result: 0

package ExplorerTest api;

class ListElement{
  fn Create(value: i32)->ListElement{
    return {
      .element = Optional(i32).Create(value),
      .next = Optional(ListElement*).CreateEmpty()
    };
  }

  fn set[addr me: Self*] (value: Optional(ListElement*)){
    (*me).next = value;
  }

  var element: Optional(i32);
  var next: Optional(ListElement*);
}

class List{
  fn Create() -> List{
    return {.next = Optional(ListElement*).CreateEmpty() };
  }
  fn insert[addr me: Self*] (value: i32){
    if( not (*me).next.HasValue() ){
      (*me).next = Optional(ListElement*).Create( heap.New(ListElement.Create(value)) );
      return;
    }
    var iter: Optional(ListElement*) = (*me).next;
    var node: auto = (iter.Get());
    while((*node).next.HasValue()){
      iter = (*node).next;
      node = (iter.Get());
    }
    var v : ListElement* = iter.Get();
    var x: Optional(ListElement*) = Optional(ListElement*).Create( heap.New(ListElement.Create(value) ));
    (*v).set(x);
  }

  fn print[me: Self] (){
    var iter: Optional(ListElement*) = me.next;
    while(iter.HasValue()){
      var node: auto = *(iter.Get());
      Print("{0}",node.element.Get());
      var node_ptr: auto = iter.Get();
      iter = node.next;
    }
  }

  fn clean[addr me: Self*] (){
    var head: auto = (*me).next;
    while(head.HasValue()){
      var tmp: auto = head;
      head = (*head.Get()).next;
      heap.Delete(tmp.Get());
    }
    (*me).next= Optional(ListElement*).CreateEmpty();
  }

  var next: Optional(ListElement*);
}

fn Main() -> i32{
  var list: List = List.Create();
  list.insert(1);
  list.insert(2);
  list.insert(3);
  list.insert(4);
  list.print();
  list.clean();
  list.print();
  list.insert(1);
  list.insert(2);
  list.insert(3);
  list.insert(4);
  list.print();
  return 0;
}
