// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: 1
// CHECK:STDOUT: 2
// CHECK:STDOUT: 3
// CHECK:STDOUT: 4
// CHECK:STDOUT: 1
// CHECK:STDOUT: 2
// CHECK:STDOUT: 3
// CHECK:STDOUT: 4
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class ListElement{
  fn Create(value: i32)->ListElement{
    return {
      .element = Optional(i32).Create(value),
      .next = Optional(ListElement*).CreateEmpty()
    };
  }

  fn set[addr self: Self*] (value: Optional(ListElement*)){
    (*self).next = value;
  }

  var element: Optional(i32);
  var next: Optional(ListElement*);
}

class List{
  fn Create() -> List{
    return {.next = Optional(ListElement*).CreateEmpty() };
  }
  fn insert[addr self: Self*] (value: i32){
    if( not (*self).next.HasValue() ){
      (*self).next = Optional(ListElement*).Create( heap.New(ListElement.Create(value)) );
      return;
    }
    var iter: Optional(ListElement*) = (*self).next;
    var node: auto = (iter.Get());
    while((*node).next.HasValue()){
      iter = (*node).next;
      node = (iter.Get());
    }
    var v : ListElement* = iter.Get();
    var x: Optional(ListElement*) = Optional(ListElement*).Create( heap.New(ListElement.Create(value) ));
    (*v).set(x);
  }

  fn print[self: Self] (){
    var iter: Optional(ListElement*) = self.next;
    while(iter.HasValue()){
      var node: auto = *(iter.Get());
      Print("{0}",node.element.Get());
      var node_ptr: auto = iter.Get();
      iter = node.next;
    }
  }

  fn clean[addr self: Self*] (){
    var head: auto = (*self).next;
    while(head.HasValue()){
      var tmp: auto = head;
      head = (*head.Get()).next;
      heap.Delete(tmp.Get());
    }
    (*self).next= Optional(ListElement*).CreateEmpty();
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
