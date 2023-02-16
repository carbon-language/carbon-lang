// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This doesn't test trace ouptut because it's too slow.
// NOAUTOUPDATE
// RUN: %{explorer-run}
// CHECK:STDOUT: String list length is: 7
// CHECK:STDOUT: This is a String Entry 4
// CHECK:STDOUT: Value 1337
// CHECK:STDOUT: result: 0
package ExplorerTest api;

class Node(T:! type) {
  fn Create(value: T)-> Node(T) {
    return {
      .value = Optional(T).Create(value),
      .next = Optional(Node(T)*).CreateEmpty(),
      .prev = Optional(Node(T)*).CreateEmpty()
    };
  }
  fn set_next[addr self: Self*](n: Optional(Node(T)*)) {
    (*self).next = n;
  }
  fn set_prev[addr self: Self*](n: Optional(Node(T)*)) {
    (*self).prev = n;
  }
  var value: Optional(T);
  var next: Optional(Node(T)*);
  var prev: Optional(Node(T)*);
}

class LinkedList(T:! type) {
  fn Create() -> LinkedList(T) {
    return {
      .head = Optional(Node(T)*).CreateEmpty(),
      .tail = Optional(Node(T)*).CreateEmpty(),
      .len = 0
    };
  }
  fn PushBack[addr self: Self*](value:T) {
    (*self).len = (*self).len + 1;
    if(not (*self).head.HasValue()) {
      (*self).head = Optional(Node(T)*).Create(heap.New(Node(T).Create(value)));
      (*self).tail = (*self).head;
      return;
    }
    var last: Optional(Node(T)*) = (*self).tail;
    var last_value:Node(T)* = last.Get();
    var v_wrapped:Optional(Node(T)*) = Optional(Node(T)*).Create(heap.New(Node(T).Create(value)));
    (*last_value).set_next(v_wrapped);
    (*v_wrapped.Get()).set_prev(last);
    (*self).tail = v_wrapped;
  }
  fn PushFront[addr self: Self*](value:T) {
    (*self).len = (*self).len + 1;
    if(not (*self).head.HasValue()) {
      (*self).head = Optional(Node(T)*).Create(heap.New(Node(T).Create(value)));
      (*self).tail = (*self).head;
      return;
    }
    var v_wrapped:Optional(Node(T)*) = Optional(Node(T)*).Create(heap.New(Node(T).Create(value)));
    var current_head: Optional(Node(T)*) = (*self).head;
    var current_head_value: Node(T)* = current_head.Get();
    (*v_wrapped.Get()).set_next(current_head);
    (*current_head_value).set_prev(v_wrapped);
    (*self).head = v_wrapped;
  }
  fn Clear[addr self: Self*]() {
    if((*self).len == 0) {
      return;
    }
    var iter: auto = (*self).head;
    while(iter.HasValue()) {
      var current: auto = iter;
      iter = (*iter.Get()).next;
      heap.Delete(current.Get());
    }
    (*self).head = Optional(Node(T)*).CreateEmpty();
    (*self).tail = Optional(Node(T)*).CreateEmpty();
    (*self).len = 0;
  }
  fn Length[self: Self]() -> i32 {
    return self.len;
  }
  var head: Optional(Node(T)*);
  var tail: Optional(Node(T)*);
  var len: i32;
}

// This function exists as a helper the context of this test case specifically and does not represent
// the common case for index based lookups in list like structures.
fn GetListEntryByIndex[T:! type](list: LinkedList(T)*, target_index: i32) -> Optional(T) {
  let list_length: i32 = (*list).Length();
  if(target_index > list_length - 1 or target_index < 0) {
    // Not in possible range
    return Optional(T).CreateEmpty();
  }
  var search_backwards:bool = target_index > list_length / 2;
  var iter:Optional(Node(T)*) = if search_backwards then (*list).tail else (*list).head;
  var c:i32 = if search_backwards then list_length -1 else 0;
  while(c != target_index) {
    var node: Node(T) = *(iter.Get());
    if(search_backwards) {
      c = c - 1;
      iter = node.prev;
    } else {
      iter = node.next;
      c = c + 1;
    }
  }

  return (*(iter.Get())).value;
}

fn Main() -> i32 {
  var test_list:auto = LinkedList(String).Create();
  test_list.PushBack("This is a String Entry 0");
  test_list.PushBack("This is a String Entry 1");
  test_list.PushBack("This is a String Entry 2");
  test_list.PushBack("This is a String Entry 3");
  test_list.PushBack("This is a String Entry 4");
  test_list.PushBack("This is a String Entry 5");
  test_list.PushFront("This is a prepended String -1");
  Print("String list length is: {0}", test_list.Length());
  var search_index:i32 = 5;
  var retrieved_entry: auto = GetListEntryByIndex(&test_list,search_index);
  if(retrieved_entry.HasValue()) {
    var retrieved_value:String = retrieved_entry.Get();
    Print(retrieved_value);
  } else {
    Print("No entry found in String list for index {0}!", search_index);
  }
  test_list.Clear();

  var second_list: auto = LinkedList(i32*).Create();
  var number: i32 = 1337;
  var number_ptr: i32* = &number;
  second_list.PushBack(number_ptr);
  var retrieved_ptr:i32* = GetListEntryByIndex(&second_list, 0).Get();
  var value: i32 = *retrieved_ptr;
  Print("Value {0}", value);
  return 0;
}
