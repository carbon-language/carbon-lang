// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodule-cache-path %t -I %S/Inputs -verify %s -Wno-objc-root-class
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodule-cache-path %t -I %S/Inputs -emit-llvm %s -o - -Wno-objc-root-class | grep pendingInstantiation | FileCheck %s

@__experimental_modules_import templates_left;
@__experimental_modules_import templates_right;


void testTemplateClasses() {
  Vector<int> vec_int;
  vec_int.push_back(0);

  List<bool> list_bool;
  list_bool.push_back(false);

  N::Set<char> set_char;
  set_char.insert('A');
}

void testPendingInstantiations() {
  // CHECK: call
  // CHECK: call
  // CHECK: {{define .*pendingInstantiation.*[(]i}}
  // CHECK: {{define .*pendingInstantiation.*[(]double}}
  // CHECK: call
  triggerPendingInstantiation();
  triggerPendingInstantiationToo();
}
