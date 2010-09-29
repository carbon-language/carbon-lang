// RUN: %llvmgcc -g -S -O2 %s -o - | FileCheck %s

class foo {
public:
      int bar(int x);
      static int baz(int x);
};

int foo::bar(int x) {
  // CHECK: {{i32 [0-9]+, i1 true(, i[0-9]+ [^\}]+[}]|[}]) ; \[ DW_TAG_subprogram \]}}
    return x*4 + 1;
}

int foo::baz(int x) {
  // CHECK: {{i32 [0-9]+, i1 true(, i[0-9]+ [^\},]+[}]|[}]) ; \[ DW_TAG_subprogram \]}}
    return x*4 + 1;
}

