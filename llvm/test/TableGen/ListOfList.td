// RUN llvm-tblgen %s | FileCheck %s

// RUN: llvm-tblgen %s | grep "foo" | count 1

class Base<string t> {
  string text = t;
}

class Derived<list<list<string>> thetext> : Base<thetext[0][0]>;

def FOO : Derived<[["foo"]]>;

// CHECK: text = "foo"
