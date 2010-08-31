// Test is line- and column-sensitive; see below.

namespace std {
  namespace rel_ops {
    void f();
  }
}

namespace std {
  void g();
}

// FIXME: using directives, namespace aliases

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: load-namespaces.cpp:3:11: Namespace=std:3:11 (Definition) Extent=[3:11 - 7:2]
// CHECK: load-namespaces.cpp:4:13: Namespace=rel_ops:4:13 (Definition) Extent=[4:13 - 6:4]
// CHECK: load-namespaces.cpp:5:10: FunctionDecl=f:5:10 Extent=[5:10 - 5:13]
// CHECK: load-namespaces.cpp:9:11: Namespace=std:9:11 (Definition) Extent=[9:11 - 11:2]
// CHECK: load-namespaces.cpp:10:8: FunctionDecl=g:10:8 Extent=[10:8 - 10:11]


