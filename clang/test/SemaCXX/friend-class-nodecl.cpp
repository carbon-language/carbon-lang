// RUN: clang-cc -ast-print %s -o %t
// RUN: not grep '^ *class B' %t

// Tests that the tag decls in friend declarations aren't added to the
// declaring class's decl chain.

class A {
  friend class B;
};

