namespace outer {
  namespace inner {
    template<typename T>
    struct vector {
      typedef T* iterator;
    };
  }
}

namespace outer_alias = outer;

struct X { };

using outer_alias::inner::vector;

struct X_vector : outer_alias::inner::vector<X> {
  using outer_alias::inner::vector<X>::iterator;
};



// RUN: c-index-test -test-annotate-tokens=%s:13:1:19:1 %s | FileCheck %s

// CHECK: Keyword: "using" [14:1 - 14:6] UsingDeclaration=vector[4:12]
// CHECK: Identifier: "outer_alias" [14:7 - 14:18] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [14:18 - 14:20] UsingDeclaration=vector[4:12]
// CHECK: Identifier: "inner" [14:20 - 14:25] NamespaceRef=inner:2:13
// CHECK: Punctuation: "::" [14:25 - 14:27] UsingDeclaration=vector[4:12]
// CHECK: Identifier: "vector" [14:27 - 14:33] OverloadedDeclRef=vector[4:12]
// CHECK: Punctuation: ";" [14:33 - 14:34]
// FIXME: Base specifiers, too
// CHECK: Keyword: "using" [17:3 - 17:8] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "outer_alias" [17:9 - 17:20] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [17:20 - 17:22] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "inner" [17:22 - 17:27] NamespaceRef=inner:2:13
// CHECK: Punctuation: "::" [17:27 - 17:29] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "vector" [17:29 - 17:35] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [17:35 - 17:36] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "X" [17:36 - 17:37] TypeRef=struct X:12:8
// CHECK: Punctuation: ">" [17:37 - 17:38] UsingDeclaration=iterator[5:18]
// CHECK: Punctuation: "::" [17:38 - 17:40] UsingDeclaration=iterator[5:18]
// CHECK: Identifier: "iterator" [17:40 - 17:48] OverloadedDeclRef=iterator[5:18]
