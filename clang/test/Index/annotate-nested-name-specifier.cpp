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

namespace outer {
  namespace inner {
    template<typename T, unsigned N>
    struct array {
      void foo();
      static int max_size;
    };
  }
}

template<typename T, unsigned N>
void outer::inner::array<T, N>::foo() {
}

template<typename T, unsigned N>
int outer::inner::array<T, N>::max_size = 17;

template<typename T>
struct X2 : outer::inner::vector<T> {
  typedef T type;
  using typename outer::inner::vector<type>::iterator;
  using outer::inner::vector<type>::push_back;
};

namespace outer {
  namespace inner {
    namespace secret {
    }
  }
}

using namespace outer_alias::inner::secret;
namespace super_secret = outer_alias::inner::secret;

template<typename T>
struct X3 {
  void f(T *t) {
    t->::outer_alias::inner::template vector<T>::~vector<T>();
  }
};

// RUN: c-index-test -test-annotate-tokens=%s:13:1:60:1 %s | FileCheck %s

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

// FIXME: Check nested-name-specifiers on VarDecl, CXXMethodDecl.

// CHECK: Keyword: "using" [40:3 - 40:8] UsingDeclaration=iterator:40:46
// CHECK: Keyword: "typename" [40:9 - 40:17] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "outer" [40:18 - 40:23] NamespaceRef=outer:20:11
// CHECK: Punctuation: "::" [40:23 - 40:25] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "inner" [40:25 - 40:30] NamespaceRef=inner:21:13
// CHECK: Punctuation: "::" [40:30 - 40:32] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "vector" [40:32 - 40:38] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [40:38 - 40:39] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "type" [40:39 - 40:43] TypeRef=type:39:13
// CHECK: Punctuation: ">" [40:43 - 40:44] UsingDeclaration=iterator:40:46
// CHECK: Punctuation: "::" [40:44 - 40:46] UsingDeclaration=iterator:40:46
// CHECK: Identifier: "iterator" [40:46 - 40:54] UsingDeclaration=iterator:40:46
// CHECK: Punctuation: ";" [40:54 - 40:55] ClassTemplate=X2:38:8 (Definition)
// CHECK: Keyword: "using" [41:3 - 41:8] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "outer" [41:9 - 41:14] NamespaceRef=outer:20:11
// CHECK: Punctuation: "::" [41:14 - 41:16] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "inner" [41:16 - 41:21] NamespaceRef=inner:21:13
// CHECK: Punctuation: "::" [41:21 - 41:23] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "vector" [41:23 - 41:29] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [41:29 - 41:30] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "type" [41:30 - 41:34] TypeRef=type:39:13
// CHECK: Punctuation: ">" [41:34 - 41:35] UsingDeclaration=push_back:41:37
// CHECK: Punctuation: "::" [41:35 - 41:37] UsingDeclaration=push_back:41:37
// CHECK: Identifier: "push_back" [41:37 - 41:46] UsingDeclaration=push_back:41:37

// Using directive
// CHECK: Keyword: "using" [51:1 - 51:6] UsingDirective=:51:37
// CHECK: Keyword: "namespace" [51:7 - 51:16] UsingDirective=:51:37
// CHECK: Identifier: "outer_alias" [51:17 - 51:28] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [51:28 - 51:30] UsingDirective=:51:37
// CHECK: Identifier: "inner" [51:30 - 51:35] NamespaceRef=inner:45:13
// CHECK: Punctuation: "::" [51:35 - 51:37] UsingDirective=:51:37
// CHECK: Identifier: "secret" [51:37 - 51:43] NamespaceRef=secret:46:15

// Namespace alias
// CHECK: Keyword: "namespace" [52:1 - 52:10] NamespaceAlias=super_secret:52:11
// CHECK: Identifier: "super_secret" [52:11 - 52:23] NamespaceAlias=super_secret:52:11
// CHECK: Punctuation: "=" [52:24 - 52:25] NamespaceAlias=super_secret:52:11
// CHECK: Identifier: "outer_alias" [52:26 - 52:37] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [52:37 - 52:39] NamespaceAlias=super_secret:52:11
// CHECK: Identifier: "inner" [52:39 - 52:44] NamespaceRef=inner:45:13
// CHECK: Punctuation: "::" [52:44 - 52:46] NamespaceAlias=super_secret:52:11
// CHECK: Identifier: "secret" [52:46 - 52:52] NamespaceRef=secret:46:15
// CHECK: Punctuation: ";" [52:52 - 52:53]

// Pseudo-destructor
// CHECK: Identifier: "t" [57:5 - 57:6] DeclRefExpr=t:56:13
// CHECK: Punctuation: "->" [57:6 - 57:8] UnexposedExpr=
// CHECK: Punctuation: "::" [57:8 - 57:10] UnexposedExpr=
// CHECK: Identifier: "outer_alias" [57:10 - 57:21] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [57:21 - 57:23] UnexposedExpr=
// CHECK: Identifier: "inner" [57:23 - 57:28] NamespaceRef=inner:45:13
// CHECK: Punctuation: "::" [57:28 - 57:30] UnexposedExpr=
// CHECK: Keyword: "template" [57:30 - 57:38] UnexposedExpr=
// CHECK: Identifier: "vector" [57:39 - 57:45] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [57:45 - 57:46] UnexposedExpr=
// CHECK: Identifier: "T" [57:46 - 57:47] UnexposedExpr=
// CHECK: Punctuation: ">" [57:47 - 57:48] UnexposedExpr=
// CHECK: Punctuation: "::" [57:48 - 57:50] UnexposedExpr=
// CHECK: Punctuation: "~" [57:50 - 57:51] UnexposedExpr=
// CHECK: Identifier: "vector" [57:51 - 57:57] TemplateRef=vector:4:12
// CHECK: Punctuation: "<" [57:57 - 57:58] UnexposedExpr=
// CHECK: Identifier: "T" [57:58 - 57:59] UnexposedExpr=
// CHECK: Punctuation: ">" [57:59 - 57:60] UnexposedExpr=
// CHECK: Punctuation: "(" [57:60 - 57:61] CallExpr=
// CHECK: Punctuation: ")" [57:61 - 57:62] CallExpr=
