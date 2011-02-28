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

namespace outer {
  namespace inner {
    void f(int);
    void f(double);
  }
}

template<typename T>
struct X4 {
  typedef T type;
  void g(int);
  void g(float);

  void h(T t) {
    ::outer_alias::inner::f(t);
    ::X4<type>::g(t);
    this->::X4<type>::g(t);
  }
};

typedef int Integer;
template<>
struct X4<Integer> {
  typedef Integer type;

  void g(int);
  void g(float);

  void h(type t) {
    ::outer_alias::inner::f(t);
    ::X4<type>::g(t);
    this->::X4<type>::g(t);
  }
};

// RUN: c-index-test -test-annotate-tokens=%s:13:1:93:1 %s | FileCheck %s

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

// Unresolved member and non-member references
// CHECK: Punctuation: "::" [75:5 - 75:7] UnexposedExpr=[63:10, 64:10]
// CHECK: Identifier: "outer_alias" [75:7 - 75:18] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [75:18 - 75:20] UnexposedExpr=[63:10, 64:10]
// CHECK: Identifier: "inner" [75:20 - 75:25] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [75:25 - 75:27] UnexposedExpr=[63:10, 64:10]
// CHECK: Identifier: "f" [75:27 - 75:28] OverloadedDeclRef=f[63:10, 64:10]
// CHECK: Punctuation: "(" [75:28 - 75:29] CallExpr=
// CHECK: Identifier: "t" [75:29 - 75:30] DeclRefExpr=t:74:12
// CHECK: Punctuation: ")" [75:30 - 75:31] CallExpr=
// CHECK: Punctuation: "::" [76:5 - 76:7] UnexposedExpr=[71:8, 72:8]
// CHECK: Identifier: "X4" [76:7 - 76:9] TemplateRef=X4:69:8
// CHECK: Punctuation: "<" [76:9 - 76:10] UnexposedExpr=[71:8, 72:8]
// CHECK: Identifier: "type" [76:10 - 76:14] TypeRef=type:70:13
// CHECK: Punctuation: ">" [76:14 - 76:15] UnexposedExpr=[71:8, 72:8]
// CHECK: Punctuation: "::" [76:15 - 76:17] UnexposedExpr=[71:8, 72:8]
// CHECK: Identifier: "g" [76:17 - 76:18] OverloadedDeclRef=g[71:8, 72:8]
// CHECK: Punctuation: "(" [76:18 - 76:19] CallExpr=
// CHECK: Identifier: "t" [76:19 - 76:20] DeclRefExpr=t:74:12
// CHECK: Punctuation: ")" [76:20 - 76:21] CallExpr=
// CHECK: Punctuation: ";" [76:21 - 76:22] UnexposedStmt=
// CHECK: Keyword: "this" [77:5 - 77:9] UnexposedExpr=
// CHECK: Punctuation: "->" [77:9 - 77:11] UnexposedExpr=
// CHECK: Punctuation: "::" [77:11 - 77:13] UnexposedExpr=
// CHECK: Identifier: "X4" [77:13 - 77:15] TemplateRef=X4:69:8
// CHECK: Punctuation: "<" [77:15 - 77:16] UnexposedExpr=
// CHECK: Identifier: "type" [77:16 - 77:20] TypeRef=type:70:13
// CHECK: Punctuation: ">" [77:20 - 77:21] UnexposedExpr=
// CHECK: Punctuation: "::" [77:21 - 77:23] UnexposedExpr=
// CHECK: Identifier: "g" [77:23 - 77:24] UnexposedExpr=
// CHECK: Punctuation: "(" [77:24 - 77:25] CallExpr=
// CHECK: Identifier: "t" [77:25 - 77:26] DeclRefExpr=t:74:12
// CHECK: Punctuation: ")" [77:26 - 77:27] CallExpr=

// Resolved member and non-member references
// CHECK: Punctuation: "::" [90:5 - 90:7] DeclRefExpr=f:63:10
// CHECK: Identifier: "outer_alias" [90:7 - 90:18] NamespaceRef=outer_alias:10:11
// CHECK: Punctuation: "::" [90:18 - 90:20] DeclRefExpr=f:63:10
// CHECK: Identifier: "inner" [90:20 - 90:25] NamespaceRef=inner:62:13
// CHECK: Punctuation: "::" [90:25 - 90:27] DeclRefExpr=f:63:10
// CHECK: Identifier: "f" [90:27 - 90:28] DeclRefExpr=f:63:10
// CHECK: Punctuation: "(" [90:28 - 90:29] CallExpr=f:63:10
// CHECK: Identifier: "t" [90:29 - 90:30] DeclRefExpr=t:89:15
// CHECK: Punctuation: ")" [90:30 - 90:31] CallExpr=f:63:10
// CHECK: Punctuation: ";" [90:31 - 90:32] UnexposedStmt=
// CHECK: Punctuation: "::" [91:5 - 91:7] MemberRefExpr=g:86:8
// CHECK: Identifier: "X4" [91:7 - 91:9] TemplateRef=X4:69:8
// CHECK: Punctuation: "<" [91:9 - 91:10] MemberRefExpr=g:86:8
// CHECK: Identifier: "type" [91:10 - 91:14] TypeRef=type:84:19
// CHECK: Punctuation: ">" [91:14 - 91:15] MemberRefExpr=g:86:8
// CHECK: Punctuation: "::" [91:15 - 91:17] MemberRefExpr=g:86:8
// CHECK: Identifier: "g" [91:17 - 91:18] MemberRefExpr=g:86:8
// CHECK: Punctuation: "(" [91:18 - 91:19] CallExpr=g:86:8
// CHECK: Identifier: "t" [91:19 - 91:20] DeclRefExpr=t:89:15
// CHECK: Punctuation: ")" [91:20 - 91:21] CallExpr=g:86:8
// CHECK: Punctuation: ";" [91:21 - 91:22] UnexposedStmt=
// CHECK: Keyword: "this" [92:5 - 92:9] UnexposedExpr=
// CHECK: Punctuation: "->" [92:9 - 92:11] MemberRefExpr=g:86:8
// CHECK: Punctuation: "::" [92:11 - 92:13] MemberRefExpr=g:86:8
// CHECK: Identifier: "X4" [92:13 - 92:15] TemplateRef=X4:69:8
// CHECK: Punctuation: "<" [92:15 - 92:16] MemberRefExpr=g:86:8
// CHECK: Identifier: "type" [92:16 - 92:20] TypeRef=type:84:19
// CHECK: Punctuation: ">" [92:20 - 92:21] MemberRefExpr=g:86:8
// CHECK: Punctuation: "::" [92:21 - 92:23] MemberRefExpr=g:86:8
// CHECK: Identifier: "g" [92:23 - 92:24] MemberRefExpr=g:86:8
// CHECK: Punctuation: "(" [92:24 - 92:25] CallExpr=g:86:8
// CHECK: Identifier: "t" [92:25 - 92:26] DeclRefExpr=t:89:15
// CHECK: Punctuation: ")" [92:26 - 92:27] CallExpr=g:86:8
