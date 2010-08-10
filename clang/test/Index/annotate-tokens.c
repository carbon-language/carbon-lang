typedef int T;
struct X { int a, b; };
void f(void *ptr) {
  T* t_ptr = (T *)ptr;
  (void)sizeof(T);
  /* A comment */
  struct X x = (struct X){1, 2};
  void *xx = ptr ? : &x;
  const char * hello = "Hello";
}

typedef int Int;
void g(int i, ...) {
  __builtin_va_list va;
  (void)__builtin_va_arg(va, Int);
  (void)__builtin_types_compatible_p(Int, Int);
}

// RUN: c-index-test -test-annotate-tokens=%s:4:1:17:1 %s | FileCheck %s
// CHECK: Identifier: "T" [4:3 - 4:4] TypeRef=T:1:13
// CHECK: Punctuation: "*" [4:4 - 4:5] VarDecl=t_ptr:4:6 (Definition)
// CHECK: Identifier: "t_ptr" [4:6 - 4:11] VarDecl=t_ptr:4:6 (Definition)
// CHECK: Punctuation: "=" [4:12 - 4:13] VarDecl=t_ptr:4:6 (Definition)
// CHECK: Punctuation: "(" [4:14 - 4:15] UnexposedExpr=ptr:3:14
// CHECK: Identifier: "T" [4:15 - 4:16] TypeRef=T:1:13
// CHECK: Punctuation: "*" [4:17 - 4:18] UnexposedExpr=ptr:3:14
// CHECK: Punctuation: ")" [4:18 - 4:19] UnexposedExpr=ptr:3:14
// CHECK: Identifier: "ptr" [4:19 - 4:22] DeclRefExpr=ptr:3:14
// CHECK: Punctuation: ";" [4:22 - 4:23] UnexposedStmt=
// CHECK: Punctuation: "(" [5:3 - 5:4] UnexposedExpr=
// CHECK: Keyword: "void" [5:4 - 5:8] UnexposedExpr=
// CHECK: Punctuation: ")" [5:8 - 5:9] UnexposedExpr=
// CHECK: Keyword: "sizeof" [5:9 - 5:15] UnexposedExpr=
// CHECK: Punctuation: "(" [5:15 - 5:16] UnexposedExpr=
// CHECK: Identifier: "T" [5:16 - 5:17] TypeRef=T:1:13
// CHECK: Punctuation: ")" [5:17 - 5:18] UnexposedExpr=
// CHECK: Punctuation: ";" [5:18 - 5:19] UnexposedStmt=
// CHECK: Comment: "/* A comment */" [6:3 - 6:18] UnexposedStmt=
// CHECK: Keyword: "struct" [7:3 - 7:9] UnexposedStmt=
// CHECK: Identifier: "X" [7:10 - 7:11] TypeRef=struct X:2:8
// CHECK: Identifier: "x" [7:12 - 7:13] VarDecl=x:7:12 (Definition)
// CHECK: Punctuation: "=" [7:14 - 7:15] VarDecl=x:7:12 (Definition)
// CHECK: Punctuation: "(" [7:16 - 7:17] UnexposedExpr=
// CHECK: Keyword: "struct" [7:17 - 7:23] UnexposedExpr=
// CHECK: Identifier: "X" [7:24 - 7:25] TypeRef=struct X:2:8
// CHECK: Punctuation: ")" [7:25 - 7:26] UnexposedExpr=
// CHECK: Punctuation: "{" [7:26 - 7:27] UnexposedExpr=
// CHECK: Literal: "1" [7:27 - 7:28] UnexposedExpr=
// CHECK: Punctuation: "," [7:28 - 7:29] UnexposedExpr=
// CHECK: Literal: "2" [7:30 - 7:31] UnexposedExpr=
// CHECK: Punctuation: "}" [7:31 - 7:32] UnexposedExpr=
// CHECK: Punctuation: ";" [7:32 - 7:33] UnexposedStmt=
// CHECK: Keyword: "void" [8:3 - 8:7] VarDecl=xx:8:9 (Definition)
// CHECK: Punctuation: "*" [8:8 - 8:9] VarDecl=xx:8:9 (Definition)
// CHECK: Identifier: "xx" [8:9 - 8:11] VarDecl=xx:8:9 (Definition)
// CHECK: Punctuation: "=" [8:12 - 8:13] VarDecl=xx:8:9 (Definition)
// CHECK: Identifier: "ptr" [8:14 - 8:17] DeclRefExpr=ptr:3:14
// CHECK: Punctuation: "?" [8:18 - 8:19] UnexposedExpr=
// CHECK: Punctuation: ":" [8:20 - 8:21] UnexposedExpr=
// CHECK: Punctuation: "&" [8:22 - 8:23] UnexposedExpr=
// CHECK: Identifier: "x" [8:23 - 8:24] DeclRefExpr=x:7:12
// CHECK: Punctuation: ";" [8:24 - 8:25] UnexposedStmt=
// CHECK: Keyword: "const" [9:3 - 9:8] UnexposedStmt=
// CHECK: Keyword: "char" [9:9 - 9:13] VarDecl=hello:9:16 (Definition)
// CHECK: Punctuation: "*" [9:14 - 9:15] VarDecl=hello:9:16 (Definition)
// CHECK: Identifier: "hello" [9:16 - 9:21] VarDecl=hello:9:16 (Definition)
// CHECK: Punctuation: "=" [9:22 - 9:23] VarDecl=hello:9:16 (Definition)
// CHECK: Literal: ""Hello"" [9:24 - 9:31] UnexposedExpr=
// CHECK: Punctuation: ";" [9:31 - 9:32] UnexposedStmt=
// CHECK: Punctuation: "}" [10:1 - 10:2] UnexposedStmt=
// CHECK: Keyword: "__builtin_va_arg" [15:9 - 15:25] UnexposedExpr=
// CHECK: Identifier: "Int" [15:30 - 15:33] TypeRef=Int:12:13
// CHECK: Keyword: "__builtin_types_compatible_p" [16:9 - 16:37] UnexposedExpr=
// CHECK: Identifier: "Int" [16:38 - 16:41] TypeRef=Int:12:13
// CHECK: Punctuation: "," [16:41 - 16:42] UnexposedExpr=
// CHECK: Identifier: "Int" [16:43 - 16:46] TypeRef=Int:12:13
// RUN: c-index-test -test-annotate-tokens=%s:4:1:165:32 %s | FileCheck %s
// RUN: c-index-test -test-annotate-tokens=%s:4:1:165:38 %s | FileCheck %s
