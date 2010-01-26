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

// RUN: c-index-test -test-annotate-tokens=%s:4:1:9:32 %s | FileCheck %s
// CHECK: Identifier: "T" [4:3 - 4:3] TypeRef=T:1:13
// CHECK: Punctuation: "*" [4:4 - 4:4]
// CHECK: Identifier: "t_ptr" [4:6 - 4:10] VarDecl=t_ptr:4:6 (Definition)
// CHECK: Punctuation: "=" [4:12 - 4:12]
// CHECK: Punctuation: "(" [4:14 - 4:14]
// CHECK: Identifier: "T" [4:15 - 4:15] TypeRef=T:1:13
// CHECK: Punctuation: "*" [4:17 - 4:17]
// CHECK: Punctuation: ")" [4:18 - 4:18]
// CHECK: Identifier: "ptr" [4:19 - 4:21] DeclRefExpr=ptr:3:14
// CHECK: Punctuation: ";" [4:22 - 4:22]
// CHECK: Punctuation: "(" [5:3 - 5:3]
// CHECK: Keyword: "void" [5:4 - 5:7]
// CHECK: Punctuation: ")" [5:8 - 5:8]
// CHECK: Keyword: "sizeof" [5:9 - 5:14]
// CHECK: Punctuation: "(" [5:15 - 5:15]
// CHECK: Identifier: "T" [5:16 - 5:16] TypeRef=T:1:13
// CHECK: Punctuation: ")" [5:17 - 5:17]
// CHECK: Punctuation: ";" [5:18 - 5:18]
// CHECK: Comment: "/* A comment */" [6:3 - 6:17]
// CHECK: Keyword: "struct" [7:3 - 7:8]
// CHECK: Identifier: "X" [7:10 - 7:10] TypeRef=struct X:2:8
// CHECK: Identifier: "x" [7:12 - 7:12] VarDecl=x:7:12 (Definition)
// CHECK: Punctuation: "=" [7:14 - 7:14]
// CHECK: Punctuation: "(" [7:16 - 7:16]
// CHECK: Keyword: "struct" [7:17 - 7:22]
// CHECK: Identifier: "X" [7:24 - 7:24] TypeRef=struct X:2:8
// CHECK: Punctuation: ")" [7:25 - 7:25]
// CHECK: Punctuation: "{" [7:26 - 7:26]
// CHECK: Literal: "1" [7:27 - 7:27]
// CHECK: Punctuation: "," [7:28 - 7:28]
// CHECK: Literal: "2" [7:30 - 7:30]
// CHECK: Punctuation: "}" [7:31 - 7:31]
// CHECK: Punctuation: ";" [7:32 - 7:32]
// CHECK: Keyword: "void" [8:3 - 8:6]
// CHECK: Punctuation: "*" [8:8 - 8:8]
// CHECK: Identifier: "xx" [8:9 - 8:10] VarDecl=xx:8:9 (Definition)
// CHECK: Punctuation: "=" [8:12 - 8:12]
// CHECK: Identifier: "ptr" [8:14 - 8:16] DeclRefExpr=ptr:3:14
// CHECK: Punctuation: "?" [8:18 - 8:18]
// CHECK: Punctuation: ":" [8:20 - 8:20]
// CHECK: Punctuation: "&" [8:22 - 8:22]
// CHECK: Identifier: "x" [8:23 - 8:23] DeclRefExpr=x:7:12
// CHECK: Punctuation: ";" [8:24 - 8:24]
// CHECK: Keyword: "const" [9:3 - 9:7]
// CHECK: Keyword: "char" [9:9 - 9:12]
// CHECK: Punctuation: "*" [9:14 - 9:14]
// CHECK: Identifier: "hello" [9:16 - 9:20] VarDecl=hello:9:16 (Definition)
// CHECK: Punctuation: "=" [9:22 - 9:22]
// CHECK: Literal: ""Hello"" [9:24 - 9:30]
// CHECK: Punctuation: ";" [9:31 - 9:31]
// CHECK: Punctuation: "}" [10:1 - 10:1]
