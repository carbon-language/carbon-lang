@interface Foo
- (int)compare:(Foo*)other;
@end

@implementation Foo
- (int)compare:(Foo*)other {
  return 0;
  (void)@encode(Foo);
}
@end

// RUN: c-index-test -test-annotate-tokens=%s:1:1:10:5 %s | FileCheck %s
// CHECK: Punctuation: "@" [1:1 - 1:2] ObjCInterfaceDecl=Foo:1:12
// CHECK: Keyword: "interface" [1:2 - 1:11] ObjCInterfaceDecl=Foo:1:12
// CHECK: Identifier: "Foo" [1:12 - 1:15] ObjCInterfaceDecl=Foo:1:12
// CHECK: Punctuation: "-" [2:1 - 2:2] ObjCInstanceMethodDecl=compare::2:1
// CHECK: Punctuation: "(" [2:3 - 2:4] ObjCInstanceMethodDecl=compare::2:1
// CHECK: Keyword: "int" [2:4 - 2:7] ObjCInstanceMethodDecl=compare::2:1
// CHECK: Punctuation: ")" [2:7 - 2:8] ObjCInstanceMethodDecl=compare::2:1
// CHECK: Identifier: "compare" [2:8 - 2:15] ObjCInstanceMethodDecl=compare::2:1
// CHECK: Punctuation: ":" [2:15 - 2:16] ObjCInstanceMethodDecl=compare::2:1
// CHECK: Punctuation: "(" [2:16 - 2:17] ObjCInstanceMethodDecl=compare::2:1
// CHECK: Identifier: "Foo" [2:17 - 2:20] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [2:20 - 2:21] ParmDecl=other:2:22 (Definition)
// CHECK: Punctuation: ")" [2:21 - 2:22] ParmDecl=other:2:22 (Definition)
// CHECK: Identifier: "other" [2:22 - 2:27] ParmDecl=other:2:22 (Definition)
// CHECK: Punctuation: ";" [2:27 - 2:28] ObjCInstanceMethodDecl=compare::2:1
// CHECK: Punctuation: "@" [3:1 - 3:2] ObjCInterfaceDecl=Foo:1:12
// CHECK: Keyword: "end" [3:2 - 3:5] ObjCInterfaceDecl=Foo:1:12
// CHECK: Punctuation: "@" [5:1 - 5:2] ObjCImplementationDecl=Foo:5:1 (Definition)
// CHECK: Keyword: "implementation" [5:2 - 5:16] ObjCImplementationDecl=Foo:5:1 (Definition)
// CHECK: Identifier: "Foo" [5:17 - 5:20] ObjCImplementationDecl=Foo:5:1 (Definition)
// CHECK: Punctuation: "-" [6:1 - 6:2] ObjCInstanceMethodDecl=compare::6:1 (Definition)
// CHECK: Punctuation: "(" [6:3 - 6:4] ObjCInstanceMethodDecl=compare::6:1 (Definition)
// CHECK: Keyword: "int" [6:4 - 6:7] ObjCInstanceMethodDecl=compare::6:1 (Definition)
// CHECK: Punctuation: ")" [6:7 - 6:8] ObjCInstanceMethodDecl=compare::6:1 (Definition)
// CHECK: Identifier: "compare" [6:8 - 6:15] ObjCInstanceMethodDecl=compare::6:1 (Definition)
// CHECK: Punctuation: ":" [6:15 - 6:16] ObjCInstanceMethodDecl=compare::6:1 (Definition)
// CHECK: Punctuation: "(" [6:16 - 6:17] ObjCInstanceMethodDecl=compare::6:1 (Definition)
// CHECK: Identifier: "Foo" [6:17 - 6:20] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [6:20 - 6:21] ParmDecl=other:6:22 (Definition)
// CHECK: Punctuation: ")" [6:21 - 6:22] ParmDecl=other:6:22 (Definition)
// CHECK: Identifier: "other" [6:22 - 6:27] ParmDecl=other:6:22 (Definition)
// CHECK: Punctuation: "{" [6:28 - 6:29] UnexposedStmt=
// CHECK: Keyword: "return" [7:3 - 7:9] UnexposedStmt=
// CHECK: Literal: "0" [7:10 - 7:11] UnexposedExpr=
// CHECK: Punctuation: ";" [7:11 - 7:12] UnexposedStmt=
// CHECK: Punctuation: "(" [8:3 - 8:4] UnexposedExpr=
// CHECK: Keyword: "void" [8:4 - 8:8] UnexposedExpr=
// CHECK: Punctuation: ")" [8:8 - 8:9] UnexposedExpr=
// CHECK: Punctuation: "@" [8:9 - 8:10] UnexposedExpr=
// CHECK: Keyword: "encode" [8:10 - 8:16] UnexposedExpr=
// CHECK: Punctuation: "(" [8:16 - 8:17] UnexposedExpr=
// CHECK: Identifier: "Foo" [8:17 - 8:20] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: ")" [8:20 - 8:21] UnexposedExpr=
// CHECK: Punctuation: ";" [8:21 - 8:22] UnexposedStmt=
// CHECK: Punctuation: "}" [9:1 - 9:2] UnexposedStmt=
// CHECK: Punctuation: "@" [10:1 - 10:2] ObjCImplementationDecl=Foo:5:1 (Definition)
// CHECK: Keyword: "end" [10:2 - 10:5]
