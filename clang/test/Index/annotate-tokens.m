@interface Foo
- (int)compare:(Foo*)other;
@end

@implementation Foo
- (int)compare:(Foo*)other {
  return 0;
  (void)@encode(Foo);
}
@end

// From <rdar://problem/7971430>, the 'barType' referenced in the ivar
// declarations should be annotated as TypeRefs.
typedef int * barType;
@interface Bar
{
    barType iVar;
    barType iVar1, iVar2;
}
@end
@implementation Bar
- (void) method
{
    barType local = iVar;
}
@end

// RUN: c-index-test -test-annotate-tokens=%s:1:1:27:1 %s | FileCheck %s
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
// CHECK: Keyword: "typedef" [14:1 - 14:8]
// CHECK: Keyword: "int" [14:9 - 14:12]
// CHECK: Punctuation: "*" [14:13 - 14:14]
// CHECK: Identifier: "barType" [14:15 - 14:22] TypedefDecl=barType:14:15 (Definition)
// CHECK: Punctuation: ";" [14:22 - 14:23]
// CHECK: Punctuation: "@" [15:1 - 15:2] ObjCInterfaceDecl=Bar:15:12
// CHECK: Keyword: "interface" [15:2 - 15:11] ObjCInterfaceDecl=Bar:15:12
// CHECK: Identifier: "Bar" [15:12 - 15:15] ObjCInterfaceDecl=Bar:15:12
// CHECK: Punctuation: "{" [16:1 - 16:2] ObjCInterfaceDecl=Bar:15:12
// CHECK: Identifier: "barType" [17:5 - 17:12] TypeRef=barType:14:15
// CHECK: Identifier: "iVar" [17:13 - 17:17] ObjCIvarDecl=iVar:17:13 (Definition)
// CHECK: Punctuation: ";" [17:17 - 17:18] ObjCInterfaceDecl=Bar:15:12
// CHECK: Identifier: "barType" [18:5 - 18:12] TypeRef=barType:14:15
// CHECK: Identifier: "iVar1" [18:13 - 18:18] ObjCIvarDecl=iVar1:18:13 (Definition)
// CHECK: Punctuation: "," [18:18 - 18:19] ObjCIvarDecl=iVar2:18:20 (Definition)
// CHECK: Identifier: "iVar2" [18:20 - 18:25] ObjCIvarDecl=iVar2:18:20 (Definition)
// CHECK: Punctuation: ";" [18:25 - 18:26] ObjCInterfaceDecl=Bar:15:12
// CHECK: Punctuation: "}" [19:1 - 19:2] ObjCInterfaceDecl=Bar:15:12
// CHECK: Punctuation: "@" [20:1 - 20:2] ObjCInterfaceDecl=Bar:15:12
// CHECK: Keyword: "end" [20:2 - 20:5] ObjCInterfaceDecl=Bar:15:12
// CHECK: Punctuation: "@" [21:1 - 21:2] ObjCImplementationDecl=Bar:21:1 (Definition)
// CHECK: Keyword: "implementation" [21:2 - 21:16] ObjCImplementationDecl=Bar:21:1 (Definition)
// CHECK: Identifier: "Bar" [21:17 - 21:20] ObjCImplementationDecl=Bar:21:1 (Definition)
// CHECK: Punctuation: "-" [22:1 - 22:2] ObjCInstanceMethodDecl=method:22:1 (Definition)
// CHECK: Punctuation: "(" [22:3 - 22:4] ObjCInstanceMethodDecl=method:22:1 (Definition)
// CHECK: Keyword: "void" [22:4 - 22:8] ObjCInstanceMethodDecl=method:22:1 (Definition)
// CHECK: Punctuation: ")" [22:8 - 22:9] ObjCInstanceMethodDecl=method:22:1 (Definition)
// CHECK: Identifier: "method" [22:10 - 22:16] ObjCInstanceMethodDecl=method:22:1 (Definition)
// CHECK: Punctuation: "{" [23:1 - 23:2] UnexposedStmt=
// CHECK: Identifier: "barType" [24:5 - 24:12] TypeRef=barType:14:15
// CHECK: Identifier: "local" [24:13 - 24:18] VarDecl=local:24:13 (Definition)
// CHECK: Punctuation: "=" [24:19 - 24:20] VarDecl=local:24:13 (Definition)
// CHECK: Identifier: "iVar" [24:21 - 24:25] MemberRefExpr=iVar:17:13
// CHECK: Punctuation: ";" [24:25 - 24:26] UnexposedStmt=
// CHECK: Punctuation: "}" [25:1 - 25:2] UnexposedStmt=
// CHECK: Punctuation: "@" [26:1 - 26:2] ObjCImplementationDecl=Bar:21:1 (Definition)
// CHECK: Keyword: "end" [26:2 - 26:5]

