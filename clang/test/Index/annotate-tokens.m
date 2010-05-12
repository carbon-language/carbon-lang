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

// From <rdar://problem/7967123>.  The ranges for attributes are not
// currently stored, causing most of the tokens to be falsely annotated.
// Since there are no source ranges for attributes, we currently don't
// annotate them.
@interface IBActionTests
- (IBAction) actionMethod:(id)arg;
- (void)foo:(int)x;
@end
extern int ibaction_test(void);
@implementation IBActionTests
- (IBAction) actionMethod:(id)arg
{
    ibaction_test();
    [self foo:0];
}
- (void) foo:(int)x
{
  (void) x;
}
@end

// From <rdar://problem/7961995>.  Essentially the same issue as 7967123,
// but impacting code marked as IBOutlets.
@interface IBOutletTests
{
    IBOutlet char * anOutlet;
}
- (IBAction) actionMethod:(id)arg;
@property IBOutlet int * aPropOutlet;
@end

// RUN: c-index-test -test-annotate-tokens=%s:1:1:58:1 %s -DIBOutlet='__attribute__((iboutlet))' -DIBAction='void)__attribute__((ibaction)' | FileCheck %s
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
// CHECK: Punctuation: "@" [32:1 - 32:2] ObjCInterfaceDecl=IBActionTests:32:12
// CHECK: Keyword: "interface" [32:2 - 32:11] ObjCInterfaceDecl=IBActionTests:32:12
// CHECK: Identifier: "IBActionTests" [32:12 - 32:25] ObjCInterfaceDecl=IBActionTests:32:12
// CHECK: Punctuation: "-" [33:1 - 33:2] ObjCInstanceMethodDecl=actionMethod::33:1
// CHECK: Punctuation: "(" [33:3 - 33:4] ObjCInstanceMethodDecl=actionMethod::33:1
// CHECK: Identifier: "IBAction" [33:4 - 33:12] macro instantiation=IBAction
// CHECK: Punctuation: ")" [33:12 - 33:13] ObjCInstanceMethodDecl=actionMethod::33:1
// CHECK: Identifier: "actionMethod" [33:14 - 33:26] ObjCInstanceMethodDecl=actionMethod::33:1
// CHECK: Punctuation: ":" [33:26 - 33:27] ObjCInstanceMethodDecl=actionMethod::33:1
// CHECK: Punctuation: "(" [33:27 - 33:28] ObjCInstanceMethodDecl=actionMethod::33:1
// CHECK: Identifier: "id" [33:28 - 33:30] TypeRef=id:0:0
// CHECK: Punctuation: ")" [33:30 - 33:31] ParmDecl=arg:33:31 (Definition)
// CHECK: Identifier: "arg" [33:31 - 33:34] ParmDecl=arg:33:31 (Definition)
// CHECK: Punctuation: ";" [33:34 - 33:35] ObjCInstanceMethodDecl=actionMethod::33:1
// CHECK: Punctuation: "-" [34:1 - 34:2] ObjCInstanceMethodDecl=foo::34:1
// CHECK: Punctuation: "(" [34:3 - 34:4] ObjCInstanceMethodDecl=foo::34:1
// CHECK: Keyword: "void" [34:4 - 34:8] ObjCInstanceMethodDecl=foo::34:1
// CHECK: Punctuation: ")" [34:8 - 34:9] ObjCInstanceMethodDecl=foo::34:1
// CHECK: Identifier: "foo" [34:9 - 34:12] ObjCInstanceMethodDecl=foo::34:1
// CHECK: Punctuation: ":" [34:12 - 34:13] ObjCInstanceMethodDecl=foo::34:1
// CHECK: Punctuation: "(" [34:13 - 34:14] ObjCInstanceMethodDecl=foo::34:1
// CHECK: Keyword: "int" [34:14 - 34:17] ParmDecl=x:34:18 (Definition)
// CHECK: Punctuation: ")" [34:17 - 34:18] ParmDecl=x:34:18 (Definition)
// CHECK: Identifier: "x" [34:18 - 34:19] ParmDecl=x:34:18 (Definition)
// CHECK: Punctuation: ";" [34:19 - 34:20] ObjCInstanceMethodDecl=foo::34:1
// CHECK: Punctuation: "@" [35:1 - 35:2] ObjCInterfaceDecl=IBActionTests:32:12
// CHECK: Keyword: "end" [35:2 - 35:5] ObjCInterfaceDecl=IBActionTests:32:12
// CHECK: Keyword: "extern" [36:1 - 36:7]
// CHECK: Keyword: "int" [36:8 - 36:11] FunctionDecl=ibaction_test:36:12
// CHECK: Identifier: "ibaction_test" [36:12 - 36:25] FunctionDecl=ibaction_test:36:12
// CHECK: Punctuation: "(" [36:25 - 36:26] FunctionDecl=ibaction_test:36:12
// CHECK: Keyword: "void" [36:26 - 36:30] FunctionDecl=ibaction_test:36:12
// CHECK: Punctuation: ")" [36:30 - 36:31] FunctionDecl=ibaction_test:36:12
// CHECK: Punctuation: ";" [36:31 - 36:32]
// CHECK: Punctuation: "@" [37:1 - 37:2] ObjCImplementationDecl=IBActionTests:37:1 (Definition)
// CHECK: Keyword: "implementation" [37:2 - 37:16] ObjCImplementationDecl=IBActionTests:37:1 (Definition)
// CHECK: Identifier: "IBActionTests" [37:17 - 37:30] ObjCImplementationDecl=IBActionTests:37:1 (Definition)
// CHECK: Punctuation: "-" [38:1 - 38:2] ObjCInstanceMethodDecl=actionMethod::38:1 (Definition)
// CHECK: Punctuation: "(" [38:3 - 38:4] ObjCInstanceMethodDecl=actionMethod::38:1 (Definition)
// CHECK: Identifier: "IBAction" [38:4 - 38:12] macro instantiation=IBAction
// CHECK: Punctuation: ")" [38:12 - 38:13] ObjCInstanceMethodDecl=actionMethod::38:1 (Definition)
// CHECK: Identifier: "actionMethod" [38:14 - 38:26] ObjCInstanceMethodDecl=actionMethod::38:1 (Definition)
// CHECK: Punctuation: ":" [38:26 - 38:27] ObjCInstanceMethodDecl=actionMethod::38:1 (Definition)
// CHECK: Punctuation: "(" [38:27 - 38:28] ObjCInstanceMethodDecl=actionMethod::38:1 (Definition)
// CHECK: Identifier: "id" [38:28 - 38:30] TypeRef=id:0:0
// CHECK: Punctuation: ")" [38:30 - 38:31] ParmDecl=arg:38:31 (Definition)
// CHECK: Identifier: "arg" [38:31 - 38:34] ParmDecl=arg:38:31 (Definition)
// CHECK: Punctuation: "{" [39:1 - 39:2] UnexposedStmt=
// CHECK: Identifier: "ibaction_test" [40:5 - 40:18] DeclRefExpr=ibaction_test:36:12
// CHECK: Punctuation: "(" [40:18 - 40:19] CallExpr=ibaction_test:36:12
// CHECK: Punctuation: ")" [40:19 - 40:20] CallExpr=ibaction_test:36:12
// CHECK: Punctuation: ";" [40:20 - 40:21] UnexposedStmt=
// CHECK: Punctuation: "[" [41:5 - 41:6] ObjCMessageExpr=foo::34:1
// CHECK: Identifier: "self" [41:6 - 41:10] DeclRefExpr=self:0:0
// CHECK: Identifier: "foo" [41:11 - 41:14] ObjCMessageExpr=foo::34:1
// CHECK: Punctuation: ":" [41:14 - 41:15] ObjCMessageExpr=foo::34:1
// CHECK: Literal: "0" [41:15 - 41:16] UnexposedExpr=
// CHECK: Punctuation: "]" [41:16 - 41:17] ObjCMessageExpr=foo::34:1
// CHECK: Punctuation: ";" [41:17 - 41:18] UnexposedStmt=
// CHECK: Punctuation: "}" [42:1 - 42:2] UnexposedStmt=
// CHECK: Punctuation: "-" [43:1 - 43:2] ObjCInstanceMethodDecl=foo::43:1 (Definition)
// CHECK: Punctuation: "(" [43:3 - 43:4] ObjCInstanceMethodDecl=foo::43:1 (Definition)
// CHECK: Keyword: "void" [43:4 - 43:8] ObjCInstanceMethodDecl=foo::43:1 (Definition)
// CHECK: Punctuation: ")" [43:8 - 43:9] ObjCInstanceMethodDecl=foo::43:1 (Definition)
// CHECK: Identifier: "foo" [43:10 - 43:13] ObjCInstanceMethodDecl=foo::43:1 (Definition)
// CHECK: Punctuation: ":" [43:13 - 43:14] ObjCInstanceMethodDecl=foo::43:1 (Definition)
// CHECK: Punctuation: "(" [43:14 - 43:15] ObjCInstanceMethodDecl=foo::43:1 (Definition)
// CHECK: Keyword: "int" [43:15 - 43:18] ParmDecl=x:43:19 (Definition)
// CHECK: Punctuation: ")" [43:18 - 43:19] ParmDecl=x:43:19 (Definition)
// CHECK: Identifier: "x" [43:19 - 43:20] ParmDecl=x:43:19 (Definition)
// CHECK: Punctuation: "{" [44:1 - 44:2] UnexposedStmt=
// CHECK: Punctuation: "(" [45:3 - 45:4] UnexposedExpr=x:43:19
// CHECK: Keyword: "void" [45:4 - 45:8] UnexposedExpr=x:43:19
// CHECK: Punctuation: ")" [45:8 - 45:9] UnexposedExpr=x:43:19
// CHECK: Identifier: "x" [45:10 - 45:11] DeclRefExpr=x:43:19
// CHECK: Punctuation: ";" [45:11 - 45:12] UnexposedStmt=
// CHECK: Punctuation: "}" [46:1 - 46:2] UnexposedStmt=
// CHECK: Punctuation: "@" [47:1 - 47:2] ObjCImplementationDecl=IBActionTests:37:1 (Definition)
// CHECK: Keyword: "end" [47:2 - 47:5]
// CHECK: Punctuation: "@" [51:1 - 51:2] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Keyword: "interface" [51:2 - 51:11] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Identifier: "IBOutletTests" [51:12 - 51:25] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "{" [52:1 - 52:2] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Identifier: "IBOutlet" [53:5 - 53:13] macro instantiation=IBOutlet:{{[0-9]+}}:{{[0-9]+}}
// CHECK: Keyword: "char" [53:14 - 53:18] ObjCIvarDecl=anOutlet:53:21 (Definition)
// CHECK: Punctuation: "*" [53:19 - 53:20] ObjCIvarDecl=anOutlet:53:21 (Definition)
// CHECK: Identifier: "anOutlet" [53:21 - 53:29] ObjCIvarDecl=anOutlet:53:21 (Definition)
// CHECK: Punctuation: ";" [53:29 - 53:30] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "}" [54:1 - 54:2] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "-" [55:1 - 55:2] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "(" [55:3 - 55:4] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Identifier: "IBAction" [55:4 - 55:12] macro instantiation=IBAction
// CHECK: Punctuation: ")" [55:12 - 55:13] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Identifier: "actionMethod" [55:14 - 55:26] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: ":" [55:26 - 55:27] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "(" [55:27 - 55:28] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Identifier: "id" [55:28 - 55:30] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: ")" [55:30 - 55:31] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Identifier: "arg" [55:31 - 55:34] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: ";" [55:34 - 55:35] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "@" [56:1 - 56:2] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Keyword: "property" [56:2 - 56:10] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Identifier: "IBOutlet" [56:11 - 56:19] macro instantiation=IBOutlet:{{[0-9]+}}:{{[0-9]+}}
// CHECK: Keyword: "int" [56:20 - 56:23] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "*" [56:24 - 56:25] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Identifier: "aPropOutlet" [56:26 - 56:37] ObjCPropertyDecl=aPropOutlet:56:26
// CHECK: Punctuation: ";" [56:37 - 56:38] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "@" [57:1 - 57:2] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Keyword: "end" [57:2 - 57:5] ObjCInterfaceDecl=IBOutletTests:51:12

