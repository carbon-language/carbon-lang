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
#define IBOutlet __attribute__((iboutlet))
#define IBAction void)__attribute__((ibaction)

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

// RUN: c-index-test -test-annotate-tokens=%s:1:1:58:1 %s | FileCheck %s
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
// CHECK: Punctuation: "#" [32:1 - 32:2] preprocessing directive=
// CHECK: Identifier: "define" [32:2 - 32:8] preprocessing directive=
// CHECK: Identifier: "IBOutlet" [32:9 - 32:17] macro definition=IBOutlet
// CHECK: Keyword: "__attribute__" [32:18 - 32:31] preprocessing directive=
// CHECK: Punctuation: "(" [32:31 - 32:32] preprocessing directive=
// CHECK: Punctuation: "(" [32:32 - 32:33] preprocessing directive=
// CHECK: Identifier: "iboutlet" [32:33 - 32:41] preprocessing directive=
// CHECK: Punctuation: ")" [32:41 - 32:42] preprocessing directive=
// CHECK: Punctuation: ")" [32:42 - 32:43] preprocessing directive=
// CHECK: Punctuation: "#" [33:1 - 33:2] preprocessing directive=
// CHECK: Identifier: "define" [33:2 - 33:8] preprocessing directive=
// CHECK: Identifier: "IBAction" [33:9 - 33:17] macro definition=IBAction
// CHECK: Keyword: "void" [33:18 - 33:22] preprocessing directive=
// CHECK: Punctuation: ")" [33:22 - 33:23] preprocessing directive=
// CHECK: Keyword: "__attribute__" [33:23 - 33:36] preprocessing directive=
// CHECK: Punctuation: "(" [33:36 - 33:37] preprocessing directive=
// CHECK: Punctuation: "(" [33:37 - 33:38] preprocessing directive=
// CHECK: Identifier: "ibaction" [33:38 - 33:46] preprocessing directive=
// CHECK: Punctuation: ")" [33:46 - 33:47] preprocessing directive=
// CHECK: Punctuation: "@" [35:1 - 35:2] ObjCInterfaceDecl=IBActionTests:35:12
// CHECK: Keyword: "interface" [35:2 - 35:11] ObjCInterfaceDecl=IBActionTests:35:12
// CHECK: Identifier: "IBActionTests" [35:12 - 35:25] ObjCInterfaceDecl=IBActionTests:35:12
// CHECK: Punctuation: "-" [36:1 - 36:2] ObjCInstanceMethodDecl=actionMethod::36:1
// CHECK: Punctuation: "(" [36:3 - 36:4] ObjCInstanceMethodDecl=actionMethod::36:1
// CHECK: Identifier: "IBAction" [36:4 - 36:12] macro instantiation=IBAction:33:9
// CHECK: Punctuation: ")" [36:12 - 36:13] ObjCInstanceMethodDecl=actionMethod::36:1
// CHECK: Identifier: "actionMethod" [36:14 - 36:26] ObjCInstanceMethodDecl=actionMethod::36:1
// CHECK: Punctuation: ":" [36:26 - 36:27] ObjCInstanceMethodDecl=actionMethod::36:1
// CHECK: Punctuation: "(" [36:27 - 36:28] ObjCInstanceMethodDecl=actionMethod::36:1
// CHECK: Identifier: "id" [36:28 - 36:30] TypeRef=id:0:0
// CHECK: Punctuation: ")" [36:30 - 36:31] ParmDecl=arg:36:31 (Definition)
// CHECK: Identifier: "arg" [36:31 - 36:34] ParmDecl=arg:36:31 (Definition)
// CHECK: Punctuation: ";" [36:34 - 36:35] ObjCInstanceMethodDecl=actionMethod::36:1
// CHECK: Punctuation: "-" [37:1 - 37:2] ObjCInstanceMethodDecl=foo::37:1
// CHECK: Punctuation: "(" [37:3 - 37:4] ObjCInstanceMethodDecl=foo::37:1
// CHECK: Keyword: "void" [37:4 - 37:8] ObjCInstanceMethodDecl=foo::37:1
// CHECK: Punctuation: ")" [37:8 - 37:9] ObjCInstanceMethodDecl=foo::37:1
// CHECK: Identifier: "foo" [37:9 - 37:12] ObjCInstanceMethodDecl=foo::37:1
// CHECK: Punctuation: ":" [37:12 - 37:13] ObjCInstanceMethodDecl=foo::37:1
// CHECK: Punctuation: "(" [37:13 - 37:14] ObjCInstanceMethodDecl=foo::37:1
// CHECK: Keyword: "int" [37:14 - 37:17] ParmDecl=x:37:18 (Definition)
// CHECK: Punctuation: ")" [37:17 - 37:18] ParmDecl=x:37:18 (Definition)
// CHECK: Identifier: "x" [37:18 - 37:19] ParmDecl=x:37:18 (Definition)
// CHECK: Punctuation: ";" [37:19 - 37:20] ObjCInstanceMethodDecl=foo::37:1
// CHECK: Punctuation: "@" [38:1 - 38:2] ObjCInterfaceDecl=IBActionTests:35:12
// CHECK: Keyword: "end" [38:2 - 38:5] ObjCInterfaceDecl=IBActionTests:35:12
// CHECK: Keyword: "extern" [39:1 - 39:7]
// CHECK: Keyword: "int" [39:8 - 39:11] FunctionDecl=ibaction_test:39:12
// CHECK: Identifier: "ibaction_test" [39:12 - 39:25] FunctionDecl=ibaction_test:39:12
// CHECK: Punctuation: "(" [39:25 - 39:26] FunctionDecl=ibaction_test:39:12
// CHECK: Keyword: "void" [39:26 - 39:30] FunctionDecl=ibaction_test:39:12
// CHECK: Punctuation: ")" [39:30 - 39:31] FunctionDecl=ibaction_test:39:12
// CHECK: Punctuation: ";" [39:31 - 39:32]
// CHECK: Punctuation: "@" [40:1 - 40:2] ObjCImplementationDecl=IBActionTests:40:1 (Definition)
// CHECK: Keyword: "implementation" [40:2 - 40:16] ObjCImplementationDecl=IBActionTests:40:1 (Definition)
// CHECK: Identifier: "IBActionTests" [40:17 - 40:30] ObjCImplementationDecl=IBActionTests:40:1 (Definition)
// CHECK: Punctuation: "-" [41:1 - 41:2] ObjCInstanceMethodDecl=actionMethod::41:1 (Definition)
// CHECK: Punctuation: "(" [41:3 - 41:4] ObjCInstanceMethodDecl=actionMethod::41:1 (Definition)
// CHECK: Identifier: "IBAction" [41:4 - 41:12] macro instantiation=IBAction:33:9
// CHECK: Punctuation: ")" [41:12 - 41:13] ObjCInstanceMethodDecl=actionMethod::41:1 (Definition)
// CHECK: Identifier: "actionMethod" [41:14 - 41:26] ObjCInstanceMethodDecl=actionMethod::41:1 (Definition)
// CHECK: Punctuation: ":" [41:26 - 41:27] ObjCInstanceMethodDecl=actionMethod::41:1 (Definition)
// CHECK: Punctuation: "(" [41:27 - 41:28] ObjCInstanceMethodDecl=actionMethod::41:1 (Definition)
// CHECK: Identifier: "id" [41:28 - 41:30] TypeRef=id:0:0
// CHECK: Punctuation: ")" [41:30 - 41:31] ParmDecl=arg:41:31 (Definition)
// CHECK: Identifier: "arg" [41:31 - 41:34] ParmDecl=arg:41:31 (Definition)
// CHECK: Punctuation: "{" [42:1 - 42:2] UnexposedStmt=
// CHECK: Identifier: "ibaction_test" [43:5 - 43:18] DeclRefExpr=ibaction_test:39:12
// CHECK: Punctuation: "(" [43:18 - 43:19] CallExpr=ibaction_test:39:12
// CHECK: Punctuation: ")" [43:19 - 43:20] CallExpr=ibaction_test:39:12
// CHECK: Punctuation: ";" [43:20 - 43:21] UnexposedStmt=
// CHECK: Punctuation: "[" [44:5 - 44:6] ObjCMessageExpr=foo::37:1
// CHECK: Identifier: "self" [44:6 - 44:10] DeclRefExpr=self:0:0
// CHECK: Identifier: "foo" [44:11 - 44:14] ObjCMessageExpr=foo::37:1
// CHECK: Punctuation: ":" [44:14 - 44:15] ObjCMessageExpr=foo::37:1
// CHECK: Literal: "0" [44:15 - 44:16] UnexposedExpr=
// CHECK: Punctuation: "]" [44:16 - 44:17] ObjCMessageExpr=foo::37:1
// CHECK: Punctuation: ";" [44:17 - 44:18] UnexposedStmt=
// CHECK: Punctuation: "}" [45:1 - 45:2] UnexposedStmt=
// CHECK: Punctuation: "-" [46:1 - 46:2] ObjCInstanceMethodDecl=foo::46:1 (Definition)
// CHECK: Punctuation: "(" [46:3 - 46:4] ObjCInstanceMethodDecl=foo::46:1 (Definition)
// CHECK: Keyword: "void" [46:4 - 46:8] ObjCInstanceMethodDecl=foo::46:1 (Definition)
// CHECK: Punctuation: ")" [46:8 - 46:9] ObjCInstanceMethodDecl=foo::46:1 (Definition)
// CHECK: Identifier: "foo" [46:10 - 46:13] ObjCInstanceMethodDecl=foo::46:1 (Definition)
// CHECK: Punctuation: ":" [46:13 - 46:14] ObjCInstanceMethodDecl=foo::46:1 (Definition)
// CHECK: Punctuation: "(" [46:14 - 46:15] ObjCInstanceMethodDecl=foo::46:1 (Definition)
// CHECK: Keyword: "int" [46:15 - 46:18] ParmDecl=x:46:19 (Definition)
// CHECK: Punctuation: ")" [46:18 - 46:19] ParmDecl=x:46:19 (Definition)
// CHECK: Identifier: "x" [46:19 - 46:20] ParmDecl=x:46:19 (Definition)
// CHECK: Punctuation: "{" [47:1 - 47:2] UnexposedStmt=
// CHECK: Punctuation: "(" [48:3 - 48:4] UnexposedExpr=x:46:19
// CHECK: Keyword: "void" [48:4 - 48:8] UnexposedExpr=x:46:19
// CHECK: Punctuation: ")" [48:8 - 48:9] UnexposedExpr=x:46:19
// CHECK: Identifier: "x" [48:10 - 48:11] DeclRefExpr=x:46:19
// CHECK: Punctuation: ";" [48:11 - 48:12] UnexposedStmt=
// CHECK: Punctuation: "}" [49:1 - 49:2] UnexposedStmt=
// CHECK: Punctuation: "@" [50:1 - 50:2] ObjCImplementationDecl=IBActionTests:40:1 (Definition)
// CHECK: Keyword: "end" [50:2 - 50:5]
// CHECK: Punctuation: "@" [54:1 - 54:2] ObjCInterfaceDecl=IBOutletTests:54:12
// CHECK: Keyword: "interface" [54:2 - 54:11] ObjCInterfaceDecl=IBOutletTests:54:12
// CHECK: Identifier: "IBOutletTests" [54:12 - 54:25] ObjCInterfaceDecl=IBOutletTests:54:12
// CHECK: Punctuation: "{" [55:1 - 55:2] ObjCInterfaceDecl=IBOutletTests:54:12
// CHECK: Identifier: "IBOutlet" [56:5 - 56:13] macro instantiation=IBOutlet:32:9
// CHECK: Keyword: "char" [56:14 - 56:18] ObjCIvarDecl=anOutlet:56:21 (Definition)
// CHECK: Punctuation: "*" [56:19 - 56:20] ObjCIvarDecl=anOutlet:56:21 (Definition)
// CHECK: Identifier: "anOutlet" [56:21 - 56:29] ObjCIvarDecl=anOutlet:56:21 (Definition)
// CHECK: Punctuation: ";" [56:29 - 56:30] ObjCInterfaceDecl=IBOutletTests:54:12
// CHECK: Punctuation: "}" [57:1 - 57:2] ObjCInterfaceDecl=IBOutletTests:54:12
// CHECK: Punctuation: "-" [58:1 - 58:2] ObjCInterfaceDecl=IBOutletTests:54:12

