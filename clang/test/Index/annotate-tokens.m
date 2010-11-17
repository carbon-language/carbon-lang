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

// From <rdar://problem/7974151>.  The first 'foo:' wasn't being annotated as 
// being part of the Objective-C message expression since the argument
// was expanded from a macro.

#define VAL 0

@interface R7974151
- (int) foo:(int)arg;
- (int) method;
@end

@implementation R7974151
- (int) foo:(int)arg {
  return arg;
}
- (int) method
{
    int local = [self foo:VAL];
    int second = [self foo:0];
    return local;
}
- (int)othermethod:(IBOutletTests *)ibt {
  return *ibt.aPropOutlet;
}
@end

@protocol Proto @end

void f() {
  (void)@protocol(Proto);
}

// <rdar://problem/8595462> - Properly annotate functions and variables
// declared within an @implementation.
@class Rdar8595462_A;
@interface Rdar8595462_B
@end

@implementation Rdar8595462_B
Rdar8595462_A * Rdar8595462_aFunction() {
  Rdar8595462_A * localVar = 0;
  return localVar;
}
static Rdar8595462_A * Rdar8595462_staticVar;
@end

// <rdar://problem/8595386> Issues doing syntax coloring of properties
@interface Rdar8595386 {
  Foo *_foo;
}

@property (readonly, copy) Foo *foo;
@property (readonly) Foo *foo2;
@end

@implementation Rdar8595386
@synthesize foo = _foo;
@dynamic foo2;
@end

// RUN: c-index-test -test-annotate-tokens=%s:1:1:118:1 %s -DIBOutlet='__attribute__((iboutlet))' -DIBAction='void)__attribute__((ibaction)' | FileCheck %s
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
// CHECK: Identifier: "IBOutlet" [53:5 - 53:13] macro instantiation=IBOutlet
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
// CHECK: Punctuation: "@" [56:1 - 56:2] ObjCPropertyDecl=aPropOutlet:56:26
// CHECK: Keyword: "property" [56:2 - 56:10] ObjCPropertyDecl=aPropOutlet:56:26
// CHECK: Identifier: "IBOutlet" [56:11 - 56:19] macro instantiation=IBOutlet
// CHECK: Keyword: "int" [56:20 - 56:23] ObjCPropertyDecl=aPropOutlet:56:26
// CHECK: Punctuation: "*" [56:24 - 56:25] ObjCPropertyDecl=aPropOutlet:56:26
// CHECK: Identifier: "aPropOutlet" [56:26 - 56:37] ObjCPropertyDecl=aPropOutlet:56:26
// CHECK: Punctuation: ";" [56:37 - 56:38] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "@" [57:1 - 57:2] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Keyword: "end" [57:2 - 57:5] ObjCInterfaceDecl=IBOutletTests:51:12
// CHECK: Punctuation: "#" [63:1 - 63:2] preprocessing directive=
// CHECK: Identifier: "define" [63:2 - 63:8] preprocessing directive=
// CHECK: Identifier: "VAL" [63:9 - 63:12] macro definition=VAL
// CHECK: Literal: "0" [63:13 - 63:14] macro definition=VAL
// CHECK: Punctuation: "@" [65:1 - 65:2] ObjCInterfaceDecl=R7974151:65:12
// CHECK: Keyword: "interface" [65:2 - 65:11] ObjCInterfaceDecl=R7974151:65:12
// CHECK: Identifier: "R7974151" [65:12 - 65:20] ObjCInterfaceDecl=R7974151:65:12
// CHECK: Punctuation: "-" [66:1 - 66:2] ObjCInstanceMethodDecl=foo::66:1
// CHECK: Punctuation: "(" [66:3 - 66:4] ObjCInstanceMethodDecl=foo::66:1
// CHECK: Keyword: "int" [66:4 - 66:7] ObjCInstanceMethodDecl=foo::66:1
// CHECK: Punctuation: ")" [66:7 - 66:8] ObjCInstanceMethodDecl=foo::66:1
// CHECK: Identifier: "foo" [66:9 - 66:12] ObjCInstanceMethodDecl=foo::66:1
// CHECK: Punctuation: ":" [66:12 - 66:13] ObjCInstanceMethodDecl=foo::66:1
// CHECK: Punctuation: "(" [66:13 - 66:14] ObjCInstanceMethodDecl=foo::66:1
// CHECK: Keyword: "int" [66:14 - 66:17] ParmDecl=arg:66:18 (Definition)
// CHECK: Punctuation: ")" [66:17 - 66:18] ParmDecl=arg:66:18 (Definition)
// CHECK: Identifier: "arg" [66:18 - 66:21] ParmDecl=arg:66:18 (Definition)
// CHECK: Punctuation: ";" [66:21 - 66:22] ObjCInstanceMethodDecl=foo::66:1
// CHECK: Punctuation: "-" [67:1 - 67:2] ObjCInstanceMethodDecl=method:67:1
// CHECK: Punctuation: "(" [67:3 - 67:4] ObjCInstanceMethodDecl=method:67:1
// CHECK: Keyword: "int" [67:4 - 67:7] ObjCInstanceMethodDecl=method:67:1
// CHECK: Punctuation: ")" [67:7 - 67:8] ObjCInstanceMethodDecl=method:67:1
// CHECK: Identifier: "method" [67:9 - 67:15] ObjCInstanceMethodDecl=method:67:1
// CHECK: Punctuation: ";" [67:15 - 67:16] ObjCInstanceMethodDecl=method:67:1
// CHECK: Punctuation: "@" [68:1 - 68:2] ObjCInterfaceDecl=R7974151:65:12
// CHECK: Keyword: "end" [68:2 - 68:5] ObjCInterfaceDecl=R7974151:65:12
// CHECK: Punctuation: "@" [70:1 - 70:2] ObjCImplementationDecl=R7974151:70:1 (Definition)
// CHECK: Keyword: "implementation" [70:2 - 70:16] ObjCImplementationDecl=R7974151:70:1 (Definition)
// CHECK: Identifier: "R7974151" [70:17 - 70:25] ObjCImplementationDecl=R7974151:70:1 (Definition)
// CHECK: Punctuation: "-" [71:1 - 71:2] ObjCInstanceMethodDecl=foo::71:1 (Definition)
// CHECK: Punctuation: "(" [71:3 - 71:4] ObjCInstanceMethodDecl=foo::71:1 (Definition)
// CHECK: Keyword: "int" [71:4 - 71:7] ObjCInstanceMethodDecl=foo::71:1 (Definition)
// CHECK: Punctuation: ")" [71:7 - 71:8] ObjCInstanceMethodDecl=foo::71:1 (Definition)
// CHECK: Identifier: "foo" [71:9 - 71:12] ObjCInstanceMethodDecl=foo::71:1 (Definition)
// CHECK: Punctuation: ":" [71:12 - 71:13] ObjCInstanceMethodDecl=foo::71:1 (Definition)
// CHECK: Punctuation: "(" [71:13 - 71:14] ObjCInstanceMethodDecl=foo::71:1 (Definition)
// CHECK: Keyword: "int" [71:14 - 71:17] ParmDecl=arg:71:18 (Definition)
// CHECK: Punctuation: ")" [71:17 - 71:18] ParmDecl=arg:71:18 (Definition)
// CHECK: Identifier: "arg" [71:18 - 71:21] ParmDecl=arg:71:18 (Definition)
// CHECK: Punctuation: "{" [71:22 - 71:23] UnexposedStmt=
// CHECK: Keyword: "return" [72:3 - 72:9] UnexposedStmt=
// CHECK: Identifier: "arg" [72:10 - 72:13] DeclRefExpr=arg:71:18
// CHECK: Punctuation: ";" [72:13 - 72:14] UnexposedStmt=
// CHECK: Punctuation: "}" [73:1 - 73:2] UnexposedStmt=
// CHECK: Punctuation: "-" [74:1 - 74:2] ObjCInstanceMethodDecl=method:74:1 (Definition)
// CHECK: Punctuation: "(" [74:3 - 74:4] ObjCInstanceMethodDecl=method:74:1 (Definition)
// CHECK: Keyword: "int" [74:4 - 74:7] ObjCInstanceMethodDecl=method:74:1 (Definition)
// CHECK: Punctuation: ")" [74:7 - 74:8] ObjCInstanceMethodDecl=method:74:1 (Definition)
// CHECK: Identifier: "method" [74:9 - 74:15] ObjCInstanceMethodDecl=method:74:1 (Definition)
// CHECK: Punctuation: "{" [75:1 - 75:2] UnexposedStmt=
// CHECK: Keyword: "int" [76:5 - 76:8] VarDecl=local:76:9 (Definition)
// CHECK: Identifier: "local" [76:9 - 76:14] VarDecl=local:76:9 (Definition)
// CHECK: Punctuation: "=" [76:15 - 76:16] VarDecl=local:76:9 (Definition)
// CHECK: Punctuation: "[" [76:17 - 76:18] ObjCMessageExpr=foo::66:1
// CHECK: Identifier: "self" [76:18 - 76:22] DeclRefExpr=self:0:0
// CHECK: Identifier: "foo" [76:23 - 76:26] ObjCMessageExpr=foo::66:1
// CHECK: Punctuation: ":" [76:26 - 76:27] ObjCMessageExpr=foo::66:1
// CHECK: Identifier: "VAL" [76:27 - 76:30] macro instantiation=VAL:63:9
// CHECK: Punctuation: "]" [76:30 - 76:31] ObjCMessageExpr=foo::66:1
// CHECK: Punctuation: ";" [76:31 - 76:32] UnexposedStmt=
// CHECK: Keyword: "int" [77:5 - 77:8] VarDecl=second:77:9 (Definition)
// CHECK: Identifier: "second" [77:9 - 77:15] VarDecl=second:77:9 (Definition)
// CHECK: Punctuation: "=" [77:16 - 77:17] VarDecl=second:77:9 (Definition)
// CHECK: Punctuation: "[" [77:18 - 77:19] ObjCMessageExpr=foo::66:1
// CHECK: Identifier: "self" [77:19 - 77:23] DeclRefExpr=self:0:0
// CHECK: Identifier: "foo" [77:24 - 77:27] ObjCMessageExpr=foo::66:1
// CHECK: Punctuation: ":" [77:27 - 77:28] ObjCMessageExpr=foo::66:1
// CHECK: Literal: "0" [77:28 - 77:29] UnexposedExpr=
// CHECK: Punctuation: "]" [77:29 - 77:30] ObjCMessageExpr=foo::66:1
// CHECK: Punctuation: ";" [77:30 - 77:31] UnexposedStmt=
// CHECK: Keyword: "return" [78:5 - 78:11] UnexposedStmt=
// CHECK: Identifier: "local" [78:12 - 78:17] DeclRefExpr=local:76:9
// CHECK: Punctuation: ";" [78:17 - 78:18] UnexposedStmt=
// CHECK: Punctuation: "}" [79:1 - 79:2] UnexposedStmt=
// CHECK: Punctuation: "-" [80:1 - 80:2] ObjCInstanceMethodDecl=othermethod::80:1 (Definition)
// CHECK: Punctuation: "(" [80:3 - 80:4] ObjCInstanceMethodDecl=othermethod::80:1 (Definition)
// CHECK: Keyword: "int" [80:4 - 80:7] ObjCInstanceMethodDecl=othermethod::80:1 (Definition)
// CHECK: Punctuation: ")" [80:7 - 80:8] ObjCInstanceMethodDecl=othermethod::80:1 (Definition)
// CHECK: Identifier: "othermethod" [80:8 - 80:19] ObjCInstanceMethodDecl=othermethod::80:1 (Definition)
// CHECK: Punctuation: ":" [80:19 - 80:20] ObjCInstanceMethodDecl=othermethod::80:1 (Definition)
// CHECK: Punctuation: "(" [80:20 - 80:21] ObjCInstanceMethodDecl=othermethod::80:1 (Definition)
// CHECK: Identifier: "IBOutletTests" [80:21 - 80:34] ObjCClassRef=IBOutletTests:51:12
// CHECK: Punctuation: "*" [80:35 - 80:36] ParmDecl=ibt:80:37 (Definition)
// CHECK: Punctuation: ")" [80:36 - 80:37] ParmDecl=ibt:80:37 (Definition)
// CHECK: Identifier: "ibt" [80:37 - 80:40] ParmDecl=ibt:80:37 (Definition)
// CHECK: Punctuation: "{" [80:41 - 80:42] UnexposedStmt=
// CHECK: Keyword: "return" [81:3 - 81:9] UnexposedStmt=
// CHECK: Punctuation: "*" [81:10 - 81:11] UnexposedExpr=
// CHECK: Identifier: "ibt" [81:11 - 81:14] DeclRefExpr=ibt:80:37
// CHECK: Punctuation: "." [81:14 - 81:15] MemberRefExpr=aPropOutlet:56:26
// CHECK: Identifier: "aPropOutlet" [81:15 - 81:26] MemberRefExpr=aPropOutlet:56:26
// CHECK: Punctuation: ";" [81:26 - 81:27] UnexposedStmt=
// CHECK: Punctuation: "}" [82:1 - 82:2] UnexposedStmt=
// CHECK: Punctuation: "@" [83:1 - 83:2] ObjCImplementationDecl=R7974151:70:1 (Definition)
// CHECK: Keyword: "end" [83:2 - 83:5]
// CHECK: Punctuation: "@" [85:1 - 85:2] ObjCProtocolDecl=Proto:85:1 (Definition)
// CHECK: Keyword: "protocol" [85:2 - 85:10] ObjCProtocolDecl=Proto:85:1 (Definition)
// CHECK: Identifier: "Proto" [85:11 - 85:16] ObjCProtocolDecl=Proto:85:1 (Definition)
// CHECK: Punctuation: "@" [85:17 - 85:18] ObjCProtocolDecl=Proto:85:1 (Definition)
// CHECK: Keyword: "end" [85:18 - 85:21] ObjCProtocolDecl=Proto:85:1 (Definition)
// CHECK: Keyword: "void" [87:1 - 87:5] FunctionDecl=f:87:6 (Definition)
// CHECK: Identifier: "f" [87:6 - 87:7] FunctionDecl=f:87:6 (Definition)
// CHECK: Punctuation: "(" [87:7 - 87:8] FunctionDecl=f:87:6 (Definition)
// CHECK: Punctuation: ")" [87:8 - 87:9] FunctionDecl=f:87:6 (Definition)
// CHECK: Punctuation: "{" [87:10 - 87:11] UnexposedStmt=
// CHECK: Punctuation: "(" [88:3 - 88:4] UnexposedExpr=Proto:85:1
// CHECK: Keyword: "void" [88:4 - 88:8] UnexposedExpr=Proto:85:1
// CHECK: Punctuation: ")" [88:8 - 88:9] UnexposedExpr=Proto:85:1
// CHECK: Punctuation: "@" [88:9 - 88:10] UnexposedExpr=Proto:85:1
// CHECK: Keyword: "protocol" [88:10 - 88:18] UnexposedExpr=Proto:85:1
// CHECK: Punctuation: "(" [88:18 - 88:19] UnexposedExpr=Proto:85:1
// CHECK: Identifier: "Proto" [88:19 - 88:24] UnexposedExpr=Proto:85:1
// CHECK: Punctuation: ")" [88:24 - 88:25] UnexposedExpr=Proto:85:1
// CHECK: Punctuation: ";" [88:25 - 88:26] UnexposedStmt=
// CHECK: Punctuation: "}" [89:1 - 89:2] UnexposedStmt=
// CHECK: Punctuation: "@" [93:1 - 93:2] UnexposedDecl=[93:8]
// CHECK: Keyword: "class" [93:2 - 93:7] UnexposedDecl=[93:8]
// CHECK: Identifier: "Rdar8595462_A" [93:8 - 93:21] ObjCClassRef=Rdar8595462_A:93:8
// CHECK: Punctuation: ";" [93:21 - 93:22]
// CHECK: Punctuation: "@" [94:1 - 94:2] ObjCInterfaceDecl=Rdar8595462_B:94:12
// CHECK: Keyword: "interface" [94:2 - 94:11] ObjCInterfaceDecl=Rdar8595462_B:94:12
// CHECK: Identifier: "Rdar8595462_B" [94:12 - 94:25] ObjCInterfaceDecl=Rdar8595462_B:94:12
// CHECK: Punctuation: "@" [95:1 - 95:2] ObjCInterfaceDecl=Rdar8595462_B:94:12
// CHECK: Keyword: "end" [95:2 - 95:5] ObjCInterfaceDecl=Rdar8595462_B:94:12
// CHECK: Punctuation: "@" [97:1 - 97:2] ObjCImplementationDecl=Rdar8595462_B:97:1 (Definition)
// CHECK: Keyword: "implementation" [97:2 - 97:16] ObjCImplementationDecl=Rdar8595462_B:97:1 (Definition)
// CHECK: Identifier: "Rdar8595462_B" [97:17 - 97:30] ObjCImplementationDecl=Rdar8595462_B:97:1 (Definition)
// CHECK: Identifier: "Rdar8595462_A" [98:1 - 98:14] ObjCClassRef=Rdar8595462_A:93:8
// CHECK: Punctuation: "*" [98:15 - 98:16] FunctionDecl=Rdar8595462_aFunction:98:17 (Definition)
// CHECK: Identifier: "Rdar8595462_aFunction" [98:17 - 98:38] FunctionDecl=Rdar8595462_aFunction:98:17 (Definition)
// CHECK: Punctuation: "(" [98:38 - 98:39] FunctionDecl=Rdar8595462_aFunction:98:17 (Definition)
// CHECK: Punctuation: ")" [98:39 - 98:40] FunctionDecl=Rdar8595462_aFunction:98:17 (Definition)
// CHECK: Punctuation: "{" [98:41 - 98:42] UnexposedStmt=
// CHECK: Identifier: "Rdar8595462_A" [99:3 - 99:16] ObjCClassRef=Rdar8595462_A:93:8
// CHECK: Punctuation: "*" [99:17 - 99:18] VarDecl=localVar:99:19 (Definition)
// CHECK: Identifier: "localVar" [99:19 - 99:27] VarDecl=localVar:99:19 (Definition)
// CHECK: Punctuation: "=" [99:28 - 99:29] VarDecl=localVar:99:19 (Definition)
// CHECK: Literal: "0" [99:30 - 99:31] UnexposedExpr=
// CHECK: Punctuation: ";" [99:31 - 99:32] UnexposedStmt=
// CHECK: Keyword: "return" [100:3 - 100:9] UnexposedStmt=
// CHECK: Identifier: "localVar" [100:10 - 100:18] DeclRefExpr=localVar:99:19
// CHECK: Punctuation: ";" [100:18 - 100:19] UnexposedStmt=
// CHECK: Punctuation: "}" [101:1 - 101:2] UnexposedStmt=
// CHECK: Keyword: "static" [102:1 - 102:7] ObjCImplementationDecl=Rdar8595462_B:97:1 (Definition)
// CHECK: Identifier: "Rdar8595462_A" [102:8 - 102:21] ObjCClassRef=Rdar8595462_A:93:8
// CHECK: Punctuation: "*" [102:22 - 102:23] VarDecl=Rdar8595462_staticVar:102:24
// CHECK: Identifier: "Rdar8595462_staticVar" [102:24 - 102:45] VarDecl=Rdar8595462_staticVar:102:24
// CHECK: Punctuation: ";" [102:45 - 102:46] ObjCImplementationDecl=Rdar8595462_B:97:1 (Definition)
// CHECK: Punctuation: "@" [103:1 - 103:2] ObjCImplementationDecl=Rdar8595462_B:97:1 (Definition)
// CHECK: Keyword: "end" [103:2 - 103:5]

// CHECK: Punctuation: "@" [110:1 - 110:2] ObjCPropertyDecl=foo:110:33
// CHECK: Keyword: "property" [110:2 - 110:10] ObjCPropertyDecl=foo:110:33
// CHECK: Punctuation: "(" [110:11 - 110:12] ObjCPropertyDecl=foo:110:33
// CHECK: Identifier: "readonly" [110:12 - 110:20] ObjCPropertyDecl=foo:110:33
// CHECK: Punctuation: "," [110:20 - 110:21] ObjCPropertyDecl=foo:110:33
// CHECK: Identifier: "copy" [110:22 - 110:26] ObjCPropertyDecl=foo:110:33
// CHECK: Punctuation: ")" [110:26 - 110:27] ObjCPropertyDecl=foo:110:33
// CHECK: Identifier: "Foo" [110:28 - 110:31] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [110:32 - 110:33] ObjCPropertyDecl=foo:110:33
// CHECK: Identifier: "foo" [110:33 - 110:36] ObjCPropertyDecl=foo:110:33
// CHECK: Keyword: "property" [111:2 - 111:10] ObjCPropertyDecl=foo2:111:27
// CHECK: Punctuation: "(" [111:11 - 111:12] ObjCPropertyDecl=foo2:111:27
// CHECK: Identifier: "readonly" [111:12 - 111:20] ObjCPropertyDecl=foo2:111:27
// CHECK: Punctuation: ")" [111:20 - 111:21] ObjCPropertyDecl=foo2:111:27
// CHECK: Identifier: "Foo" [111:22 - 111:25] ObjCClassRef=Foo:1:12
// CHECK: Punctuation: "*" [111:26 - 111:27] ObjCPropertyDecl=foo2:111:27
// CHECK: Identifier: "foo2" [111:27 - 111:31] ObjCPropertyDecl=foo2:111:27

// FIXME: Very poor handling of @synthesized
