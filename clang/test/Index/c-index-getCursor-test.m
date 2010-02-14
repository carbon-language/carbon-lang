// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fblocks -emit-pch -x objective-c %s -o %t.ast
// RUN: c-index-test -test-file-scan %t.ast %s | FileCheck %s
@interface Foo 
{
}

- foo;
+ fooC;

@end

@interface Bar : Foo 
{
}

@end

@interface Foo (FooCat)
- (int) catMethodWithFloat:(float) fArg;
- (float) floatMethod;
@end

@protocol Proto
- pMethod;
@end

@protocol SubP <Proto>
- spMethod;
@end

@interface Baz : Bar <SubP>
{
    int _anIVar;
}

- (Foo *) bazMethod;

@end

enum {
  someEnum
};

int main (int argc, const char * argv[]) {
	Baz * bee;
	id a = [bee foo];
	id <SubP> c = [Foo fooC];
	id <Proto> d;
	d = c;
	[d pMethod];
	[bee catMethodWithFloat:[bee floatMethod]];
  main(someEnum, (const char **)bee);
}

// CHECK: [1:1 - 2:62] Invalid Cursor => NoDeclFound
// CHECK: [3:1 - 6:1] ObjCInterfaceDecl=Foo:3:12
// CHECK: [7:1 - 7:6] ObjCInstanceMethodDecl=foo:7:1
// CHECK: [7:7 - 7:7] ObjCInterfaceDecl=Foo:3:12
// CHECK: [8:1 - 8:7] ObjCClassMethodDecl=fooC:8:1
// CHECK: [8:8 - 10:4] ObjCInterfaceDecl=Foo:3:12
// CHECK: [10:5 - 11:1] Invalid Cursor => NoDeclFound
// CHECK: [12:1 - 12:17] ObjCInterfaceDecl=Bar:12:12
// CHECK: [12:18 - 12:20] ObjCSuperClassRef=Foo:3:12
// CHECK: [12:21 - 16:4] ObjCInterfaceDecl=Bar:12:12
// CHECK: [16:5 - 17:1] Invalid Cursor => NoDeclFound
// CHECK: [18:1 - 18:11] ObjCCategoryDecl=FooCat:18:12
// CHECK: [18:12 - 18:14] ObjCClassRef=Foo:3:12
// CHECK: [18:15 - 18:24] ObjCCategoryDecl=FooCat:18:12
// CHECK: [19:1 - 19:28] ObjCInstanceMethodDecl=catMethodWithFloat::19:1
// CHECK: [19:29 - 19:39] ParmDecl=fArg:19:36 (Definition)
// CHECK: [19:40 - 19:40] ObjCInstanceMethodDecl=catMethodWithFloat::19:1
// CHECK: [19:41 - 19:41] ObjCCategoryDecl=FooCat:18:12
// CHECK: [20:1 - 20:22] ObjCInstanceMethodDecl=floatMethod:20:1
// CHECK: [20:23 - 21:4] ObjCCategoryDecl=FooCat:18:12
// CHECK: [21:5 - 22:1] Invalid Cursor => NoDeclFound
// CHECK: [23:1 - 23:16] ObjCProtocolDecl=Proto:23:1 (Definition)
// CHECK: [24:1 - 24:10] ObjCInstanceMethodDecl=pMethod:24:1
// CHECK: [24:11 - 25:4] ObjCProtocolDecl=Proto:23:1 (Definition)
// CHECK: [25:5 - 26:1] Invalid Cursor => NoDeclFound
// CHECK: [27:1 - 27:16] ObjCProtocolDecl=SubP:27:1 (Definition)
// CHECK: [27:17 - 27:21] ObjCProtocolRef=Proto:23:1
// CHECK: [27:22 - 27:23] ObjCProtocolDecl=SubP:27:1 (Definition)
// CHECK: [28:1 - 28:11] ObjCInstanceMethodDecl=spMethod:28:1
// CHECK: [28:12 - 29:4] ObjCProtocolDecl=SubP:27:1 (Definition)
// CHECK: [29:5 - 30:1] Invalid Cursor => NoDeclFound
// CHECK: [31:1 - 31:17] ObjCInterfaceDecl=Baz:31:12
// CHECK: [31:18 - 31:20] ObjCSuperClassRef=Bar:12:12
// CHECK: [31:21 - 31:22] ObjCInterfaceDecl=Baz:31:12
// CHECK: [31:23 - 31:26] ObjCProtocolRef=SubP:27:1
// CHECK: [31:27 - 33:8] ObjCInterfaceDecl=Baz:31:12
// CHECK: [33:9 - 33:15] ObjCIvarDecl=_anIVar:33:9 (Definition)
// CHECK: [33:16 - 35:1] ObjCInterfaceDecl=Baz:31:12
// CHECK: [36:1 - 36:20] ObjCInstanceMethodDecl=bazMethod:36:1
// CHECK: [36:21 - 38:4] ObjCInterfaceDecl=Baz:31:12
// CHECK: [38:5 - 39:1] Invalid Cursor => NoDeclFound
// CHECK: [40:1 - 41:2] EnumDecl=:40:1 (Definition)
// CHECK: [41:3 - 41:10] EnumConstantDecl=someEnum:41:3 (Definition)
// CHECK: [41:11 - 42:1] EnumDecl=:40:1 (Definition)
// CHECK: [42:2 - 44:4] Invalid Cursor => NoDeclFound
// CHECK: [44:5 - 44:10] FunctionDecl=main:44:5 (Definition)
// CHECK: [44:11 - 44:18] ParmDecl=argc:44:15 (Definition)
// CHECK: [44:19 - 44:26] FunctionDecl=main:44:5 (Definition)
// CHECK: [44:27 - 44:37] ParmDecl=argv:44:34 (Definition)
// CHECK: [44:38 - 44:41] FunctionDecl=main:44:5 (Definition)
// CHECK: [44:42 - 45:1] UnexposedStmt=
// CHECK: [45:2 - 45:4] ObjCClassRef=Baz:31:12
// CHECK: [45:5 - 45:10] VarDecl=bee:45:8 (Definition)
// CHECK: [45:11 - 45:11] UnexposedStmt=
// CHECK: [45:12 - 46:1] UnexposedStmt=
// CHECK: [46:2 - 46:3] TypeRef=id:0:0
// CHECK: [46:4 - 46:8] VarDecl=a:46:5 (Definition)
// CHECK: [46:9 - 46:9] ObjCMessageExpr=foo:7:1
// CHECK: [46:10 - 46:12] DeclRefExpr=bee:45:8
// CHECK: [46:13 - 46:17] ObjCMessageExpr=foo:7:1
// CHECK: [46:18 - 46:18] UnexposedStmt=
// CHECK: [46:19 - 47:1] UnexposedStmt=
// CHECK: [47:2 - 47:3] TypeRef=id:0:0
// CHECK: [47:4 - 47:5] VarDecl=c:47:12 (Definition)
// CHECK: [47:6 - 47:9] ObjCProtocolRef=SubP:27:1
// CHECK: [47:10 - 47:15] VarDecl=c:47:12 (Definition)
// CHECK: [47:16 - 47:25] ObjCMessageExpr=fooC:8:1
// CHECK: [47:26 - 47:26] UnexposedStmt=
// CHECK: [47:27 - 48:1] UnexposedStmt=
// CHECK: [48:2 - 48:3] TypeRef=id:0:0
// CHECK: [48:4 - 48:5] VarDecl=d:48:13 (Definition)
// CHECK: [48:6 - 48:10] ObjCProtocolRef=Proto:23:1
// CHECK: [48:11 - 48:13] VarDecl=d:48:13 (Definition)
// CHECK: [48:14 - 48:14] UnexposedStmt=
// CHECK: [48:15 - 49:1] UnexposedStmt=
// CHECK: [49:2 - 49:2] DeclRefExpr=d:48:13
// CHECK: [49:3 - 49:5] UnexposedExpr=
// CHECK: [49:6 - 49:6] DeclRefExpr=c:47:12
// CHECK: [49:7 - 50:1] UnexposedStmt=
// CHECK: [50:2 - 50:2] ObjCMessageExpr=pMethod:24:1
// CHECK: [50:3 - 50:3] DeclRefExpr=d:48:13
// CHECK: [50:4 - 50:12] ObjCMessageExpr=pMethod:24:1
// CHECK: [50:13 - 51:1] UnexposedStmt=
// CHECK: [51:2 - 51:2] ObjCMessageExpr=catMethodWithFloat::19:1
// CHECK: [51:3 - 51:5] DeclRefExpr=bee:45:8
// CHECK: [51:6 - 51:25] ObjCMessageExpr=catMethodWithFloat::19:1
// CHECK: [51:26 - 51:26] ObjCMessageExpr=floatMethod:20:1
// CHECK: [51:27 - 51:29] DeclRefExpr=bee:45:8
// CHECK: [51:30 - 51:42] ObjCMessageExpr=floatMethod:20:1
// CHECK: [51:43 - 51:43] ObjCMessageExpr=catMethodWithFloat::19:1
// CHECK: [51:44 - 52:2] UnexposedStmt=
// CHECK: [52:3 - 52:6] DeclRefExpr=main:44:5
// CHECK: [52:7 - 52:7] CallExpr=main:44:5
// CHECK: [52:8 - 52:15] DeclRefExpr=someEnum:41:3
// CHECK: [52:16 - 52:17] CallExpr=main:44:5
// CHECK: [52:18 - 52:32] UnexposedExpr=bee:45:8
// CHECK: [52:33 - 52:35] DeclRefExpr=bee:45:8
// CHECK: [52:36 - 52:36] CallExpr=main:44:5
// CHECK: [52:37 - 53:1] UnexposedStmt=
