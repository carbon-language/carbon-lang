// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fblocks -emit-pch -x objective-c %s -o %t.ast
// RUN: c-index-test -test-load-tu %t.ast all | FileCheck %s

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

// CHECK: c-index-api-loadTU-test.m:4:12: ObjCInterfaceDecl=Foo:4:12 [Extent=4:1:11:4]
// CHECK: c-index-api-loadTU-test.m:8:1: ObjCInstanceMethodDecl=foo:8:1 [Extent=8:1:8:6]
// CHECK: c-index-api-loadTU-test.m:9:1: ObjCClassMethodDecl=fooC:9:1 [Extent=9:1:9:7]
// CHECK: c-index-api-loadTU-test.m:13:12: ObjCInterfaceDecl=Bar:13:12 [Extent=13:1:17:4]
// CHECK: c-index-api-loadTU-test.m:13:18: ObjCSuperClassRef=Foo:4:12 [Extent=13:18:13:20]
// CHECK: c-index-api-loadTU-test.m:19:12: ObjCCategoryDecl=FooCat:19:12 [Extent=19:1:22:4]
// CHECK: c-index-api-loadTU-test.m:19:12: ObjCClassRef=Foo:4:12 [Extent=19:12:19:14]
// CHECK: c-index-api-loadTU-test.m:20:1: ObjCInstanceMethodDecl=catMethodWithFloat::20:1 [Extent=20:1:20:40]
// CHECK: c-index-api-loadTU-test.m:21:1: ObjCInstanceMethodDecl=floatMethod:21:1 [Extent=21:1:21:22]
// CHECK: c-index-api-loadTU-test.m:24:1: ObjCProtocolDecl=Proto:24:1 (Definition) [Extent=24:1:26:4]
// CHECK: c-index-api-loadTU-test.m:25:1: ObjCInstanceMethodDecl=pMethod:25:1 [Extent=25:1:25:10]
// CHECK: c-index-api-loadTU-test.m:28:1: ObjCProtocolDecl=SubP:28:1 (Definition) [Extent=28:1:30:4]
// CHECK: c-index-api-loadTU-test.m:28:17: ObjCProtocolRef=Proto:24:1 [Extent=28:17:28:21]
// CHECK: c-index-api-loadTU-test.m:29:1: ObjCInstanceMethodDecl=spMethod:29:1 [Extent=29:1:29:11]
// CHECK: c-index-api-loadTU-test.m:32:12: ObjCInterfaceDecl=Baz:32:12 [Extent=32:1:39:4]
// CHECK: c-index-api-loadTU-test.m:32:18: ObjCSuperClassRef=Bar:13:12 [Extent=32:18:32:20]
// CHECK: c-index-api-loadTU-test.m:32:23: ObjCProtocolRef=SubP:28:1 [Extent=32:23:32:26]
// CHECK: c-index-api-loadTU-test.m:34:9: ObjCIvarDecl=_anIVar:34:9 (Definition) [Extent=34:9:34:15]
// CHECK: c-index-api-loadTU-test.m:37:1: ObjCInstanceMethodDecl=bazMethod:37:1 [Extent=37:1:37:20]
// CHECK: c-index-api-loadTU-test.m:41:1: EnumDecl=:41:1 (Definition) [Extent=41:1:43:1]
// CHECK: c-index-api-loadTU-test.m:42:3: EnumConstantDecl=someEnum:42:3 (Definition) [Extent=42:3:42:10]
// CHECK: c-index-api-loadTU-test.m:45:5: FunctionDecl=main:45:5 (Definition) [Extent=45:5:54:1]
// CHECK: c-index-api-loadTU-test.m:45:15: ParmDecl=argc:45:15 (Definition) [Extent=45:15:45:18]
// CHECK: c-index-api-loadTU-test.m:45:34: ParmDecl=argv:45:34 (Definition) [Extent=45:34:45:37]
// CHECK: c-index-api-loadTU-test.m:46:8: VarDecl=bee:46:8 (Definition) [Extent=46:8:46:10]
// CHECK: c-index-api-loadTU-test.m:46:2: ObjCClassRef=Baz:32:12 [Extent=46:2:46:4]
// CHECK: c-index-api-loadTU-test.m:47:5: VarDecl=a:47:5 (Definition) [Extent=47:5:47:17]
// CHECK: c-index-api-loadTU-test.m:47:2: TypeRef=id:0:0 [Extent=47:2:47:3]
// CHECK: c-index-api-loadTU-test.m:48:12: VarDecl=c:48:12 (Definition) [Extent=48:12:48:25]
// CHECK: c-index-api-loadTU-test.m:48:2: TypeRef=id:0:0 [Extent=48:2:48:3]
// CHECK: c-index-api-loadTU-test.m:48:6: ObjCProtocolRef=SubP:28:1 [Extent=48:6:48:9]
// CHECK: c-index-api-loadTU-test.m:49:13: VarDecl=d:49:13 (Definition) [Extent=49:13:49:13]
// CHECK: c-index-api-loadTU-test.m:49:2: TypeRef=id:0:0 [Extent=49:2:49:3]
// CHECK: c-index-api-loadTU-test.m:49:6: ObjCProtocolRef=Proto:24:1 [Extent=49:6:49:10]
// CHECK: c-index-api-loadTU-test.m:50:2: DeclRefExpr=d:49:13 [Extent=50:2:50:2]
// CHECK: c-index-api-loadTU-test.m:50:6: DeclRefExpr=c:48:12 [Extent=50:6:50:6]
// CHECK: c-index-api-loadTU-test.m:51:2: ObjCMessageExpr=pMethod:25:1 [Extent=51:2:51:12]
// CHECK: c-index-api-loadTU-test.m:51:3: DeclRefExpr=d:49:13 [Extent=51:3:51:3]
// CHECK: c-index-api-loadTU-test.m:52:2: ObjCMessageExpr=catMethodWithFloat::20:1 [Extent=52:2:52:43]
// CHECK: c-index-api-loadTU-test.m:52:3: DeclRefExpr=bee:46:8 [Extent=52:3:52:5]
// CHECK: c-index-api-loadTU-test.m:52:26: ObjCMessageExpr=floatMethod:21:1 [Extent=52:26:52:42]
// CHECK: c-index-api-loadTU-test.m:52:27: DeclRefExpr=bee:46:8 [Extent=52:27:52:29]
// CHECK: c-index-api-loadTU-test.m:53:3: CallExpr=main:45:5 [Extent=53:3:53:36]
// CHECK: c-index-api-loadTU-test.m:53:3: DeclRefExpr=main:45:5 [Extent=53:3:53:6]
// CHECK: c-index-api-loadTU-test.m:53:8: DeclRefExpr=someEnum:42:3 [Extent=53:8:53:15]
// CHECK: c-index-api-loadTU-test.m:53:33: DeclRefExpr=bee:46:8 [Extent=53:33:53:35]
