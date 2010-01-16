// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fblocks -emit-pch -x objective-c %s -o %t.ast
// RUN: c-index-test -test-load-tu %t.ast all | FileCheck %s
// XFAIL: *
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

// CHECK: c-index-api-loadTU-test.m:4:12: ObjCInterfaceDecl=Foo:4:1 [Context=c-index-api-loadTU-test.m] [Extent=4:1:11:4]
// CHECK: c-index-api-loadTU-test.m:8:1: ObjCInstanceMethodDecl=foo:8:1 [Context=Foo] [Extent=8:1:8:6]
// CHECK: c-index-api-loadTU-test.m:9:1: ObjCClassMethodDecl=fooC:9:1 [Context=Foo] [Extent=9:1:9:7]
// CHECK: c-index-api-loadTU-test.m:13:12: ObjCInterfaceDecl=Bar:13:1 [Context=c-index-api-loadTU-test.m] [Extent=13:1:17:4]
// CHECK: c-index-api-loadTU-test.m:13:18: ObjCSuperClassRef=Foo:4:1 [Context=Bar] [Extent=4:1:11:4]
// CHECK: c-index-api-loadTU-test.m:19:1: ObjCCategoryDecl=FooCat:19:1 [Context=c-index-api-loadTU-test.m] [Extent=19:1:22:4]
// CHECK: c-index-api-loadTU-test.m:4:1: ObjCClassRef=Foo:19:1 [Context=FooCat] [Extent=19:1:22:4]
// CHECK: c-index-api-loadTU-test.m:20:1: ObjCInstanceMethodDecl=catMethodWithFloat::20:1 [Context=FooCat] [Extent=20:1:20:40]
// CHECK: c-index-api-loadTU-test.m:21:1: ObjCInstanceMethodDecl=floatMethod:21:1 [Context=FooCat] [Extent=21:1:21:22]
// CHECK: c-index-api-loadTU-test.m:24:1: ObjCProtocolDecl=Proto:24:1 [Context=c-index-api-loadTU-test.m] [Extent=24:1:26:4]
// CHECK: c-index-api-loadTU-test.m:25:1: ObjCInstanceMethodDecl=pMethod:25:1 [Context=Proto] [Extent=25:1:25:10]
// CHECK: c-index-api-loadTU-test.m:28:1: ObjCProtocolDecl=SubP:28:1 [Context=c-index-api-loadTU-test.m] [Extent=28:1:30:4]
// CHECK: c-index-api-loadTU-test.m:28:17: ObjCProtocolRef=Proto:24:1 [Context=SubP] [Extent=24:1:26:4]
// CHECK: c-index-api-loadTU-test.m:29:1: ObjCInstanceMethodDecl=spMethod:29:1 [Context=SubP] [Extent=29:1:29:11]
// CHECK: c-index-api-loadTU-test.m:32:12: ObjCInterfaceDecl=Baz:32:1 [Context=c-index-api-loadTU-test.m] [Extent=32:1:39:4]
// CHECK: c-index-api-loadTU-test.m:32:18: ObjCSuperClassRef=Bar:13:1 [Context=Baz] [Extent=13:1:17:4]
// CHECK: c-index-api-loadTU-test.m:32:23: ObjCProtocolRef=SubP:28:1 [Context=Baz] [Extent=28:1:30:4]
// CHECK: c-index-api-loadTU-test.m:34:9: ObjCIvarDecl=_anIVar:34:9 [Context=Baz] [Extent=34:9:34:15]
// CHECK: c-index-api-loadTU-test.m:37:1: ObjCInstanceMethodDecl=bazMethod:37:1 [Context=Baz] [Extent=37:1:37:20]
// CHECK: c-index-api-loadTU-test.m:41:1: EnumDecl=:41:1 [Context=c-index-api-loadTU-test.m] [Extent=41:1:43:1]
// CHECK: c-index-api-loadTU-test.m:42:3: EnumConstantDecl=someEnum:42:3 [Context=] [Extent=42:3:42:10]
// CHECK: c-index-api-loadTU-test.m:45:5: FunctionDefn=main:45:5 [Context=c-index-api-loadTU-test.m] [Extent=45:5:54:1]
// CHECK: c-index-api-loadTU-test.m:45:15: ParmDecl=argc:45:15 [Context=main] [Extent=45:15:45:18]
// CHECK: c-index-api-loadTU-test.m:45:34: ParmDecl=argv:45:34 [Context=main] [Extent=45:34:45:37]
// CHECK: c-index-api-loadTU-test.m:46:8: VarDecl=bee:46:8 [Context=main] [Extent=46:8:46:10]
// CHECK: c-index-api-loadTU-test.m:47:5: VarDecl=a:47:5 [Context=main] [Extent=47:5:47:17]
// CHECK: c-index-api-loadTU-test.m:48:12: VarDecl=c:48:12 [Context=main] [Extent=48:12:48:25]
// CHECK: c-index-api-loadTU-test.m:49:13: VarDecl=d:49:13 [Context=main] [Extent=49:13:49:13]


