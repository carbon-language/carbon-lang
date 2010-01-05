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

// CHECK: c-index-api-loadTU-test.m:19:12: ObjCInterfaceDecl=Foo:19:1 [Context=c-index-api-loadTU-test.m] [Extent=19:1:26:1]
// CHECK: c-index-api-loadTU-test.m:23:1: ObjCInstanceMethodDecl=foo:23:1 [Context=Foo] [Extent=23:1:23:6]
// CHECK: c-index-api-loadTU-test.m:24:1: ObjCClassMethodDecl=fooC:24:1 [Context=Foo] [Extent=24:1:24:7]
// CHECK: c-index-api-loadTU-test.m:28:12: ObjCInterfaceDecl=Bar:28:1 [Context=c-index-api-loadTU-test.m] [Extent=28:1:32:1]
// CHECK: c-index-api-loadTU-test.m:28:18: ObjCSuperClassRef=Foo:28:1 [Context=Bar] [Extent=28:1:32:1]
// CHECK: c-index-api-loadTU-test.m:34:1: ObjCCategoryDecl=FooCat:34:1 [Context=c-index-api-loadTU-test.m] [Extent=34:1:37:1]
// CHECK: c-index-api-loadTU-test.m:19:1: ObjCClassRef=Foo:34:1 [Context=FooCat] [Extent=34:1:37:1]
// CHECK: c-index-api-loadTU-test.m:35:1: ObjCInstanceMethodDecl=catMethodWithFloat::35:1 [Context=FooCat] [Extent=35:1:35:40]
// CHECK: c-index-api-loadTU-test.m:36:1: ObjCInstanceMethodDecl=floatMethod:36:1 [Context=FooCat] [Extent=36:1:36:22]
// CHECK: c-index-api-loadTU-test.m:39:1: ObjCProtocolDecl=Proto:39:1 [Context=c-index-api-loadTU-test.m] [Extent=39:1:41:1]
// CHECK: c-index-api-loadTU-test.m:40:1: ObjCInstanceMethodDecl=pMethod:40:1 [Context=Proto] [Extent=40:1:40:10]
// CHECK: c-index-api-loadTU-test.m:43:1: ObjCProtocolDecl=SubP:43:1 [Context=c-index-api-loadTU-test.m] [Extent=43:1:45:1]
// CHECK: c-index-api-loadTU-test.m:39:1: ObjCProtocolRef=Proto:39:1 [Context=SubP] [Extent=39:1:41:1]
// CHECK: c-index-api-loadTU-test.m:44:1: ObjCInstanceMethodDecl=spMethod:44:1 [Context=SubP] [Extent=44:1:44:11]
// CHECK: c-index-api-loadTU-test.m:47:12: ObjCInterfaceDecl=Baz:47:1 [Context=c-index-api-loadTU-test.m] [Extent=47:1:54:1]
// CHECK: c-index-api-loadTU-test.m:47:18: ObjCSuperClassRef=Bar:47:1 [Context=Baz] [Extent=47:1:54:1]
// CHECK: c-index-api-loadTU-test.m:43:1: ObjCProtocolRef=SubP:43:1 [Context=Baz] [Extent=43:1:45:1]
// CHECK: c-index-api-loadTU-test.m:49:9: ObjCIvarDecl=_anIVar:49:9 [Context=Baz] [Extent=49:9:49:9]
// CHECK: c-index-api-loadTU-test.m:52:1: ObjCInstanceMethodDecl=bazMethod:52:1 [Context=Baz] [Extent=52:1:52:20]
// CHECK: c-index-api-loadTU-test.m:56:1: EnumDecl=:56:1 [Context=c-index-api-loadTU-test.m] [Extent=56:1:58:1]
// CHECK: c-index-api-loadTU-test.m:57:3: EnumConstantDecl=someEnum:57:3 [Context=] [Extent=57:3:57:3]
// CHECK: c-index-api-loadTU-test.m:60:5: FunctionDefn=main [Context=c-index-api-loadTU-test.m] [Extent=60:5:69:1]
// CHECK: c-index-api-loadTU-test.m:60:15: ParmDecl=argc:60:15 [Context=main] [Extent=60:15:60:15]
// CHECK: c-index-api-loadTU-test.m:60:34: ParmDecl=argv:60:34 [Context=main] [Extent=60:34:60:34]
// CHECK: c-index-api-loadTU-test.m:61:8: VarDecl=bee:61:8 [Context=main] [Extent=61:8:61:8]
// CHECK: c-index-api-loadTU-test.m:62:5: VarDecl=a:62:5 [Context=main] [Extent=62:5:62:17]
// CHECK: c-index-api-loadTU-test.m:63:12: VarDecl=c:63:12 [Context=main] [Extent=63:12:63:25]
// CHECK: c-index-api-loadTU-test.m:64:13: VarDecl=d:64:13 [Context=main] [Extent=64:13:64:13]

