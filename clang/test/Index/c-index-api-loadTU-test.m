// RUN: clang -cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fblocks -emit-pch -x objective-c %s -o %t.ast
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

// CHECK: c-index-api-loadTU-test.m:20:12: ObjCInterfaceDecl=Foo:20:1 [Context=c-index-api-loadTU-test.m]
// CHECK: c-index-api-loadTU-test.m:24:1: ObjCInstanceMethodDecl=foo:24:1 [Context=Foo]
// CHECK: c-index-api-loadTU-test.m:25:1: ObjCClassMethodDecl=fooC:25:1 [Context=Foo]
// CHECK: c-index-api-loadTU-test.m:29:12: ObjCInterfaceDecl=Bar:29:1 [Context=c-index-api-loadTU-test.m]
// CHECK: c-index-api-loadTU-test.m:29:18: ObjCSuperClassRef=Foo:29:1 [Context=Bar]
// CHECK: c-index-api-loadTU-test.m:35:1: ObjCCategoryDecl=FooCat:35:1 [Context=c-index-api-loadTU-test.m]
// CHECK: c-index-api-loadTU-test.m:20:1: ObjCClassRef=Foo:35:1 [Context=FooCat]
// CHECK: c-index-api-loadTU-test.m:36:1: ObjCInstanceMethodDecl=catMethodWithFloat::36:1 [Context=FooCat]
// CHECK: c-index-api-loadTU-test.m:37:1: ObjCInstanceMethodDecl=floatMethod:37:1 [Context=FooCat]
// CHECK: c-index-api-loadTU-test.m:40:1: ObjCProtocolDecl=Proto:40:1 [Context=c-index-api-loadTU-test.m]
// CHECK: c-index-api-loadTU-test.m:41:1: ObjCInstanceMethodDecl=pMethod:41:1 [Context=Proto]
// CHECK: c-index-api-loadTU-test.m:44:1: ObjCProtocolDecl=SubP:44:1 [Context=c-index-api-loadTU-test.m]
// CHECK: c-index-api-loadTU-test.m:40:1: ObjCProtocolRef=Proto:40:1 [Context=SubP]
// CHECK: c-index-api-loadTU-test.m:45:1: ObjCInstanceMethodDecl=spMethod:45:1 [Context=SubP]
// CHECK: c-index-api-loadTU-test.m:48:12: ObjCInterfaceDecl=Baz:48:1 [Context=c-index-api-loadTU-test.m]
// CHECK: c-index-api-loadTU-test.m:48:18: ObjCSuperClassRef=Bar:48:1 [Context=Baz]
// CHECK: c-index-api-loadTU-test.m:44:1: ObjCProtocolRef=SubP:44:1 [Context=Baz]
// CHECK: c-index-api-loadTU-test.m:50:9: ObjCIvarDecl=_anIVar:50:9 [Context=Baz]
// CHECK: c-index-api-loadTU-test.m:53:1: ObjCInstanceMethodDecl=bazMethod:53:1 [Context=Baz]
// CHECK: c-index-api-loadTU-test.m:57:1: EnumDecl=:57:1 [Context=c-index-api-loadTU-test.m]
// CHECK: c-index-api-loadTU-test.m:58:3: EnumConstantDecl=someEnum:58:3 [Context=]

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

// CHECK: c-index-api-loadTU-test.m:83:5: FunctionDefn=main [Context=c-index-api-loadTU-test.m]
// CHECK: c-index-api-loadTU-test.m:83:15: ParmDecl=argc:83:15 [Context=main]
// CHECK: c-index-api-loadTU-test.m:83:34: ParmDecl=argv:83:34 [Context=main]
// CHECK: c-index-api-loadTU-test.m:84:8: VarDecl=bee:84:8 [Context=main]
// CHECK: c-index-api-loadTU-test.m:85:5: VarDecl=a:85:5 [Context=main]
// CHECK: c-index-api-loadTU-test.m:86:12: VarDecl=c:86:12 [Context=main]
// CHECK: c-index-api-loadTU-test.m:87:13: VarDecl=d:87:13 [Context=main]
