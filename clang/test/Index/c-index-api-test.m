// RUN: clang-cc -triple i386-apple-darwin9 -emit-pch %s -o %t.ast &&
// RUN: c-index-test %t.ast all | FileCheck %s

// CHECK: <invalid loc>:0:0: StructDecl=objc_selector [Context=c-index-api-test.m]
// CHECK: <invalid loc>:0:0: TypedefDecl=SEL [Context=c-index-api-test.m]
// CHECK: <invalid loc>:0:0: ObjCInterfaceDecl=Protocol [Context=c-index-api-test.m]
// CHECK: <invalid loc>:0:0: TypedefDecl=id [Context=c-index-api-test.m]
// CHECK: <invalid loc>:0:0: TypedefDecl=Class [Context=c-index-api-test.m]
// 

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

// CHECK: c-index-api-test.m:11:12: ObjCInterfaceDecl=Foo [Context=c-index-api-test.m]
// CHECK: c-index-api-test.m:15:1: ObjCInstanceMethodDecl=foo [Context=Foo]
// CHECK: c-index-api-test.m:16:1: ObjCClassMethodDecl=fooC [Context=Foo]
// CHECK: c-index-api-test.m:20:12: ObjCInterfaceDecl=Bar [Context=c-index-api-test.m]
// CHECK: c-index-api-test.m:20:18: ObjCSuperClassRef=Foo [Context=Bar]
// CHECK: c-index-api-test.m:26:1: ObjCCategoryDecl=FooCat [Context=c-index-api-test.m]
// CHECK: <invalid loc>:0:0: ObjCClassRef=Foo [Context=FooCat]
// CHECK: c-index-api-test.m:27:1: ObjCInstanceMethodDecl=catMethodWithFloat: [Context=FooCat]
// CHECK: c-index-api-test.m:28:1: ObjCInstanceMethodDecl=floatMethod [Context=FooCat]
// CHECK: c-index-api-test.m:31:1: ObjCProtocolDecl=Proto [Context=c-index-api-test.m]
// CHECK: c-index-api-test.m:32:1: ObjCInstanceMethodDecl=pMethod [Context=Proto]
// CHECK: c-index-api-test.m:35:1: ObjCProtocolDecl=SubP [Context=c-index-api-test.m]
// CHECK: c-index-api-test.m:31:1: ObjCProtocolRef=Proto [Context=SubP]
// CHECK: c-index-api-test.m:36:1: ObjCInstanceMethodDecl=spMethod [Context=SubP]
// CHECK: c-index-api-test.m:39:12: ObjCInterfaceDecl=Baz [Context=c-index-api-test.m]
// CHECK: c-index-api-test.m:39:18: ObjCSuperClassRef=Bar [Context=Baz]
// CHECK: c-index-api-test.m:35:1: ObjCProtocolRef=SubP [Context=Baz]
// CHECK: c-index-api-test.m:41:9: ObjCIvarDecl=_anIVar [Context=Baz]
// CHECK: c-index-api-test.m:44:1: ObjCInstanceMethodDecl=bazMethod [Context=Baz]
// CHECK: c-index-api-test.m:48:1: EnumDecl= [Context=c-index-api-test.m]
// CHECK: c-index-api-test.m:49:3: EnumConstantDecl=someEnum [Context=]

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

// CHECK: c-index-api-test.m:74:5: FunctionDefn=main [Context=c-index-api-test.m]
// CHECK: c-index-api-test.m:74:15: ParmDecl=argc [Context=main]
// CHECK: c-index-api-test.m:74:34: ParmDecl=argv [Context=main]
// CHECK: c-index-api-test.m:75:8: VarDecl=bee [Context=main]
// CHECK: c-index-api-test.m:76:5: VarDecl=a [Context=main]
// CHECK: c-index-api-test.m:77:12: VarDecl=c [Context=main]
// CHECK: c-index-api-test.m:78:13: VarDecl=d [Context=main]
// CHECK: c-index-api-test.m:75:8: VarDecl=bee [Context:bee]
// CHECK: c-index-api-test.m:75:9: VarDecl=bee [Context:bee]
// CHECK: c-index-api-test.m:75:10: VarDecl=bee [Context:bee]
// CHECK: c-index-api-test.m:76:5: VarDecl=a [Context:a]
// CHECK: c-index-api-test.m:76:6: VarDecl=a [Context:a]
// CHECK: c-index-api-test.m:76:7: VarDecl=a [Context:a]
// CHECK: c-index-api-test.m:76:8: VarDecl=a [Context:a]
// CHECK: c-index-api-test.m:76:9: ObjCSelectorRef=foo:15:1 [Context:a]
// CHECK: c-index-api-test.m:76:10: VarRef=bee:75:8 [Context:a]
// CHECK: c-index-api-test.m:76:11: VarRef=bee:75:8 [Context:a]
// CHECK: c-index-api-test.m:76:12: VarRef=bee:75:8 [Context:a]
// CHECK: c-index-api-test.m:76:13: ObjCSelectorRef=foo:15:1 [Context:a]
// CHECK: c-index-api-test.m:76:14: ObjCSelectorRef=foo:15:1 [Context:a]
// CHECK: c-index-api-test.m:76:15: ObjCSelectorRef=foo:15:1 [Context:a]
// CHECK: c-index-api-test.m:76:16: ObjCSelectorRef=foo:15:1 [Context:a]
// CHECK: c-index-api-test.m:76:17: ObjCSelectorRef=foo:15:1 [Context:a]
// CHECK: c-index-api-test.m:77:12: VarDecl=c [Context:c]
// CHECK: c-index-api-test.m:77:13: VarDecl=c [Context:c]
// CHECK: c-index-api-test.m:77:14: VarDecl=c [Context:c]
// CHECK: c-index-api-test.m:77:15: VarDecl=c [Context:c]
// CHECK: c-index-api-test.m:77:16: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:77:17: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:77:18: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:77:19: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:77:20: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:77:21: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:77:22: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:77:23: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:77:24: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:77:25: ObjCSelectorRef=fooC:16:1 [Context:c]
// CHECK: c-index-api-test.m:78:13: VarDecl=d [Context:d]
// CHECK: c-index-api-test.m:79:2: VarRef=d:78:13 [Context:main]
// CHECK: c-index-api-test.m:79:6: VarRef=c:77:12 [Context:main]
// CHECK: c-index-api-test.m:80:2: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:80:3: VarRef=d:78:13 [Context:main]
// CHECK: c-index-api-test.m:80:4: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:80:5: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:80:6: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:80:7: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:80:8: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:80:9: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:80:10: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:80:11: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:80:12: ObjCSelectorRef=pMethod:32:1 [Context:main]
// CHECK: c-index-api-test.m:81:2: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:3: VarRef=bee:75:8 [Context:main]
// CHECK: c-index-api-test.m:81:4: VarRef=bee:75:8 [Context:main]
// CHECK: c-index-api-test.m:81:5: VarRef=bee:75:8 [Context:main]
// CHECK: c-index-api-test.m:81:6: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:7: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:8: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:9: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:10: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:11: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:12: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:13: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:14: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:15: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:16: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:17: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:18: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:19: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:20: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:21: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:22: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:23: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:24: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:25: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:81:26: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:27: VarRef=bee:75:8 [Context:main]
// CHECK: c-index-api-test.m:81:28: VarRef=bee:75:8 [Context:main]
// CHECK: c-index-api-test.m:81:29: VarRef=bee:75:8 [Context:main]
// CHECK: c-index-api-test.m:81:30: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:31: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:32: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:33: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:34: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:35: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:36: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:37: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:38: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:39: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:40: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:41: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:42: ObjCSelectorRef=floatMethod:28:1 [Context:main]
// CHECK: c-index-api-test.m:81:43: ObjCSelectorRef=catMethodWithFloat::27:1 [Context:main]
// CHECK: c-index-api-test.m:82:3: FunctionRef=main:74:5 [Context:main]
// CHECK: c-index-api-test.m:82:4: FunctionRef=main:74:5 [Context:main]
// CHECK: c-index-api-test.m:82:5: FunctionRef=main:74:5 [Context:main]
// CHECK: c-index-api-test.m:82:6: FunctionRef=main:74:5 [Context:main]
// CHECK: c-index-api-test.m:82:8: EnumConstantRef=someEnum:49:3 [Context:main]
// CHECK: c-index-api-test.m:82:9: EnumConstantRef=someEnum:49:3 [Context:main]
// CHECK: c-index-api-test.m:82:10: EnumConstantRef=someEnum:49:3 [Context:main]
// CHECK: c-index-api-test.m:82:11: EnumConstantRef=someEnum:49:3 [Context:main]
// CHECK: c-index-api-test.m:82:12: EnumConstantRef=someEnum:49:3 [Context:main]
// CHECK: c-index-api-test.m:82:13: EnumConstantRef=someEnum:49:3 [Context:main]
// CHECK: c-index-api-test.m:82:14: EnumConstantRef=someEnum:49:3 [Context:main]
// CHECK: c-index-api-test.m:82:15: EnumConstantRef=someEnum:49:3 [Context:main]
// CHECK: c-index-api-test.m:82:33: VarRef=bee:75:8 [Context:main]
// CHECK: c-index-api-test.m:82:34: VarRef=bee:75:8 [Context:main]
// CHECK: c-index-api-test.m:82:35: VarRef=bee:75:8 [Context:main]
