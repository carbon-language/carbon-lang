// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fblocks -emit-pch -x objective-c %s -o %t.ast
// RUN: c-index-test -test-load-tu %t.ast scan-function | FileCheck %s

















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

// CHECK: c-index-api-fn-scan.m:84:2: ObjCClassRef=Baz:48:1
// CHECK: c-index-api-fn-scan.m:84:3: ObjCClassRef=Baz:48:1
// CHECK: c-index-api-fn-scan.m:84:4: ObjCClassRef=Baz:48:1
// CHECK: c-index-api-fn-scan.m:84:6: VarDecl=bee:84:8
// CHECK: c-index-api-fn-scan.m:84:8: VarDecl=bee:84:8
// CHECK: c-index-api-fn-scan.m:84:9: VarDecl=bee:84:8
// CHECK: c-index-api-fn-scan.m:84:10: VarDecl=bee:84:8
// CHECK: <invalid loc>:85:2: TypedefDecl=id:0:0
// CHECK: <invalid loc>:85:3: TypedefDecl=id:0:0
// CHECK: c-index-api-fn-scan.m:85:5: VarDecl=a:85:5
// CHECK: c-index-api-fn-scan.m:85:6: VarDecl=a:85:5
// CHECK: c-index-api-fn-scan.m:85:7: VarDecl=a:85:5
// CHECK: c-index-api-fn-scan.m:85:8: VarDecl=a:85:5
// CHECK: c-index-api-fn-scan.m:85:9: ObjCSelectorRef=foo:24:1
// CHECK: c-index-api-fn-scan.m:85:10: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:85:11: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:85:12: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:85:13: ObjCSelectorRef=foo:24:1
// CHECK: c-index-api-fn-scan.m:85:14: ObjCSelectorRef=foo:24:1
// CHECK: c-index-api-fn-scan.m:85:15: ObjCSelectorRef=foo:24:1
// CHECK: c-index-api-fn-scan.m:85:16: ObjCSelectorRef=foo:24:1
// CHECK: c-index-api-fn-scan.m:85:17: ObjCSelectorRef=foo:24:1
// CHECK: <invalid loc>:86:2: TypedefDecl=id:0:0
// CHECK: <invalid loc>:86:3: TypedefDecl=id:0:0
// CHECK: c-index-api-fn-scan.m:86:5: VarDecl=c:86:12
// CHECK: c-index-api-fn-scan.m:86:6: ObjCProtocolRef=SubP:44:1
// CHECK: c-index-api-fn-scan.m:86:7: ObjCProtocolRef=SubP:44:1
// CHECK: c-index-api-fn-scan.m:86:8: ObjCProtocolRef=SubP:44:1
// CHECK: c-index-api-fn-scan.m:86:9: ObjCProtocolRef=SubP:44:1
// CHECK: c-index-api-fn-scan.m:86:10: VarDecl=c:86:12
// CHECK: c-index-api-fn-scan.m:86:12: VarDecl=c:86:12
// CHECK: c-index-api-fn-scan.m:86:13: VarDecl=c:86:12
// CHECK: c-index-api-fn-scan.m:86:14: VarDecl=c:86:12
// CHECK: c-index-api-fn-scan.m:86:15: VarDecl=c:86:12
// CHECK: c-index-api-fn-scan.m:86:16: ObjCSelectorRef=fooC:25:1
// CHECK: c-index-api-fn-scan.m:86:17: ObjCSelectorRef=fooC:25:1
// CHECK: c-index-api-fn-scan.m:86:18: ObjCSelectorRef=fooC:25:1
// CHECK: c-index-api-fn-scan.m:86:19: ObjCSelectorRef=fooC:25:1
// CHECK: c-index-api-fn-scan.m:86:20: ObjCSelectorRef=fooC:25:1
// CHECK: c-index-api-fn-scan.m:86:21: ObjCSelectorRef=fooC:25:1
// CHECK: c-index-api-fn-scan.m:86:22: ObjCSelectorRef=fooC:25:1
// CHECK: c-index-api-fn-scan.m:86:23: ObjCSelectorRef=fooC:25:1
// CHECK: c-index-api-fn-scan.m:86:24: ObjCSelectorRef=fooC:25:1
// CHECK: c-index-api-fn-scan.m:86:25: ObjCSelectorRef=fooC:25:1
// CHECK: <invalid loc>:87:2: TypedefDecl=id:0:0
// CHECK: <invalid loc>:87:3: TypedefDecl=id:0:0
// CHECK: c-index-api-fn-scan.m:87:5: VarDecl=d:87:13
// CHECK: c-index-api-fn-scan.m:87:6: ObjCProtocolRef=Proto:40:1
// CHECK: c-index-api-fn-scan.m:87:7: ObjCProtocolRef=Proto:40:1
// CHECK: c-index-api-fn-scan.m:87:8: ObjCProtocolRef=Proto:40:1
// CHECK: c-index-api-fn-scan.m:87:9: ObjCProtocolRef=Proto:40:1
// CHECK: c-index-api-fn-scan.m:87:10: ObjCProtocolRef=Proto:40:1
// CHECK: c-index-api-fn-scan.m:87:11: VarDecl=d:87:13
// CHECK: c-index-api-fn-scan.m:87:13: VarDecl=d:87:13
// CHECK: c-index-api-fn-scan.m:88:2: VarRef=d:87:13
// CHECK: c-index-api-fn-scan.m:88:6: VarRef=c:86:12
// CHECK: c-index-api-fn-scan.m:89:2: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:89:3: VarRef=d:87:13
// CHECK: c-index-api-fn-scan.m:89:4: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:89:5: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:89:6: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:89:7: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:89:8: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:89:9: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:89:10: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:89:11: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:89:12: ObjCSelectorRef=pMethod:41:1
// CHECK: c-index-api-fn-scan.m:90:2: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:3: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:90:4: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:90:5: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:90:6: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:7: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:8: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:9: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:10: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:11: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:12: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:13: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:14: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:15: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:16: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:17: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:18: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:19: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:20: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:21: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:22: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:23: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:24: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:25: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:90:26: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:27: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:90:28: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:90:29: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:90:30: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:31: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:32: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:33: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:34: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:35: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:36: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:37: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:38: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:39: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:40: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:41: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:42: ObjCSelectorRef=floatMethod:37:1
// CHECK: c-index-api-fn-scan.m:90:43: ObjCSelectorRef=catMethodWithFloat::36:1
// CHECK: c-index-api-fn-scan.m:91:3: FunctionRef=main:83:5
// CHECK: c-index-api-fn-scan.m:91:4: FunctionRef=main:83:5
// CHECK: c-index-api-fn-scan.m:91:5: FunctionRef=main:83:5
// CHECK: c-index-api-fn-scan.m:91:6: FunctionRef=main:83:5
// CHECK: c-index-api-fn-scan.m:91:8: EnumConstantRef=someEnum:58:3
// CHECK: c-index-api-fn-scan.m:91:9: EnumConstantRef=someEnum:58:3
// CHECK: c-index-api-fn-scan.m:91:10: EnumConstantRef=someEnum:58:3
// CHECK: c-index-api-fn-scan.m:91:11: EnumConstantRef=someEnum:58:3
// CHECK: c-index-api-fn-scan.m:91:12: EnumConstantRef=someEnum:58:3
// CHECK: c-index-api-fn-scan.m:91:13: EnumConstantRef=someEnum:58:3
// CHECK: c-index-api-fn-scan.m:91:14: EnumConstantRef=someEnum:58:3
// CHECK: c-index-api-fn-scan.m:91:15: EnumConstantRef=someEnum:58:3
// CHECK: c-index-api-fn-scan.m:91:33: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:91:34: VarRef=bee:84:8
// CHECK: c-index-api-fn-scan.m:91:35: VarRef=bee:84:8
