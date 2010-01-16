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

// CHECK: {start_line=1 start_col=1 end_line=2 end_col=62} Invalid Cursor => NoDeclFound
// CHECK: {start_line=3 start_col=1 end_line=6 end_col=1} ObjCInterfaceDecl=Foo:3:1
// CHECK: {start_line=7 start_col=1 end_line=7 end_col=6} ObjCInstanceMethodDecl=foo:7:1
// CHECK: {start_line=7 start_col=7 end_line=7 end_col=7} ObjCInterfaceDecl=Foo:3:1
// CHECK: {start_line=8 start_col=1 end_line=8 end_col=7} ObjCClassMethodDecl=fooC:8:1
// CHECK: {start_line=8 start_col=8 end_line=10 end_col=4} ObjCInterfaceDecl=Foo:3:1
// CHECK: {start_line=10 start_col=5 end_line=11 end_col=1} Invalid Cursor => NoDeclFound
// CHECK: {start_line=12 start_col=1 end_line=16 end_col=4} ObjCInterfaceDecl=Bar:12:1
// CHECK: {start_line=16 start_col=5 end_line=17 end_col=1} Invalid Cursor => NoDeclFound
// CHECK: {start_line=18 start_col=1 end_line=18 end_col=24} ObjCCategoryDecl=FooCat:18:1
// CHECK: {start_line=19 start_col=1 end_line=19 end_col=28} ObjCInstanceMethodDecl=catMethodWithFloat::19:1
// CHECK: {start_line=19 start_col=29 end_line=19 end_col=33} ParmDecl=fArg:19:36
// CHECK: {start_line=19 start_col=34 end_line=19 end_col=35} ObjCInstanceMethodDecl=catMethodWithFloat::19:1
// CHECK: {start_line=19 start_col=36 end_line=19 end_col=39} ParmDecl=fArg:19:36
// CHECK: {start_line=19 start_col=40 end_line=19 end_col=40} ObjCInstanceMethodDecl=catMethodWithFloat::19:1
// CHECK: {start_line=19 start_col=41 end_line=19 end_col=41} ObjCCategoryDecl=FooCat:18:1
// CHECK: {start_line=20 start_col=1 end_line=20 end_col=22} ObjCInstanceMethodDecl=floatMethod:20:1
// CHECK: {start_line=20 start_col=23 end_line=21 end_col=4} ObjCCategoryDecl=FooCat:18:1
// CHECK: {start_line=21 start_col=5 end_line=22 end_col=1} Invalid Cursor => NoDeclFound
// CHECK: {start_line=23 start_col=1 end_line=23 end_col=16} ObjCProtocolDecl=Proto:23:1
// CHECK: {start_line=24 start_col=1 end_line=24 end_col=10} ObjCInstanceMethodDecl=pMethod:24:1
// CHECK: {start_line=24 start_col=11 end_line=25 end_col=4} ObjCProtocolDecl=Proto:23:1
// CHECK: {start_line=25 start_col=5 end_line=26 end_col=1} Invalid Cursor => NoDeclFound
// CHECK: {start_line=27 start_col=1 end_line=27 end_col=23} ObjCProtocolDecl=SubP:27:1
// CHECK: {start_line=28 start_col=1 end_line=28 end_col=11} ObjCInstanceMethodDecl=spMethod:28:1
// CHECK: {start_line=28 start_col=12 end_line=29 end_col=4} ObjCProtocolDecl=SubP:27:1
// CHECK: {start_line=29 start_col=5 end_line=30 end_col=1} Invalid Cursor => NoDeclFound
// CHECK: {start_line=31 start_col=1 end_line=33 end_col=4} ObjCInterfaceDecl=Baz:31:1
// CHECK: {start_line=33 start_col=5 end_line=33 end_col=7} ObjCIvarDecl=_anIVar:33:9
// CHECK: {start_line=33 start_col=8 end_line=33 end_col=8} ObjCInterfaceDecl=Baz:31:1
// CHECK: {start_line=33 start_col=9 end_line=33 end_col=15} ObjCIvarDecl=_anIVar:33:9
// CHECK: {start_line=33 start_col=16 end_line=35 end_col=1} ObjCInterfaceDecl=Baz:31:1
// CHECK: {start_line=36 start_col=1 end_line=36 end_col=20} ObjCInstanceMethodDecl=bazMethod:36:1
// CHECK: {start_line=36 start_col=21 end_line=38 end_col=4} ObjCInterfaceDecl=Baz:31:1
// CHECK: {start_line=38 start_col=5 end_line=39 end_col=1} Invalid Cursor => NoDeclFound
// CHECK: {start_line=40 start_col=1 end_line=41 end_col=2} EnumDecl=:40:1
// CHECK: {start_line=41 start_col=3 end_line=41 end_col=10} EnumConstantDecl=someEnum:41:3
// CHECK: {start_line=41 start_col=11 end_line=42 end_col=1} EnumDecl=:40:1
// CHECK: {start_line=42 start_col=2 end_line=43 end_col=1} Invalid Cursor => NoDeclFound
// CHECK: {start_line=44 start_col=1 end_line=44 end_col=3} FunctionDefn=main:44:5
// CHECK: {start_line=44 start_col=4 end_line=44 end_col=4} Invalid Cursor => NoDeclFound
// CHECK: {start_line=44 start_col=5 end_line=44 end_col=10} FunctionDefn=main:44:5
// CHECK: {start_line=44 start_col=11 end_line=44 end_col=13} ParmDecl=argc:44:15
// CHECK: {start_line=44 start_col=14 end_line=44 end_col=14} FunctionDefn=main:44:5
// CHECK: {start_line=44 start_col=15 end_line=44 end_col=18} ParmDecl=argc:44:15
// CHECK: {start_line=44 start_col=19 end_line=44 end_col=26} FunctionDefn=main:44:5
// CHECK: {start_line=44 start_col=27 end_line=44 end_col=30} ParmDecl=argv:44:34
// CHECK: {start_line=44 start_col=31 end_line=44 end_col=31} FunctionDefn=main:44:5
// CHECK: {start_line=44 start_col=32 end_line=44 end_col=32} ParmDecl=argv:44:34
// CHECK: {start_line=44 start_col=33 end_line=44 end_col=33} FunctionDefn=main:44:5
// CHECK: {start_line=44 start_col=34 end_line=44 end_col=39} ParmDecl=argv:44:34
// CHECK: {start_line=44 start_col=40 end_line=45 end_col=1} FunctionDefn=main:44:5
// CHECK: {start_line=45 start_col=2 end_line=45 end_col=4} ObjCClassRef=Baz:45:8
// CHECK: {start_line=45 start_col=5 end_line=45 end_col=5} FunctionDefn=main:44:5
// CHECK: {start_line=45 start_col=6 end_line=45 end_col=6} VarDecl=bee:45:8
// CHECK: {start_line=45 start_col=7 end_line=45 end_col=7} FunctionDefn=main:44:5
// CHECK: {start_line=45 start_col=8 end_line=45 end_col=10} VarDecl=bee:45:8
// CHECK: {start_line=45 start_col=11 end_line=46 end_col=1} FunctionDefn=main:44:5
// CHECK: {start_line=46 start_col=2 end_line=46 end_col=3} TypedefDecl=id:0:0
// CHECK: {start_line=46 start_col=4 end_line=46 end_col=4} FunctionDefn=main:44:5
// CHECK: {start_line=46 start_col=5 end_line=46 end_col=8} VarDecl=a:46:5
// CHECK: {start_line=46 start_col=9 end_line=46 end_col=9} ObjCSelectorRef=foo:7:1
// CHECK: {start_line=46 start_col=10 end_line=46 end_col=12} VarRef=bee:45:8
// CHECK: {start_line=46 start_col=13 end_line=46 end_col=17} ObjCSelectorRef=foo:7:1
// CHECK: {start_line=46 start_col=18 end_line=47 end_col=1} FunctionDefn=main:44:5
// CHECK: {start_line=47 start_col=2 end_line=47 end_col=3} TypedefDecl=id:0:0
// CHECK: {start_line=47 start_col=4 end_line=47 end_col=4} FunctionDefn=main:44:5
// CHECK: {start_line=47 start_col=5 end_line=47 end_col=5} VarDecl=c:47:12
// CHECK: {start_line=47 start_col=6 end_line=47 end_col=9} ObjCProtocolRef=SubP:27:1
// CHECK: {start_line=47 start_col=10 end_line=47 end_col=10} VarDecl=c:47:12
// CHECK: {start_line=47 start_col=11 end_line=47 end_col=11} FunctionDefn=main:44:5
// CHECK: {start_line=47 start_col=12 end_line=47 end_col=15} VarDecl=c:47:12
// CHECK: {start_line=47 start_col=16 end_line=47 end_col=25} ObjCSelectorRef=fooC:8:1
// CHECK: {start_line=47 start_col=26 end_line=48 end_col=1} FunctionDefn=main:44:5
// CHECK: {start_line=48 start_col=2 end_line=48 end_col=3} TypedefDecl=id:0:0
// CHECK: {start_line=48 start_col=4 end_line=48 end_col=4} FunctionDefn=main:44:5
// CHECK: {start_line=48 start_col=5 end_line=48 end_col=5} VarDecl=d:48:13
// CHECK: {start_line=48 start_col=6 end_line=48 end_col=10} ObjCProtocolRef=Proto:23:1
// CHECK: {start_line=48 start_col=11 end_line=48 end_col=11} VarDecl=d:48:13
// CHECK: {start_line=48 start_col=12 end_line=48 end_col=12} FunctionDefn=main:44:5
// CHECK: {start_line=48 start_col=13 end_line=48 end_col=13} VarDecl=d:48:13
// CHECK: {start_line=48 start_col=14 end_line=49 end_col=1} FunctionDefn=main:44:5
// CHECK: {start_line=49 start_col=2 end_line=49 end_col=2} VarRef=d:48:13
// CHECK: {start_line=49 start_col=3 end_line=49 end_col=5} FunctionDefn=main:44:5
// CHECK: {start_line=49 start_col=6 end_line=49 end_col=6} VarRef=c:47:12
// CHECK: {start_line=49 start_col=7 end_line=50 end_col=1} FunctionDefn=main:44:5
// CHECK: {start_line=50 start_col=2 end_line=50 end_col=2} ObjCSelectorRef=pMethod:24:1
// CHECK: {start_line=50 start_col=3 end_line=50 end_col=3} VarRef=d:48:13
// CHECK: {start_line=50 start_col=4 end_line=50 end_col=12} ObjCSelectorRef=pMethod:24:1
// CHECK: {start_line=50 start_col=13 end_line=51 end_col=1} FunctionDefn=main:44:5
// CHECK: {start_line=51 start_col=2 end_line=51 end_col=2} ObjCSelectorRef=catMethodWithFloat::19:1
// CHECK: {start_line=51 start_col=3 end_line=51 end_col=5} VarRef=bee:45:8
// CHECK: {start_line=51 start_col=6 end_line=51 end_col=25} ObjCSelectorRef=catMethodWithFloat::19:1
// CHECK: {start_line=51 start_col=26 end_line=51 end_col=26} ObjCSelectorRef=floatMethod:20:1
// CHECK: {start_line=51 start_col=27 end_line=51 end_col=29} VarRef=bee:45:8
// CHECK: {start_line=51 start_col=30 end_line=51 end_col=42} ObjCSelectorRef=floatMethod:20:1
// CHECK: {start_line=51 start_col=43 end_line=51 end_col=43} ObjCSelectorRef=catMethodWithFloat::19:1
// CHECK: {start_line=51 start_col=44 end_line=52 end_col=2} FunctionDefn=main:44:5
// CHECK: {start_line=52 start_col=3 end_line=52 end_col=6} FunctionRef=main:44:5
// CHECK: {start_line=52 start_col=7 end_line=52 end_col=7} FunctionDefn=main:44:5
// CHECK: {start_line=52 start_col=8 end_line=52 end_col=15} EnumConstantRef=someEnum:41:3
// CHECK: {start_line=52 start_col=16 end_line=52 end_col=32} FunctionDefn=main:44:5
// CHECK: {start_line=52 start_col=33 end_line=52 end_col=35} VarRef=bee:45:8
// CHECK: {start_line=52 start_col=36 end_line=53 end_col=1} FunctionDefn=main:44:5
// CHECK: {start_line=53 start_col=2 end_line=160 end_col=1} Invalid Cursor => NoDeclFound

