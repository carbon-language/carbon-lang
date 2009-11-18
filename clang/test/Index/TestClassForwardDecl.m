// RUN: clang-cc -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fblocks -emit-pch -x objective-c %s -o %t.ast
// RUN: c-index-test -test-file-scan %t.ast %s | FileCheck -check-prefix=scan %s
// RUN: c-index-test -test-load-tu %t.ast local | FileCheck -check-prefix=load %s

// This test checks how the @class resolves as a cursor when the @interface is implicitly defined.
// See TestClassDecl.m for the corresponding test case. (<rdar://problem/7383421>)

@class Foo;

void function(Foo * arg)
{
    // nothing here.
}

// CHECK-scan: {start_line=1 start_col=1 end_line=7 end_col=1} Invalid Cursor => NoDeclFound
// CHECK-scan: {start_line=8 start_col=1 end_line=8 end_col=7} Invalid Cursor => NotImplemented
// CHECK-scan: {start_line=8 start_col=8 end_line=8 end_col=10} ObjCClassRef=Foo:8:1
// CHECK-scan: {start_line=8 start_col=11 end_line=9 end_col=1} Invalid Cursor => NoDeclFound
// CHECK-scan: {start_line=10 start_col=1 end_line=10 end_col=4} FunctionDecl=function:10:6
// CHECK-scan: {start_line=10 start_col=5 end_line=10 end_col=5} Invalid Cursor => NoDeclFound
// CHECK-scan: {start_line=10 start_col=6 end_line=10 end_col=14} FunctionDecl=function:10:6
// CHECK-scan: {start_line=10 start_col=15 end_line=10 end_col=17} ObjCClassRef=Foo:10:21
// CHECK-scan: {start_line=10 start_col=18 end_line=10 end_col=18} FunctionDecl=function:10:6
// CHECK-scan: {start_line=10 start_col=19 end_line=10 end_col=19} ParmDecl=arg:10:21
// CHECK-scan: {start_line=10 start_col=20 end_line=10 end_col=20} FunctionDecl=function:10:6
// CHECK-scan: {start_line=10 start_col=21 end_line=10 end_col=23} ParmDecl=arg:10:21
// CHECK-scan: {start_line=10 start_col=24 end_line=13 end_col=1} FunctionDecl=function:10:6
// CHECK-scan: {start_line=13 start_col=2 end_line=46 end_col=1} Invalid Cursor => NoDeclFound

// CHECK-load: <invalid loc>:0:0: TypedefDecl=__int128_t:0:0 [Context=TestClassForwardDecl.m]
// CHECK-load: <invalid loc>:0:0: TypedefDecl=__uint128_t:0:0 [Context=TestClassForwardDecl.m]
// CHECK-load: <invalid loc>:0:0: StructDecl=objc_selector:0:0 [Context=TestClassForwardDecl.m]
// CHECK-load: <invalid loc>:0:0: TypedefDecl=SEL:0:0 [Context=TestClassForwardDecl.m]
// CHECK-load: <invalid loc>:0:0: ObjCInterfaceDecl=Protocol:0:0 [Context=TestClassForwardDecl.m]
// CHECK-load: <invalid loc>:0:0: TypedefDecl=id:0:0 [Context=TestClassForwardDecl.m]
// CHECK-load: <invalid loc>:0:0: TypedefDecl=Class:0:0 [Context=TestClassForwardDecl.m]
// CHECK-load: <invalid loc>:80:16: StructDecl=__va_list_tag:80:16 [Context=TestClassForwardDecl.m]
// CHECK-load: <invalid loc>:80:42: FieldDecl=gp_offset:80:42 [Context=__va_list_tag]
// CHECK-load: <invalid loc>:80:63: FieldDecl=fp_offset:80:63 [Context=__va_list_tag]
// CHECK-load: <invalid loc>:80:81: FieldDecl=overflow_arg_area:80:81 [Context=__va_list_tag]
// CHECK-load: <invalid loc>:80:107: FieldDecl=reg_save_area:80:107 [Context=__va_list_tag]
// CHECK-load: <invalid loc>:80:123: TypedefDecl=__va_list_tag:80:123 [Context=TestClassForwardDecl.m]
// CHECK-load: <invalid loc>:80:159: TypedefDecl=__builtin_va_list:80:159 [Context=TestClassForwardDecl.m]
// CHECK-load: TestClassForwardDecl.m:10:6: FunctionDefn=function [Context=TestClassForwardDecl.m]
// CHECK-load: TestClassForwardDecl.m:10:21: ParmDecl=arg:10:21 [Context=function]

