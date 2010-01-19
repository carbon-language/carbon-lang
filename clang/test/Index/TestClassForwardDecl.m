// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fblocks -emit-pch -x objective-c %s -o %t.ast
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
// CHECK-scan: {start_line=8 start_col=8 end_line=8 end_col=10} ObjCClassRef=Foo:8:8
// CHECK-scan: {start_line=8 start_col=11 end_line=9 end_col=1} Invalid Cursor => NoDeclFound
// CHECK-scan: {start_line=10 start_col=1 end_line=10 end_col=4} FunctionDefn=function:10:6
// CHECK-scan: {start_line=10 start_col=5 end_line=10 end_col=5} Invalid Cursor => NoDeclFound
// CHECK-scan: {start_line=10 start_col=6 end_line=10 end_col=14} FunctionDefn=function:10:6
// CHECK-scan: {start_line=10 start_col=15 end_line=10 end_col=17} ObjCClassRef=Foo:8:8
// CHECK-scan: {start_line=10 start_col=18 end_line=10 end_col=18} FunctionDefn=function:10:6
// CHECK-scan: {start_line=10 start_col=19 end_line=10 end_col=19} ParmDecl=arg:10:21
// CHECK-scan: {start_line=10 start_col=20 end_line=10 end_col=20} FunctionDefn=function:10:6
// CHECK-scan: {start_line=10 start_col=21 end_line=10 end_col=23} ParmDecl=arg:10:21
// CHECK-scan: {start_line=10 start_col=24 end_line=13 end_col=1} FunctionDefn=function:10:6
// CHECK-scan: {start_line=13 start_col=2 end_line=46 end_col=1} Invalid Cursor => NoDeclFound















// CHECK-load: TestClassForwardDecl.m:10:6: FunctionDefn=function:10:6
// CHECK-load: TestClassForwardDecl.m:10:21: ParmDecl=arg:10:21

