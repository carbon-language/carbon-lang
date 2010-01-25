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
// CHECK-scan: {start_line=8 start_col=1 end_line=8 end_col=7} UnexposedDecl=:8:1
// CHECK-scan: {start_line=8 start_col=8 end_line=8 end_col=10} ObjCClassRef=Foo:8:8
// CHECK-scan: {start_line=8 start_col=11 end_line=10 end_col=5} Invalid Cursor => NoDeclFound
// CHECK-scan: {start_line=10 start_col=6 end_line=10 end_col=14} FunctionDecl=function:10:6 (Definition)
// CHECK-scan: {start_line=10 start_col=15 end_line=10 end_col=17} ObjCClassRef=Foo:8:8
// CHECK-scan: {start_line=10 start_col=18 end_line=10 end_col=23} ParmDecl=arg:10:21 (Definition)
// CHECK-scan: {start_line=10 start_col=24 end_line=10 end_col=25} FunctionDecl=function:10:6 (Definition)
// CHECK-scan: {start_line=11 start_col=1 end_line=13 end_col=1} UnexposedStmt=















// CHECK-load: TestClassForwardDecl.m:10:6: FunctionDecl=function:10:6 (Definition)
// CHECK-load: TestClassForwardDecl.m:10:21: ParmDecl=arg:10:21

