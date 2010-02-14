// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fblocks -emit-pch -x objective-c %s -o %t.ast
// RUN: c-index-test -test-file-scan %t.ast %s | FileCheck -check-prefix=scan %s
// RUN: c-index-test -test-load-tu %t.ast local | FileCheck -check-prefix=load %s

// This test checks how the @class resolves as a cursor when there is a real definition
// that follows. <rdar://problem/7383421>

@class Foo;

@interface Foo
@end

void function(Foo * arg)
{
    // nothing here.
}

// CHECK-scan: [1:1 - 7:1] Invalid Cursor => NoDeclFound
// CHECK-scan: [8:1 - 8:7] UnexposedDecl=:8:1
// CHECK-scan: [8:8 - 8:10] ObjCClassRef=Foo:10:12
// CHECK-scan: [8:11 - 9:1] Invalid Cursor => NoDeclFound
// CHECK-scan: [10:1 - 11:4] ObjCInterfaceDecl=Foo:10:12
// CHECK-scan: [11:5 - 13:5] Invalid Cursor => NoDeclFound
// CHECK-scan: [13:6 - 13:14] FunctionDecl=function:13:6 (Definition)
// CHECK-scan: [13:15 - 13:17] ObjCClassRef=Foo:10:12
// CHECK-scan: [13:18 - 13:23] ParmDecl=arg:13:21 (Definition)
// CHECK-scan: [13:24 - 13:25] FunctionDecl=function:13:6 (Definition)
// CHECK-scan: [14:1 - 16:1] UnexposedStmt=

// CHECK-load: TestClassDecl.m:10:12: ObjCInterfaceDecl=Foo:10:12 Extent=[10:1 - 11:4]
// CHECK-load: TestClassDecl.m:13:6: FunctionDecl=function:13:6 (Definition) Extent=[13:6 - 16:1]
// CHECK-load: TestClassDecl.m:13:21: ParmDecl=arg:13:21 (Definition) Extent=[13:15 - 13:23]

