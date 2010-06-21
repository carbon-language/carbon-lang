@interface Foo
@property (readonly) id x;
@end

// RUN: c-index-test -test-print-typekind %s | FileCheck %s
// CHECK: ObjCPropertyDecl=x:2:25 typekind=Typedef [canonical=ObjCObjectPointer]

