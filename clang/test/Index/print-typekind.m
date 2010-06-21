@interface Foo
@property (readonly) id x;
-(int) mymethod;
@end

// RUN: c-index-test -test-print-typekind %s | FileCheck %s
// CHECK: ObjCPropertyDecl=x:2:25 typekind=Typedef [canonical=ObjCObjectPointer]
// CHECK: ObjCInstanceMethodDecl=mymethod:3:1 typekind=Invalid [result=Int]


