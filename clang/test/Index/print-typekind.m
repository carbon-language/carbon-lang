@interface Foo
@property (readonly) id x;
-(int) mymethod;
-(int) mymethod2:(int)x blah:(float)y;
@end

// RUN: c-index-test -test-print-typekind %s | FileCheck %s
// CHECK: ObjCPropertyDecl=x:2:25 typekind=Typedef [canonical=ObjCObjectPointer]
// CHECK: ObjCInstanceMethodDecl=mymethod:3:8 typekind=Invalid [result=Int]
// CHECK: ObjCInstanceMethodDecl=mymethod2:blah::4:8 typekind=Invalid [result=Int] [args= Int Float]
