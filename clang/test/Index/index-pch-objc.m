// RUN: c-index-test -write-pch %t.pch %s
// RUN: env LIBCLANG_NOTHREADS=1 c-index-test -index-tu %t.pch | FileCheck %s

@interface SomeClass
@property (retain) id foo;
@end
@implementation SomeClass
@end

// CHECK: [indexDeclaration]: kind: objc-ivar | name: _foo
