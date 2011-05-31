// RUN: %clang -cc1 -triple x86_64-apple-darwin10  -g -S %s -o %t
// RUN: FileCheck %s < %t

//CHECK:        .long   Lset6
//CHECK-NEXT:   .long   256
//CHECK-NEXT:   .asciz   "H"
//CHECK-NEXT:   .long   0
//CHECK-NEXT:   Lpubtypes_end1:

@interface H
-(void) foo;
@end

@implementation H
-(void) foo {
}
@end

