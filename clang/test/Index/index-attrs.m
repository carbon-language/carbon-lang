@class Foo;
@interface Bar
@property (retain) __attribute__((iboutletcollection(Foo))) Foo *prop;
@end

// RUN: c-index-test -index-file %s | FileCheck %s
// CHECK:      <attribute>: attribute(iboutletcollection)= [IBOutletCollection=ObjCInterface]
