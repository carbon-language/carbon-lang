@interface I
@property (readonly) id prop;
 -(id)prop;
@end

@interface I()
@property (assign,readwrite) id prop;
@end

@implementation I
@synthesize prop = _prop;
@end

rdar://11015325
@interface I1
__attribute__((something)) @interface I2 @end
@end

// RUN: c-index-test -index-file %s > %t
// RUN: FileCheck %s -input-file=%t
// CHECK: [indexDeclaration]: kind: objc-class | name: I | {{.*}} | loc: 1:12
// CHECK: [indexDeclaration]: kind: objc-instance-method | name: prop | {{.*}} | loc: 3:2
// CHECK: [indexDeclaration]: kind: objc-property | name: prop | {{.*}} | loc: 2:25
// CHECK: [indexDeclaration]: kind: objc-category | name:  | {{.*}} | loc: 6:12
// CHECK: [indexDeclaration]: kind: objc-instance-method | name: setProp: | {{.*}} | loc: 7:33
// CHECK: [indexDeclaration]: kind: objc-property | name: prop | {{.*}} | loc: 7:33

// CHECK: [indexDeclaration]: kind: objc-ivar | name: _prop | {{.*}} | loc: 11:20
// CHECK: [indexDeclaration]: kind: objc-instance-method | name: prop | {{.*}} | loc: 11:13 | {{.*}} | lexical-container: [I:10:17]
// CHECK: [indexDeclaration]: kind: objc-instance-method | name: setProp: | {{.*}} | loc: 11:13 | {{.*}} | lexical-container: [I:10:17]
