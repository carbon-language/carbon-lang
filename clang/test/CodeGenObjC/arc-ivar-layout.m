// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -triple x86_64-apple-darwin -S %s -o %t-64.s
// RUN: FileCheck -check-prefix CHECK-LP64 --input-file=%t-64.s %s
// REQUIRES: x86-registered-target
// rdar://8991729

@interface NSObject {
  id isa;
}
@end

@interface AllPointers : NSObject {
    id foo;
    id __strong bar;    
    NSObject *bletch;
}
@end

@implementation AllPointers
@end
// CHECK-LP64: L_OBJC_CLASS_NAME_1:
// CHECK-LP64-NEXT: .asciz	"\003"

@class NSString, NSNumber;
@interface A : NSObject {
   NSString *foo;
   NSNumber *bar;
   unsigned int bletch;
   __weak id delegate;
}
@end

@interface B : A {
  unsigned int x;
  NSString *y;
  NSString *z;
}
@end

@implementation A @end

@implementation B @end

// CHECK-LP64: L_OBJC_CLASS_NAME_15:
// CHECK-LP64-NEXT: .asciz	"\022"

@interface UnsafePerson {
@public
    __unsafe_unretained id name;
    __unsafe_unretained id age;
    id value;
}
@end

@implementation UnsafePerson @end
// CHECK-LP64: L_OBJC_CLASS_NAME_20:
// CHECK-LP64-NEXT: .asciz      "!"

// rdar://16136439
@interface rdar16136439
    @property (nonatomic, readonly, weak) id first;
@end

@implementation rdar16136439 @end
// CHECK-LP64: L_OBJC_PROP_NAME_ATTR_29:
// CHECK-LP64-NEXT: .asciz  "T@,R,W,N,V_first"
