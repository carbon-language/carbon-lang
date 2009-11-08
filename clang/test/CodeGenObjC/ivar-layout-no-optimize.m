// RUN: clang-cc -fobjc-gc -triple x86_64-apple-darwin -O0 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s

@interface NSObject {
  id isa;
}
@end

@interface AllPointers : NSObject {
    id foo;
    void *__strong bar;    NSObject *bletch;}
@end
@implementation AllPointers
@end

// CHECK-LP64: L_OBJC_CLASS_NAME_6:
// CHECK-LP64-NEXT: .asciz	"\004"
