// RUN: %llvmgcc -x objective-c -fwritable-strings -S %s -o - | FileCheck %s
// CHECK: @.str = private unnamed_addr constant
// CHECK: @.str1 = internal unnamed_addr global

// rdar://7634471

@class NSString;

@interface A
- (void)foo:(NSString*)msg;
- (void)bar:(const char*)msg;
@end

void func(A *a) {
  [a foo:@"Hello world!"];
  [a bar:"Goodbye world!"];
}
