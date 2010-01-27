// RUN: %clang -ccc-host-triple i386-apple-darwin9 -arch armv7 -flto -S -o - %s | FileCheck %s

// CHECK: @f0
// CHECK-NOT: ssp
// CHECK: ) {
// CHECK: @__f0_block_invoke

int f0() {
  return ^(){ return 0; }();
}

@interface I0
@property (assign) int p0;
@end

@implementation I0
@synthesize p0 = __sythesized_p0;
@end
