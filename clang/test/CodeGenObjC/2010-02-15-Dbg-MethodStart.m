// RUN: %clang_cc1 -x objective-c -emit-llvm -g < %s | FileCheck "%s"
// Test to check that subprogram start location.

@interface Foo
-(int) barMethod;
@end

@implementation Foo
-(int) barMethod {
  // CHECK: !MDSubprogram({{.*}}line: [[@LINE-1]]
  int i = 0;
  int j = 1;
  int k = 1;
  return i + j + k;
}
@end
