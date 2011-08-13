// RUN: %clang -x objc-cpp-output -c %s -o /dev/null

// Should compile without errors
@protocol P
- (void)m;
@end
void f() {}
