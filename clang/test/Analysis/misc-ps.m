// RUN: clang -checker-cfref --verify %s

// Reduced test case from crash in <rdar://problem/6253157>
@class NSObject;
@interface A @end
@implementation A
- (void)foo:(void (^)(NSObject *x))block {
  if (!((block != ((void *)0)))) {}
}
@end

