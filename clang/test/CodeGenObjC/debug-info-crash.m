// RUN: %clang_cc1 -triple i386-apple-darwin10 -fblocks -g -S %s -o -

// rdar://7556129
@implementation test
- (void)wait {
  ^{};
}
@end

