// RUN: clang-cc -triple x86_64-apple-darwin9 -fobjc-gc -emit-llvm -o %t %s
@interface I
{
  struct {
    unsigned int d : 1;
  } bitfield;
}
@end

@implementation I
@end

