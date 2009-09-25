// RUN: clang-cc -triple x86_64-apple-darwin9 -fobjc-gc -g -emit-llvm -o - %s &&
// RUN: clang-cc -triple i386-apple-darwin9 -fobjc-gc -g -emit-llvm -o - %s

// rdar://7252252
@interface Loop {
@public
  __weak Loop *_loop;
}
@end

@implementation Loop @end

void loop(Loop *L) {
  L->_loop = 0;
}
