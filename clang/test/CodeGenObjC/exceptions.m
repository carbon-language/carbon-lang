// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o %t %s
//
// <rdar://problem/7471679> [irgen] [eh] Exception code built with clang (x86_64) crashes

// Just check that we don't emit any dead blocks.
//
// RUN: grep 'No predecessors' %t | count 0

@interface NSArray @end
void f0() {
  @try {
    @try {
      @throw @"a";
    } @catch(NSArray *e) {
    }
  } @catch (id e) {
  }
}
