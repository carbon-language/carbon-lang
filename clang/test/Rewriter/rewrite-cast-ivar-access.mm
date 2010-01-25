// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-rw.cpp %s
// radar 7575882

@interface F {
  int supervar;
}
@end

@interface G : F {
@public
  int ivar;
}
@end

@implementation G
- (void)foo:(F *)arg {
        int q = arg->supervar;
        int v = ((G *)arg)->ivar;
}
@end

// CHECK-LP: ((struct G_IMPL *)arg)->ivar

