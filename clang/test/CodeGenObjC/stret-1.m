// RUN: %clang_cc1 -fblocks -triple arm64-apple-darwin %s -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-ARM64
// rdar://12416433

struct stret { int x[100]; };
struct stret zero;
struct stret one = {{1}};

@interface Test  @end

@implementation Test
+(struct stret) method { return one; }
@end

int main(int argc, const char **argv)
{
    struct stret st2 = one;
    if (argc) st2 = [(id)(argc&~255) method];
}

// CHECK-ARM64: call void @llvm.memset.p0i8.i64(i8* [[T0:%.*]], i8 0, i64 400, i32 4, i1 false)
