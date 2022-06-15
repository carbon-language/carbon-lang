// RUN: %clang_cc1 -no-opaque-pointers -triple i386-apple-ios -fblocks -emit-llvm -o - %s -Wno-objc-root-class \
// RUN:   | FileCheck %s

void __assert_rtn(const char *, const char *, int, const char *);

@interface Test
- (void (^)(void)) mangle;
@end

@implementation Test
- (void (^)(void)) mangle {
  return ^() {
    void (^b)(void) = ^() {
      __assert_rtn(__func__, __FILE__, __LINE__, "mangle");
    };
  };
}
@end

// CHECK: @"__func__.__14-[Test mangle]_block_invoke_2" = private unnamed_addr constant [30 x i8] c"-[Test mangle]_block_invoke_2\00", align 1
// CHECK: @.str{{.*}} = private unnamed_addr constant {{.*}}, align 1
// CHECK: @.str[[STR1:.*]] = private unnamed_addr constant [7 x i8] c"mangle\00", align 1

// CHECK: define internal void @"__14-[Test mangle]_block_invoke"(i8* noundef %.block_descriptor)

// CHECK: define internal void @"__14-[Test mangle]_block_invoke_2"(i8* noundef %.block_descriptor){{.*}}{
// CHECK: call void @__assert_rtn(i8* noundef getelementptr inbounds ([30 x i8], [30 x i8]* @"__func__.__14-[Test mangle]_block_invoke_2", i32 0, i32 0), i8* noundef getelementptr inbounds {{.*}}, i32 noundef 14, i8* noundef getelementptr inbounds ([7 x i8], [7 x i8]* @.str[[STR1]], i32 0, i32 0))
// CHECK: }
