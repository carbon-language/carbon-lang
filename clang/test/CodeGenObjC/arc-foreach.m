// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-nonfragile-abi -triple x86_64-apple-darwin -O0 -emit-llvm %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// rdar://9503326

typedef void (^dispatch_block_t)(void);

@class NSString;
extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
@class NSArray;

int main (int argc, const char * argv[])
{
    NSArray *array;
    for ( NSString *str in array) {
        dispatch_block_t blk = ^{
            NSLog(@"str in block: %@", str);
        };
        blk();
    }
    return 0;
}

// CHECK-LP64: define internal void @__main_block_invoke
// CHECK-LP64: [[BLOCK:%.*]] = bitcast i8* {{%.*}} to [[BLOCK_T:%.*]]*
// CHECK-LP64-NEXT: [[T0:%.*]] = getelementptr inbounds [[BLOCK_T]]* [[BLOCK]], i32 0, i32 5
// CHECK-LP64-NEXT: [[T2:%.*]] = load [[OPAQUE_T:%.*]]** [[T0]], align 8 
// CHECK-LP64-NEXT: call void ([[OPAQUE_T]]*, ...)* @NSLog
