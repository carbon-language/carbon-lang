// RUN: %clang_cc1 -emit-llvm -fblocks -g  -triple x86_64-apple-darwin10 -fobjc-dispatch-method=mixed  %s -o - | FileCheck %s

// rdar://problem/9279956
// Test that we generate the proper debug location for a captured self.
// The second half of this patch is in llvm/tests/DebugInfo/debug-info-blocks.ll

// CHECK: define {{.*}}_block_invoke
// CHECK: %[[BLOCK:.*]] = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg
// CHECK-NEXT: store <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %[[BLOCK]], <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %[[ALLOCA:.*]], align
// CHECK-NEXT: getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %[[BLOCK]], i32 0, i32 5
// CHECK-NEXT: call void @llvm.dbg.declare(metadata !{<{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %[[ALLOCA]]}, metadata ![[SELF:[0-9]+]])
// CHECK-NEXT: call void @llvm.dbg.declare(metadata !{%1** %d}, metadata ![[D:[0-9]+]])
// CHECK: ![[SELF]] = {{.*}} [ DW_TAG_auto_variable ] [self] [line 52]
// CHECK: ![[D]] = {{.*}} [d] [line 50]

typedef unsigned int NSUInteger;

@protocol NSObject
@end  
   
@interface NSObject <NSObject>
- (id)init;
+ (id)alloc;
@end 

@interface NSDictionary : NSObject 
- (NSUInteger)count;
@end    

@interface NSMutableDictionary : NSDictionary  
@end       

@interface A : NSObject {
@public
    int ivar;
}
@end

static void run(void (^block)(void))
{
    block();
}

@implementation A

- (id)init
{
    if ((self = [super init])) {
      run(^{
          NSMutableDictionary *d = [[NSMutableDictionary alloc] init]; 
          ivar = 42 + (int)[d count];
        });
    }
    return self;
}

@end

int main()
{
	A *a = [[A alloc] init];
	return 0;
}
