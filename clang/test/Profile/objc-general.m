// Test instrumentation of general constructs in objective C.

// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name objc-general.m %s -o - -emit-llvm -fblocks -fprofile-instr-generate | FileCheck -check-prefix=PGOGEN %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name objc-general.m %s -o - -emit-llvm -fblocks -fprofile-instr-use=%S/Inputs/objc-general.profdata | FileCheck -check-prefix=PGOUSE %s

#ifdef HAVE_FOUNDATION

// Use this to build an instrumented version to regenerate the input file.
#import <Foundation/Foundation.h>

#else

// Minimal definitions to get this to compile without Foundation.h.

@protocol NSObject
@end

@interface NSObject <NSObject>
- (id)init;
+ (id)alloc;
@end

struct NSFastEnumerationState;
@interface NSArray : NSObject
- (unsigned long) countByEnumeratingWithState: (struct NSFastEnumerationState*) state
                  objects: (id*) buffer
                  count: (unsigned long) bufferSize;
+(NSArray*) arrayWithObjects: (id) first, ...;
@end;
#endif

// PGOGEN: @[[FRC:"__llvm_profile_counters_\+\[A foreach:\]"]] = internal global [2 x i64] zeroinitializer
// PGOGEN: @[[BLC:"__llvm_profile_counters___13\+\[A foreach:\]_block_invoke"]] = internal global [2 x i64] zeroinitializer
// PGOGEN: @[[MAC:__llvm_profile_counters_main]] = global [1 x i64] zeroinitializer

@interface A : NSObject
+ (void)foreach: (NSArray *)array;
@end

@implementation A
// PGOGEN: define {{.*}}+[A foreach:]
// PGOUSE: define {{.*}}+[A foreach:]
// PGOGEN: store {{.*}} @[[FRC]], i64 0, i64 0
+ (void)foreach: (NSArray *)array
{
  __block id result;
  // PGOGEN: store {{.*}} @[[FRC]], i64 0, i64 1
  // PGOUSE: br {{.*}} !prof ![[FR1:[0-9]+]]
  // PGOUSE: br {{.*}} !prof ![[FR2:[0-9]+]]
  for (id x in array) {
    // PGOGEN: define {{.*}}_block_invoke
    // PGOUSE: define {{.*}}_block_invoke
    // PGOGEN: store {{.*}} @[[BLC]], i64 0, i64 0
    ^{ static int init = 0;
      // PGOGEN: store {{.*}} @[[BLC]], i64 0, i64 1
      // PGOUSE: br {{.*}} !prof ![[BL1:[0-9]+]]
       if (init)
         result = x;
       init = 1; }();
  }
}
@end

// PGOUSE-DAG: ![[FR1]] = metadata !{metadata !"branch_weights", i32 2, i32 3}
// PGOUSE-DAG: ![[FR2]] = metadata !{metadata !"branch_weights", i32 3, i32 2}
// PGOUSE-DAG: ![[BL1]] = metadata !{metadata !"branch_weights", i32 2, i32 2}

int main(int argc, const char *argv[]) {
  A *a = [[A alloc] init];
  NSArray *array = [NSArray arrayWithObjects: @"0", @"1", (void*)0];
  [A foreach: array];
  return 0;
}
