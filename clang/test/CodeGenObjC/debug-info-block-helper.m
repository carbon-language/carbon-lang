// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -emit-llvm -fblocks -g -triple x86_64-apple-darwin10 -fobjc-fragile-abi %s -o - | FileCheck %s
extern void foo(void(^)(void));

// CHECK: !40 = metadata !{i32 {{.*}}, i32 0, metadata !10, metadata !"__destroy_helper_block_", metadata !"__destroy_helper_block_", metadata !"", metadata !10, i32 24, metadata !41, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8*)* @__destroy_helper_block_, null, null, metadata !43} ; [ DW_TAG_subprogram ]

@interface NSObject {
  struct objc_object *isa;
}
@end

@interface A:NSObject @end
@implementation A
- (void) helper {
 int master = 0;
 __block int m2 = 0;
 __block int dbTransaction = 0;
 int (^x)(void) = ^(void) { (void) self; 
	(void) master; 
	(void) dbTransaction; 
	m2++;
	return m2;

	};
  master = x();
}
@end

void foo(void(^x)(void)) {}

