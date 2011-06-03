// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -masm-verbose -S -fblocks -g  -triple x86_64-apple-darwin10  %s -o - | FileCheck %s
extern void foo(void(^)(void));

// CHECK:	.ascii	 "__destroy_helper_block_" ## DW_AT_name

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

