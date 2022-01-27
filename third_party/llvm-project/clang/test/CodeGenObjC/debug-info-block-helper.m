// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -emit-llvm -fblocks -debug-info-kind=limited -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 %s -o - | FileCheck %s
extern void foo(void(^)(void));

// CHECK: !DISubprogram(linkageName: "__destroy_helper_block_8_32o40r48r"

@interface NSObject {
  struct objc_object *isa;
}
@end

@interface A:NSObject @end
@implementation A
- (void) helper {
 int controller = 0;
 __block int m2 = 0;
 __block int dbTransaction = 0;
 int (^x)(void) = ^(void) { (void) self; 
	(void) controller; 
	(void) dbTransaction; 
	m2++;
	return m2;

	};
  controller = x();
}
@end

void foo(void(^x)(void)) {}

