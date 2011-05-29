// RUN: %clang -O0 -g %s -c -o %t.o
// RUN: %clang %t.o -o %t.out -framework Foundation
// RUN: %test_debuginfo %s %t.out 
// XFAIL: *
// XTARGET: darwin
// Radar 9279956

// DEBUGGER: break 31
// DEBUGGER: r
// DEBUGGER: p m2
// DEBUGGER: p dbTransaction
// DEBUGGER: p master
// CHECK: $1 = 1
// CHECK: $2 = 0
// CHECK: $3 = 0

#include <Cocoa/Cocoa.h>

extern void foo(void(^)(void));

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

int main() {
	A *a = [A alloc];
	[a helper];
	return 0;
}
