// RUN: %clang_cc1  -g -S -o %t %s
// FIXME: Reenable this test once this check is less picky.
// RUN: not grep 001 %t
//
// REQUIRES: disabled

@interface F 
-(int) bar;
@end

@implementation F
-(int) bar {
	return 42;
}
@end

extern int f(F *fn) {
	return [fn bar];
}
	
