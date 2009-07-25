// RUN: clang-cc  -g -S -o %t %s &&
// RUN: not grep 001 %t

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
	
