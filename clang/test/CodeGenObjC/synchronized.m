// RUN: clang -emit-llvm -triple=i686-apple-darwin8 -o %t %s
// RUNX: clang -emit-llvm -o %t %s

#include <stdio.h>

@interface MyClass
{
}
- (void)method;
@end

@implementation MyClass

- (void)method
{
	@synchronized(self)
	{
		NSLog(@"sync");
	}
}

@end

void foo(id a) {
  @synchronized(a) {
    printf("Swimming? No.");
    return;
  }
}



