// RUN: clang-cc %s -verify -Wunused -fsyntax-only
#include <stdio.h>

@interface Greeter
+ (void) hello;
@end

@implementation Greeter
+ (void) hello {
    fprintf(stdout, "Hello, World!\n");
}
@end




@interface NSObject @end
@interface NSString : NSObject 
- (int)length;
@end

void test() {
  // No unused warning: rdar://7126285
  @"pointless example call for test purposes".length;
}



int main (void) {
    [Greeter hello];
    return 0;
}

