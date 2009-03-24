// RUN: clang-cc %s -verify -fsyntax-only
#include <stdio.h>

@interface Greeter
+ (void) hello;
@end

@implementation Greeter
+ (void) hello {
    fprintf(stdout, "Hello, World!\n");
}
@end

int main (void) {
    [Greeter hello];
    return 0;
}

