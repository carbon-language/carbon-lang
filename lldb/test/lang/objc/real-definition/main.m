#include <stdio.h>
#include <stdint.h>
#import <Foundation/Foundation.h>
#import "Foo.h"

int main (int argc, char const *argv[])
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    Foo *foo = [[Foo alloc] init];
    NSLog (@"foo is %@", foo);
    [pool release];
    return 0;
}
