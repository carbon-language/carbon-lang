#import <Foundation/Foundation.h>
#import "InternalDefiner.h"

int main(int argc, const char * argv[])
{

    @autoreleasepool {
        
        InternalDefiner *i = [InternalDefiner alloc];
        
        [i setBarTo:3];
        
        printf("ivar value = %d", i->foo); // Set breakpoint 0 here.
        
    }
    return 0;
}

