#import <Foundation/Foundation.h>
#import "InternalDefiner.h"

@interface Container : NSObject {
@public
    InternalDefiner *_definer;
}

-(id)init;
@end

@implementation Container

-(id)init
{
    _definer = [InternalDefiner alloc];
    [_definer setBarTo:5];
    return self;
}

@end

int main(int argc, const char * argv[])
{

    @autoreleasepool {
        Container *j = [[Container alloc] init];

        printf("ivar value = %d", j->_definer->foo); // Set breakpoint 0 here.
    }   
    return 0;
}

