#import "InternalDefiner.h"

@interface InternalDefiner () {
    int bar;
}

@end

@implementation InternalDefiner

-(int)setBarTo:(int)newBar
{
    int oldBar = bar;
    bar = newBar;
    return oldBar;
}

@end
