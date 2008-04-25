// RUN: %llvmgcc -x objective-c -c %s -o /dev/null -m64
// rdar://5812818

@interface NSObject
@end

@interface Base:NSObject
@property int whatever;
@end

@interface Oops:Base
@end

@implementation Base
@synthesize whatever;
@end

@implementation Oops

-(void)whatthe {
 self.whatever=1;
}

@end

