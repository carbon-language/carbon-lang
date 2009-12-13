// RUN: clang-cc -fobjc-nonfragile-abi -emit-llvm -o - %s
// FIXME. Test is incomplete.

@protocol P 
@property int i;
@end

@protocol P1 
@property int i1;
@end

@protocol P2 < P1> 
@property int i2;
@end

@interface C1 { id isa; } @end

@interface C2 : C1 <P, P2> {
    int i;
}
@property int i2;
@end

@implementation C1
+(void)initialize { }
@end

@implementation C2
@synthesize i;
@synthesize i1;
@synthesize i2;
@end
