// RUN: clang-cc -fobjc-nonfragile-abi -fvisibility=hidden -S -o - %s | grep -e "private_extern _OBJC_IVAR_"
@interface I
{
	int P;
}

@property int P;
@end

@implementation I
@synthesize P;
@end

