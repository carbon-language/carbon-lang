// RUN: clang-cc -fvisibility=hidden -triple x86_64-apple-darwin10  -S -o - %s | grep -e "private_extern _OBJC_IVAR_"
@interface I
{
	int P;
}

@property int P;
@end

@implementation I
@synthesize P;
@end

