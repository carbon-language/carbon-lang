// RUN: %llvmgcc -x objective-c -m64 -S %s -o /dev/null

@interface A
@end
@protocol P
@end
@interface B : A <P>
{
}
@end
@implementation B
- (void)test {
}
@end
