// RUN: %clang -fverbose-asm -g -S %s -o - | grep DW_AT_name | count 42
// rdar://8493239

@class NSString;

@interface InstanceVariablesEverywhereButTheInterface 
@end

@interface InstanceVariablesEverywhereButTheInterface()
{
  NSString *_someString;
}

@property(readonly) NSString *someString;
@property(readonly) unsigned long someNumber;
@end

@implementation InstanceVariablesEverywhereButTheInterface
{
  unsigned long _someNumber;
}
@synthesize someString = _someString, someNumber = _someNumber;
@end

@interface AutomaticSynthesis
{
  int real_ivar;
}
@property(copy) NSString *someString;
@property unsigned long someNumber;
@end

@implementation AutomaticSynthesis
@end
