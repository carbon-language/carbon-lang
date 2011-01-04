// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -S -g %s -o %t
// RUN: grep "\[InstanceVariablesEverywhereButTheInterface someString\]" %t | count 6

//rdar: //8498026

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

- init {
  return self;
}
@end

@interface AutomaticSynthesis 
{
  int real_ivar;
}
@property(copy) NSString *someString;
@property unsigned long someNumber;
@end

@implementation AutomaticSynthesis
- init
{
  return self;
}
@end

int main()
{
  return 0;
}
