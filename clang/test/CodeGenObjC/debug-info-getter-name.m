// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -fno-dwarf2-cfi-asm -triple x86_64-apple-darwin10 -fexceptions -fobjc-exceptions -S -g %s -o - | FileCheck %s

//CHECK: "-[InstanceVariablesEverywhereButTheInterface someString]":
//CHECK: .quad	"-[InstanceVariablesEverywhereButTheInterface someString]"
//CHECK: .ascii	 "-[InstanceVariablesEverywhereButTheInterface someString]"
//CHECK: .asciz	 "-[InstanceVariablesEverywhereButTheInterface someString]"
//CHECK:  "-[InstanceVariablesEverywhereButTheInterface someString].eh":

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
@synthesize someString;
@synthesize someNumber;
- init
{
  return self;
}
@end

int main()
{
  return 0;
}
