// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin10 -fexceptions -fobjc-exceptions -g %s -o - | FileCheck %s

// CHECK: !30 = metadata !{i32 720942, i32 0, metadata !6, metadata !"-[InstanceVariablesEverywhereButTheInterface someString]", metadata !"-[InstanceVariablesEverywhereButTheInterface someString]", metadata !"", metadata !6, i32 27, metadata !31, i1 true, i1 true, i32 0, i32 0, null, i32 320, i1 false, %1* (%0*, i8*)* @"\01-[InstanceVariablesEverywhereButTheInterface someString]", null, null, metadata !33} ; [ DW_TAG_subprogram ]

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
