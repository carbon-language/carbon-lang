// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -arch x86_64 %s > %t
// RUN: diff %t %s.result

#define nil ((void*) 0)

@interface Foo 
@property (retain) id x;
@property (retain) id y;
@property (retain) id w;
@property (retain) id z;
@end

@implementation Foo 
@synthesize x;
@synthesize y;
@synthesize w;
@synthesize z;

- (void) dealloc {
  self.x = 0;
  [self setY:nil];
  w = nil;
  self.z = nil;
}
@end
