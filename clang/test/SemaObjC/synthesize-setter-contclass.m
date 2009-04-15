// RUN: clang-cc  -fsyntax-only -verify %s

@interface TestClass 
{
 int _isItIsOrIsItAint;
}
@property (readonly) int itIsOrItAint;
-(void) doSomething;
@end

@interface TestClass()
@property (readwrite) int itIsOrItAint;
@end

@implementation TestClass
@synthesize itIsOrItAint = _isItIsOrIsItAint;

-(void) doSomething
{
  int i = [self itIsOrItAint];

 [self setItIsOrItAint:(int)1];
}
@end
