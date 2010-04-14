// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
// rdar: // 7860960

@interface I
{
  int window;
}
@property int window, noWarningNeeded;
@end

@implementation I

@synthesize window;

@dynamic noWarningNeeded;
@end
