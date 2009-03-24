// RUN: clang-cc -fsyntax-only -verify %s

@interface I {
  int Y;
}
@property int X;
@property int Y;
@property int Z;
@end

@implementation I
@dynamic X; // expected-note {{previous declaration is here}}
@dynamic X; // expected-error {{property 'X' is already implemented}}
@synthesize Y; // expected-note {{previous use is here}}
@synthesize Z=Y; // expected-error {{synthesized properties 'Z' and 'Y' both claim ivar 'Y'}}
@end
