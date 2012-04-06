// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

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

// rdar://8703553
@interface IDEPathCell 
{
@private
    id _gradientStyle;
}

@property (readwrite, assign, nonatomic) id gradientStyle;
@end

@implementation IDEPathCell

@synthesize gradientStyle = _gradientStyle;
- (void)setGradientStyle:(id)value { }

+ (void)_componentCellWithRepresentedObject {
    self.gradientStyle; // expected-error {{property 'gradientStyle' not found on object of type 'Class'}}
}
@end
