// RUN: clang -cc1 -fsyntax-only -verify %s

@protocol P
  @property(readonly) int X;
@end

@protocol P1<P>
  @property (copy) id ID;
@end

@interface I
@end

@interface I (Cat) <P>
@property float X; // expected-warning {{property type 'float' is incompatible with type 'int' inherited from 'P'}}
@end

@interface I (Cat2) <P1>
@property (retain) id ID; // expected-warning {{property 'ID' 'copy' attribute does not match the property inherited from 'P1'}}
@end


@interface A 
@property(assign) int categoryProperty;
@end

// Don't issue warning on unimplemented setter/getter
// because property is @dynamic.
@implementation A 
@dynamic categoryProperty;
@end
