@protocol P
  @property(readonly) int X;
@end

@protocol P1<P>
  @property (copy) id ID;
@end

@interface I
@end

@interface I (Cat) <P>
@property float X; // expected-warning {{property type 'float' does not match property type inherited from 'P'}}
@end

@interface I (Cat2) <P1>
@property (retain) id ID; // expected-warning {{property 'ID' 'copy' attribute does not match the property inherited from 'P1'}}
@end



