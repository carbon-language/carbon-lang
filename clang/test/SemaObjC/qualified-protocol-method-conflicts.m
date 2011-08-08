// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://6191214

@protocol Xint
-(void) setX: (int) arg0; // expected-note 3 {{previous definition is here}}
+(int) C; // expected-note 3 {{previous definition is here}}
@end

@protocol Xfloat
-(void) setX: (float) arg0; // expected-warning 3 {{conflicting parameter types in declaration of 'setX:': 'int' vs 'float'}}
+(float) C; // expected-warning 3 {{conflicting return type in declaration of 'C': 'int' vs 'float'}}
@end

@interface A <Xint, Xfloat> // expected-note {{class is declared here}}
@end

@implementation A
-(void) setX: (int) arg0 { }
+(int) C {return 0; }
@end

@interface B <Xfloat, Xint> // expected-note {{class is declared here}}
@end

@implementation B 
-(void) setX: (float) arg0 { }
+ (float) C {return 0.0; }
@end

@protocol Xint_float<Xint, Xfloat>
@end

@interface C<Xint_float> // expected-note {{class is declared here}}
@end

@implementation C
-(void) setX: (int) arg0 { }
+ (int) C {return 0;}
@end
