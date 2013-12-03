// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s

#define NS_DESIGNATED_INITIALIZER __attribute__((objc_designated_initializer))

void fnfoo(void) NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods}}

@protocol P1
-(id)init NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of interface declarations}}
@end

__attribute__((objc_root_class))
@interface I1
-(void)meth NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of the init family}}
-(id)init NS_DESIGNATED_INITIALIZER;
+(id)init NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of the init family}}
@end

@interface I1(cat)
-(id)init2 NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of interface declarations}}
@end

@interface I1()
-(id)init3 NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of interface declarations}}
@end

@implementation I1
-(void)meth {}
-(id)init NS_DESIGNATED_INITIALIZER { return 0; } // expected-error {{only applies to methods of interface declarations}}
+(id)init { return 0; }
-(id)init3 { return 0; } // expected-warning {{secondary initializer missing a 'self' call to another initializer}}
-(id)init4 NS_DESIGNATED_INITIALIZER { return 0; } // expected-error {{only applies to methods of interface declarations}} \
									 			   // expected-warning {{secondary initializer missing a 'self' call to another initializer}}
@end

__attribute__((objc_root_class))
@interface B1
-(id)initB1 NS_DESIGNATED_INITIALIZER; // expected-note 2 {{method marked as designated initializer of the class here}}
-(id)initB2;
-(id)initB3 NS_DESIGNATED_INITIALIZER; // expected-note {{method marked as designated initializer of the class here}}
@end

@implementation B1
-(id)initB1 { return 0; }
-(id)initB2 { return 0; } // expected-warning {{secondary initializer missing a 'self' call to another initializer}}
-(id)initB3 { return 0; }
@end

@interface S1 : B1
-(id)initS1 NS_DESIGNATED_INITIALIZER; // expected-note {{method marked as designated initializer of the class here}}
-(id)initS2 NS_DESIGNATED_INITIALIZER;
-(id)initS3 NS_DESIGNATED_INITIALIZER; // expected-note 2 {{method marked as designated initializer of the class here}}
-(id)initS4 NS_DESIGNATED_INITIALIZER; // expected-note 2 {{method marked as designated initializer of the class here}}
-(id)initB1;
@end

@implementation S1
-(id)initS1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return 0;
}
-(id)initS2 {
  return [super initB1];
}
-(id)initS3 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return [super initB2]; // expected-warning {{designated initializer invoked a non-designated initializer}}
}
-(id)initS4 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return [self initB1]; // expected-warning {{designated initializer should only invoke a designated initializer on 'super'}}
}
-(id)initB1 {
  return [self initS1];
}
-(id)initB3 {
  return [self initS1];
}
@end

@interface S2 : B1
-(id)initB1;
@end

@interface SS2 : S2
-(id)initSS1 NS_DESIGNATED_INITIALIZER;
@end

@implementation SS2
-(id)initSS1 {
  return [super initB1];
}
@end

@interface S3 : B1
-(id)initS1 NS_DESIGNATED_INITIALIZER;
@end

@interface SS3 : S3
-(id)initSS1 NS_DESIGNATED_INITIALIZER; // expected-note 2 {{method marked as designated initializer of the class here}}
@end

@implementation SS3
-(id)initSS1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return [super initB1]; // expected-warning {{designated initializer invoked a non-designated initializer}}
}
@end

@interface S4 : B1
-(id)initB1;
-(id)initB3;
@end

@implementation S4
-(id)initB1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return 0;
}
-(id)initB3 {
  return [super initB3];
}
@end

@interface S5 : B1
-(void)meth;
@end

@implementation S5
-(id)initB1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return 0;
}
-(id)initB3 {
  [self initB1]; // expected-warning {{designated initializer should only invoke a designated initializer on 'super'}}
  S5 *s;
  [s initB1];
  [self meth];
  void (^blk)(void) = ^{
    [self initB1];
  };
  return [super initB3];
}
-(void)meth {}
-(id)initS1 {
  return 0;
}
-(id)initS2 {
  return [super initB1];
}
@end

@interface S6 : B1
-(id)initS1 NS_DESIGNATED_INITIALIZER;
-(id)initS2;
-(id)initS3;
-(id)initS4;
@end

@implementation S6
-(id)initS1 {
  return [super initB1];
}
-(id)initS2 { // expected-warning {{secondary initializer missing a 'self' call to another initializer}}
  return [super initB1]; // expected-warning {{secondary initializer should not invoke an initializer on 'super'}}
}
-(id)initS3 {
  return [self initB1];
}
-(id)initS4 {
  return [self initS1];
}
-(id)initS5 {
  [super initB1]; // expected-warning {{secondary initializer should not invoke an initializer on 'super'}}
  void (^blk)(void) = ^{
    [super initB1];
  };
  return [self initS1];
}
-(id)initS6 { // expected-warning {{secondary initializer missing a 'self' call to another initializer}}
  S6 *s;
  return [s initS1];
}
@end
