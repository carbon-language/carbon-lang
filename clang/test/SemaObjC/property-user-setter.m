// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface I0 
@property(readonly) int x;
@property(readonly) int y;
@property(readonly) int z;
-(void) setY: (int) y0;
@end

@interface I0 (Cat0)
-(void) setX: (int) a0;
@end

@implementation I0
@dynamic x;
@dynamic y;
@dynamic z;
-(void) setY: (int) y0{}

-(void) im0 {
  self.x = 0;
  self.y = 2;
  self.z = 2; // expected-error {{assigning to property with 'readonly' attribute not allowed}}
}
@end

// Test when property is 'readonly' but it has a setter in
// its implementation only.
@interface I1  {
}
@property(readonly) int identifier;
@end


@implementation I1
@dynamic identifier;
- (void)setIdentifier:(int)ident {}

- (id)initWithIdentifier:(int)Arg {
    self.identifier = 0;
}

@end


// Also in a category implementation
@interface I1(CAT)  
@property(readonly) int rprop;
@end


@implementation I1(CAT)
@dynamic rprop;
- (void)setRprop:(int)ident {}

- (id)initWithIdentifier:(int)Arg {
    self.rprop = 0;
}

@end

static int g_val;

@interface Root 
+ alloc;
- init;
@end

@interface Subclass : Root
{
    int setterOnly;
}
- (void) setSetterOnly:(int)value;	// expected-note {{or because setter is declared here, but no getter method 'setterOnly' is found}}
@end

@implementation Subclass
- (void) setSetterOnly:(int)value {
    setterOnly = value;
    g_val = setterOnly;
}
@end

@interface C {}
// - (int)Foo;
- (void)setFoo:(int)value;	// expected-note 2 {{or because setter is declared here, but no getter method 'Foo' is found}}
@end

void g(int);

void f(C *c) {
    c.Foo = 17; // expected-error {{property 'Foo' not found on object of type 'C *'}}
    g(c.Foo); // expected-error {{property 'Foo' not found on object of type 'C *'}}
}


void abort(void);
int main (void) {
    Subclass *x = [[Subclass alloc] init];

    x.setterOnly = 4;  // expected-error {{property 'setterOnly' not found on object of type 'Subclass *'}}
    if (g_val != 4)
      abort ();
    return 0;
}
