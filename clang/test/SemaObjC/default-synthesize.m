// RUN: %clang_cc1 -fsyntax-only -fobjc-nonfragile-abi -fobjc-default-synthesize-properties -verify %s

@interface NSString @end

@interface NSObject @end

@interface SynthItAll
@property int howMany;
@property (retain) NSString* what;
@end

@implementation SynthItAll
//@synthesize howMany, what;
@end


@interface SynthSetter : NSObject
@property (nonatomic) int howMany;  // REM: nonatomic to avoid warnings about only implementing one of the pair
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthSetter
//@synthesize howMany, what;

- (int) howMany {
    return self.howMany;
}
// - (void) setHowMany: (int) value

- (NSString*) what {
    return self.what;
}
// - (void) setWhat: (NSString*) value    
@end


@interface SynthGetter : NSObject
@property (nonatomic) int howMany;  // REM: nonatomic to avoid warnings about only implementing one of the pair
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthGetter
//@synthesize howMany, what;

// - (int) howMany
- (void) setHowMany: (int) value {
    self.howMany = value;
}

// - (NSString*) what
- (void) setWhat: (NSString*) value {
    if (self.what != value) {
    }
}
@end


@interface SynthNone : NSObject
@property int howMany;
@property (retain) NSString* what;
@end

@implementation SynthNone
//@synthesize howMany, what;  // REM: Redundant anyway

- (int) howMany {
    return self.howMany;
}
- (void) setHowMany: (int) value {
    self.howMany = value;
}

- (NSString*) what {
    return self.what;
}
- (void) setWhat: (NSString*) value {
    if (self.what != value) {
    }
}
@end

@protocol TopProtocol
  @property (readonly) id myString;
@end

@interface TopClass <TopProtocol> 
{
  id myString; 
}
@end

@interface SubClass : TopClass <TopProtocol> 
@end

@implementation SubClass @end 

// rdar://7920807
@interface C @end
@interface C (Category)
@property int p; // expected-warning {{property 'p' requires method 'p' to be defined }} \
                 // expected-warning {{property 'p' requires method 'setP:' to be defined}}
@end
@implementation C (Category) // expected-note 2 {{implementation is here}}
@end

// Don't complain if a property is already @synthesized by usr.
@interface D
{
}
@property int PROP;
@end

@implementation D
@synthesize PROP=IVAR;
- (int) Meth { return self.PROP; }
@end

