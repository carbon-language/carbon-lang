// RUN: %clang_cc1 -fsyntax-only -fobjc-nonfragile-abi2 -verify %s

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
  id myString; // expected-note {{previously declared 'myString' here}}
}
@end

@interface SubClass : TopClass <TopProtocol> 
@end

@implementation SubClass @end // expected-error {{property 'myString' attempting to use ivar 'myString' declared in super class 'TopClass'}}

// rdar: // 7920807
@interface C @end
@interface C (Category)
@property int p; // expected-warning {{property 'p' requires method 'p' to be defined }} \
                 // expected-warning {{property 'p' requires method 'setP:' to be defined}}
@end
@implementation C (Category) // expected-note 2 {{implementation is here}}
@end

