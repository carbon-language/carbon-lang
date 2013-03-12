// RUN: %clang_cc1 -x objective-c -fsyntax-only -fobjc-default-synthesize-properties -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -fobjc-default-synthesize-properties -verify -Wno-objc-root-class %s

#if __has_attribute(objc_requires_property_definitions)
__attribute ((objc_requires_property_definitions)) 
#endif
@interface NoAuto // expected-note 2 {{class with specified objc_requires_property_definitions attribute is declared here}}
@property int NoAutoProp; // expected-note 2 {{property declared here}}
@end

@implementation NoAuto  // expected-warning {{property 'NoAutoProp' requires method 'NoAutoProp' to be defined}} \
                        // expected-warning {{property 'NoAutoProp' requires method 'setNoAutoProp:'}}
@end

__attribute ((objc_requires_property_definitions))  // redundant, just for testing
@interface Sub : NoAuto  // expected-note 3 {{class with specified objc_requires_property_definitions attribute is declared here}}
@property (copy) id SubProperty; // expected-note 2 {{property declared here}}
@end

@implementation Sub // expected-warning {{property 'SubProperty' requires method 'SubProperty' to be defined}} \
                    // expected-warning {{property 'SubProperty' requires method 'setSubProperty:' to be defined}}
@end

@interface Deep : Sub
@property (copy) id DeepProperty;
@property (copy) id DeepSynthProperty;
@property (copy) id DeepMustSynthProperty; // expected-note {{property declared here}}
@end

@implementation Deep // expected-warning {{property 'DeepMustSynthProperty' requires method 'setDeepMustSynthProperty:' to be defined}}
@dynamic DeepProperty;
@synthesize DeepSynthProperty;
- (id) DeepMustSynthProperty { return 0; }
@end

__attribute ((objc_requires_property_definitions)) 
@interface Deep(CAT)  // expected-error {{attributes may not be specified on a category}}
@end

__attribute ((objc_requires_property_definitions)) // expected-error {{objc_requires_property_definitions attribute may only be specified on a class}} 
@protocol P @end

// rdar://13388503
@interface NSObject @end
@protocol Foo
@property (readonly) char isFoo; // expected-note {{property declared here}}
@end

@interface Bar : NSObject <Foo>
@end

@implementation Bar
- (char)isFoo {
    return 0;
}
@end

@interface Baz : Bar
@end

@interface Baz ()
@property (readwrite) char isFoo; // expected-warning {{auto property synthesis will not synthesize property 'isFoo' because it is 'readwrite' but it will be synthesized 'readonly' via another property}}
@property char Property1; // expected-warning {{auto property synthesis will not synthesize property 'Property1' because it cannot share an ivar with another synthesized property}}
@property char Property2;
@end

@implementation Baz {
    char _isFoo;
}
@synthesize Property2 = Property1; // expected-note {{property synthesized here}}
@end
