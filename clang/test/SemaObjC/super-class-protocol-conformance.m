// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: // 7884086

@interface NSObject @end

@protocol TopProtocol
  @property (readonly) id myString; // expected-warning {{property 'myString' requires method 'myString' to be defined}}
@end

@protocol SubProtocol <TopProtocol>
@end

@interface TopClass : NSObject <TopProtocol> {}
@end

@interface SubClass : TopClass <SubProtocol> {}
@end

@interface SubClass1 : TopClass {}	
@end

@implementation SubClass1 @end // Test1 - No Warning

@implementation TopClass  // expected-note {{implementation is here}}
@end

@implementation SubClass // Test3 - No Warning 
@end

@interface SubClass2  : TopClass<TopProtocol> 
@end

@implementation SubClass2 @end // Test 4 - No Warning

@interface SubClass3 : TopClass<SubProtocol> @end
@implementation SubClass3 @end	// Test 5 - No Warning 

@interface SubClass4 : SubClass3 @end
@implementation SubClass4 @end	// Test 5 - No Warning

@protocol NewProtocol
  @property (readonly) id myNewString; // expected-warning {{property 'myNewString' requires method 'myNewString' to be defined}}
@end

@interface SubClass5 : SubClass4 <NewProtocol> @end
@implementation SubClass5 @end   // expected-note {{implementation is here}}

