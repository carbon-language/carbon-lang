// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface MyClass1 @end

@protocol p1,p2,p3;

@interface MyClass1 (Category1)  <p1> // expected-warning {{cannot find protocol definition for 'p1'}} expected-note {{previous definition is here}}
@end

@interface MyClass1 (Category1)  // expected-warning {{duplicate definition of category 'Category1' on interface 'MyClass1'}}
@end

@interface MyClass1 (Category3) 
@end

@interface MyClass1 (Category4) @end // expected-note {{previous definition is here}}
@interface MyClass1 (Category5) @end
@interface MyClass1 (Category6) @end
@interface MyClass1 (Category7) @end // expected-note {{previous definition is here}}
@interface MyClass1 (Category8) @end // expected-note {{previous definition is here}}


@interface MyClass1 (Category4) @end // expected-warning {{duplicate definition of category 'Category4' on interface 'MyClass1'}}
@interface MyClass1 (Category7) @end // expected-warning {{duplicate definition of category 'Category7' on interface 'MyClass1'}}
@interface MyClass1 (Category8) @end // expected-warning {{duplicate definition of category 'Category8' on interface 'MyClass1'}}


@protocol p3 @end

@interface MyClass1 (Category) <p2, p3> @end  // expected-warning {{cannot find protocol definition for 'p2'}}

@interface UnknownClass  (Category) @end // expected-error {{cannot find interface declaration for 'UnknownClass'}}

@class MyClass2;

@interface MyClass2  (Category) @end  // expected-error {{cannot find interface declaration for 'MyClass2'}}

@interface XCRemoteComputerManager
@end

@interface XCRemoteComputerManager() 
@end 

@interface XCRemoteComputerManager()
@end

@interface XCRemoteComputerManager(x) // expected-note {{previous definition is here}}
@end 

@interface XCRemoteComputerManager(x) // expected-warning {{duplicate definition of category 'x' on interface 'XCRemoteComputerManager'}}
@end

@implementation XCRemoteComputerManager
@end

@implementation XCRemoteComputerManager(x) // expected-note {{previous definition is here}}
@end

@implementation XCRemoteComputerManager(x) // expected-error {{reimplementation of category 'x' for class 'XCRemoteComputerManager'}}
@end

// <rdar://problem/7249233>

@protocol MultipleCat_P
-(void) im0; // expected-note {{method declared here}}
@end

@interface MultipleCat_I @end // expected-note {{required for direct or indirect protocol 'MultipleCat_P'}}

@interface MultipleCat_I()  @end

@interface MultipleCat_I() <MultipleCat_P>  @end

@implementation MultipleCat_I // expected-warning {{incomplete implementation}} \
                              // expected-warning {{method in protocol not implemented [-Wprotocol]}}
@end

// <rdar://problem/7680391> - Handle nameless categories with no name that refer
// to an undefined class
@interface RDar7680391 () @end // expected-error{{cannot find interface declaration}}
