// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://12958878

@interface NSObject @end

@protocol DVTInvalidation
- (void)invalidate;  // expected-note {{method 'invalidate' declared here}}
@property int Prop; // expected-note {{property declared here}}
@end



@protocol DVTInvalidation;

@interface IBImageCatalogDocument : NSObject <DVTInvalidation> // expected-note {{required for direct or indirect protocol 'DVTInvalidation'}}
@end

@implementation IBImageCatalogDocument // expected-warning {{auto property synthesis will not synthesize property declared in a protocol}} \ 
				       // expected-warning {{method 'invalidate' in protocol not implemented}}
@end
