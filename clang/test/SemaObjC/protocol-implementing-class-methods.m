// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://7020493

@protocol P1
@optional
- (int) PMeth;
@required
- (void) : (double) arg; // expected-note {{method declared here}}
@end

@interface NSImage <P1>
- (void) initialize; // expected-note {{method declared here}}
@end

@interface NSImage (AirPortUI)
- (void) initialize;
@end

@interface NSImage()
- (void) CEMeth; // expected-note {{method declared here}}
@end

@implementation NSImage (AirPortUI)
- (void) initialize {NSImage *p=0; [p initialize]; } // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
- (int) PMeth{ return 0; }
- (void) : (double) arg{}; // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
- (void) CEMeth {}; // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
@end
