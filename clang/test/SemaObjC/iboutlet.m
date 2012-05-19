// RUN: %clang_cc1 -fsyntax-only -fobjc-default-synthesize-properties  -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -fobjc-default-synthesize-properties  -verify %s
// rdar://11448209

@class NSView;

#define IBOutlet __attribute__((iboutlet))

@interface I
@property (getter = MyGetter, readonly, assign) IBOutlet NSView *myView; // expected-note {{property declared here}} \
							// expected-note {{readonly IBOutlet property should be changed to be readwrite}}
@end

@implementation I // expected-warning {{readonly IBOutlet property when auto-synthesized may not work correctly with 'nib' loader}}
@end
