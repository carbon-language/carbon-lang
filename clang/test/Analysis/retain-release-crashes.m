// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,osx.cocoa.RetainCount,osx.cocoa.SelfInit,debug.ExprInspection -verify -w %s

// This file contains crash regression tests; please do not remove any checkers
// from the RUN line because they may have been necessary to produce the crash.
// (Adding checkers should be fine.)

void clang_analyzer_eval(int);

@interface NSObject
- (id)init;
@end

@interface Foo : NSObject
@end

void touch(id, SEL);
id getObject();
int getInt();


@implementation Foo
// Bizarre crash related to the ExprEngine reaching a previously-seen
// ExplodedNode /during/ the processing of a message. Removing any
// parts of this test case seem not to trigger the crash any longer.
// <rdar://problem/12243648>
- (id)init {
  // Step 0: properly call the superclass's initializer
  self = [super init];
  if (!self) return self;

  // Step 1: Perturb the state with a new conjured symbol.
  int value = getInt();

  // Step 2: Loop. Some loops seem to trigger this, some don't.
  // The original used a for-in loop.
  while (--value) {
    // Step 3: Make it impossible to retain-count 'self' by calling
    // a function that takes a "callback" (in this case, a selector).
    // Note that this does not trigger the crash if you use a message!
    touch(self, @selector(hi));
  }

  // Step 4: Use 'self', so that we know it's non-nil.
  [self bar];

  // Step 5: Once again, make it impossible to retain-count 'self'...
  // ...while letting ObjCSelfInitChecker mark this as an interesting
  // message, since 'self' is an argument...
  // ...but this time do it in such a way that we'll also assume that
  // 'other' is non-nil. Once we've made the latter assumption, we
  // should cache out.
  id other = getObject();
  [other use:self withSelector:@selector(hi)];

  // Step 6: Check that we did, in fact, keep the assumptions about 'self'
  // and 'other' being non-nil.
  clang_analyzer_eval(other != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(self != 0); // expected-warning{{TRUE}}

  return self;
}
@end
