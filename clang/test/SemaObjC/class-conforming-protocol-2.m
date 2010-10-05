// RUN: %clang_cc1  -fsyntax-only -verify %s

@protocol NSWindowDelegate @end

@interface NSWindow
- (void)setDelegate:(id <NSWindowDelegate>)anObject; // expected-note {{previous definition is here}}
- (id <NSWindowDelegate>) delegate; // expected-note {{previous definition is here}}
@end

@protocol IBStringsTableWindowDelegate <NSWindowDelegate>
@end

@interface IBStringsTableWindow : NSWindow {}
@end

@implementation IBStringsTableWindow
- (void)setDelegate:(id <IBStringsTableWindowDelegate>)delegate { // expected-warning {{conflicting parameter types in implementation of 'setDelegate:'}}
}
- (id <IBStringsTableWindowDelegate>)delegate { // expected-warning {{conflicting return type in implementation of 'delegate':}}
        return 0;
}
@end
