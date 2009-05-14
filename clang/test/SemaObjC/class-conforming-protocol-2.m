// RUN: clang-cc  -fsyntax-only -verify %s

@protocol NSWindowDelegate @end

@interface NSWindow
- (void)setDelegate:(id <NSWindowDelegate>)anObject;
- (id <NSWindowDelegate>) delegate;
@end

@protocol IBStringsTableWindowDelegate <NSWindowDelegate>
@end

@interface IBStringsTableWindow : NSWindow {}
@end

@implementation IBStringsTableWindow
- (void)setDelegate:(id <IBStringsTableWindowDelegate>)delegate {
}
- (id <IBStringsTableWindowDelegate>)delegate {
        return 0;
}
@end
