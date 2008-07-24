// RUN: clang -warn-objc-missing-dealloc '-DIBOutlet=__attribute__((iboutlet))' %s --verify

#ifndef IBOutlet
#define IBOutlet
#endif

@class NSWindow;

@interface NSObject {}
- (void)dealloc;
@end

@interface A : NSObject {
IBOutlet NSWindow *window;
}
@end

@implementation A // no-warning
@end

