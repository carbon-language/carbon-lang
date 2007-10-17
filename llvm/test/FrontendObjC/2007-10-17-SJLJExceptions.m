// RUN: %llvmgcc -x objective-c -arch i386 -pipe -std=gnu99 -O2 -fexceptions -S -o - t.m | not grep Unwind_Resume

#import <Foundation/Foundation.h>

static NSMutableArray *anArray = nil;

CFArrayRef bork(void) {
    CFArrayRef result = NULL;
    NSAutoreleasePool *pool = [NSAutoreleasePool new];
    @try {
	result = CFRetain(anArray);
    } @catch(id any) {
	NSLog(@"Swallowed exception %@", any);
    }

    [pool release];
    return result;
}
