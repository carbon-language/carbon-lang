#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	NSSize size = {10,10};
	NSImage *image = [[NSImage alloc] initWithSize:size];
    [pool release]; // Set breakpoint here.
    return 0;
}

