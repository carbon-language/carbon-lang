#import <Foundation/Foundation.h>

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	NSDate *date1 = [NSDate date];
	CFGregorianDate cf_greg_date = CFAbsoluteTimeGetGregorianDate(CFDateGetAbsoluteTime((CFDateRef)date1), NULL);
	CFRange cf_range = {4,4};
// Set breakpoint here.
    [pool release];
    return 0;
}
