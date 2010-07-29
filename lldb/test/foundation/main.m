#import <Foundation/Foundation.h>

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    NSString *str = [NSString stringWithFormat:@"Hello from '%s'", argv[0]];
    [pool release];
    return 0;
}
