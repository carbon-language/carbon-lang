#import <Foundation/Foundation.h>


int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    static NSString *g_global_nsstr = @"Howdy";
    NSString *str1 = [NSString stringWithFormat:@"string %i", 1];
    NSString *str2 = [NSString stringWithFormat:@"string %i", 2];
    NSString *str3 = [NSString stringWithFormat:@"string %i", 3];
    NSArray *array = [NSArray arrayWithObjects: str1, str2, str3, nil];
    NSDictionary *dict = [NSDictionary dictionaryWithObjectsAndKeys:
                            str1, @"1", 
                            str2, @"2", 
                            str3, @"3", 
                            nil];

    id str_id = str1;
    SEL sel = @selector(length);
    [pool release];
    return 0;
}
