#import <Foundation/Foundation.h>

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    NSString *str = [NSString stringWithFormat:@"Hello from '%s'", argv[0]];      
    id str_id = str;
    SEL sel = @selector(length);
    BOOL responds = [str respondsToSelector:sel];
    printf("sizeof(id) = %zu\n", sizeof(id));
    printf("sizeof(Class) = %zu\n", sizeof(Class));
    printf("sizeof(SEL) = %zu\n", sizeof(SEL));
    [pool release];
    return 0;
}
