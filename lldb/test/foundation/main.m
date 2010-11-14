#import <Foundation/Foundation.h>

@interface MyString : NSObject {
    NSString *str;
    NSDate *date;
}
- (id)initWithNSString:(NSString *)string;
@end

@implementation MyString
- (id)initWithNSString:(NSString *)string
{
    if (self = [super init])
    {
        str = [NSString stringWithString:string];
        date = [NSDate date];
    }
    return self;
}

- (void)dealloc
{
    [date release];
    [str release];
    [super dealloc];
}

- (NSString *)description
{
    return [str stringByAppendingFormat:@" with timestamp: %@", date];
}
@end

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    NSString *str = [NSString stringWithFormat:@"Hello from '%s'", argv[0]];
    NSLog(@"NSString instance: %@", str);

    MyString *my = [[MyString alloc] initWithNSString:str];
    NSLog(@"MyString instance: %@", [my description]);

    id str_id = str; // Set break point at this line.
    SEL sel = @selector(length);
    BOOL responds = [str respondsToSelector:sel];
    printf("sizeof(id) = %zu\n", sizeof(id));
    printf("sizeof(Class) = %zu\n", sizeof(Class));
    printf("sizeof(SEL) = %zu\n", sizeof(SEL));
    printf("[str length] = %zu\n", (size_t)[str length]);
    printf("str = '%s'\n", [str cStringUsingEncoding: [NSString defaultCStringEncoding]]);

    [pool release];
    return 0;
}
