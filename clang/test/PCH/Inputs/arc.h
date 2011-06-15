// Header for Objective-C ARC-related PCH tests

typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;

CFTypeRef CFCreateSomething();
CFStringRef CFCreateString();
CFTypeRef CFGetSomething();
CFStringRef CFGetString();

@interface NSString
@end

id CreateSomething();
NSString *CreateNSString();

typedef int array0[sizeof((__bridge id)CFCreateSomething())];
typedef int array1[sizeof((__bridge CFTypeRef)CreateSomething())];


