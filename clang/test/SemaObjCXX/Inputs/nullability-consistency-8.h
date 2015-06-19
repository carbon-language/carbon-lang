typedef int* __nonnull mynonnull;

__attribute__((objc_root_class))
@interface typedefClass
- (void) func1:(mynonnull)i;
@end

void func2(mynonnull i);

void func3(int *); // expected-warning{{pointer is missing a nullability type specifier}}

#define CF_RETURNS_NOT_RETAINED __attribute__((cf_returns_not_retained))
typedef void *CFTypeRef;
void cf1(CFTypeRef * p CF_RETURNS_NOT_RETAINED); // expected-warning {{pointer is missing a nullability type specifier}}

void cf2(CFTypeRef * __nullable p CF_RETURNS_NOT_RETAINED);
void cf3(CFTypeRef * __nonnull p CF_RETURNS_NOT_RETAINED);

void cf4(CFTypeRef __nullable * __nullable p CF_RETURNS_NOT_RETAINED);
void cf5(CFTypeRef __nonnull * __nullable p CF_RETURNS_NOT_RETAINED);

void cf6(CFTypeRef * __nullable CF_RETURNS_NOT_RETAINED p);
void cf7(CF_RETURNS_NOT_RETAINED CFTypeRef * __nonnull p);

typedef CFTypeRef __nullable *CFTypeRefPtr;
void cfp1(CFTypeRefPtr p CF_RETURNS_NOT_RETAINED); // expected-warning {{pointer is missing a nullability type specifier}}
void cfp2(CFTypeRefPtr __nonnull p CF_RETURNS_NOT_RETAINED);
