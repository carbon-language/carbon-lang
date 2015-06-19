int *ptr; // expected-warning {{missing a nullability type specifier}}

#pragma clang assume_nonnull begin

extern void **blah; // expected-warning 2{{missing a nullability type specifier}}

#pragma clang assume_nonnull end

