// RUN: %clang_cc1 -fsyntax-only -fblocks -Woverriding-method-mismatch -Wno-nullability-declspec %s -verify

@interface NSFoo
@end

// Nullability applies to all pointer types.
typedef NSFoo * __nonnull nonnull_NSFoo_ptr;
typedef id __nonnull nonnull_id;
typedef SEL __nonnull nonnull_SEL;

// Nullability can move into Objective-C pointer types.
typedef __nonnull NSFoo * nonnull_NSFoo_ptr_2;

// Conflicts from nullability moving into Objective-C pointer type.
typedef __nonnull NSFoo * __nullable conflict_NSFoo_ptr_2; // expected-error{{'__nonnull' cannot be applied to non-pointer type 'NSFoo'}}

void testBlocksPrinting(NSFoo * __nullable (^bp)(int)) {
  int *ip = bp; // expected-error{{'NSFoo * __nullable (^)(int)'}}
}
