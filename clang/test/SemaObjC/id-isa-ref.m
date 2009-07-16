// RUN: clang-cc -fsyntax-only -verify %s

typedef struct objc_object {
  struct objc_class *isa;
} *id;

@interface Whatever
+self;
@end

static void func() {
 
  id x;

  // FIXME: The following needs to compile without error. I will fix this tomorrow (7/15/09). Until I do, we will produce an error.
  [x->isa self]; // expected-error {{member reference base type 'id' is not a structure or union}}
}
