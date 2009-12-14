// RUN: clang -cc1 -fsyntax-only -verify %s

typedef struct objc_object {
  struct objc_class *isa;
} *id;

@interface NSObject {
  struct objc_class *isa;
}
@end
@interface Whatever : NSObject
+self;
@end

static void func() {
 
  id x;

  [(*x).isa self];
  [x->isa self];
  
  Whatever *y;

  // GCC allows this, with the following warning: 
  //   instance variable ‘isa’ is @protected; this will be a hard error in the future
  //
  // FIXME: see if we can avoid the 2 warnings that follow the error.
  [(*y).isa self]; // expected-error {{instance variable 'isa' is protected}} \
                      expected-warning{{receiver type 'struct objc_class *' is not 'id' or interface pointer, consider casting it to 'id'}} \
                      expected-warning{{method '-self' not found (return type defaults to 'id')}}
  [y->isa self]; // expected-error {{instance variable 'isa' is protected}} \
                    expected-warning{{receiver type 'struct objc_class *' is not 'id' or interface pointer, consider casting it to 'id'}} \
                    expected-warning{{method '-self' not found (return type defaults to 'id')}}
}
