// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D_Bool=bool -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// radar 8143056

typedef unsigned NSPointerFunctionsOptions;
extern "C" id NSClassFromObject(id object);
void *sel_registerName(const char *);

struct NSSlice {
  int i1;
};

@interface NSConcretePointerFunctions {
  @public
    struct NSSlice slice;
}
- (bool)initializeSlice:(struct NSSlice *)slicep withOptions:(NSPointerFunctionsOptions)options;
@end

@implementation NSConcretePointerFunctions
- (id)initWithOptions:(NSPointerFunctionsOptions)options {
      if (![NSClassFromObject(self) initializeSlice:&slice withOptions:options])
        return 0;
      return self;
  }
- (bool)initializeSlice:(struct NSSlice *)slicep withOptions:(NSPointerFunctionsOptions)options {
    return 0;
  }
@end

