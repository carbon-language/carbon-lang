// RUN: clang -cc1 -fsyntax-only -verify %s

typedef struct CGColor * __attribute__ ((NSObject)) CGColorRef;
static int count;
static CGColorRef tmp = 0;

typedef struct S1  __attribute__ ((NSObject)) CGColorRef1; // expected-error {{__attribute ((NSObject)) is for pointer types only}}
typedef void *  __attribute__ ((NSObject)) CGColorRef2; // expected-error {{__attribute ((NSObject)) is for pointer types only}}

@interface HandTested {
@public
    CGColorRef x;
}
@property(copy) CGColorRef x;
@end

void setProperty(id self, id value)  {
  ((HandTested *)self)->x = value;
}

id getProperty(id self) {
     return (id)((HandTested *)self)->x;
}

@implementation HandTested
@synthesize x=x;
@end

int main(int argc, char *argv[]) {
    HandTested *to;
    to.x = tmp;  // setter
    if (tmp != to.x)
      to.x = tmp;
    return 0;
}

