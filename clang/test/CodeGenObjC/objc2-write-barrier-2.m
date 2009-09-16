// RUN: clang-cc -fnext-runtime -fobjc-gc -fobjc-newgc-api -emit-llvm -o %t %s &&
// RUN: grep -F '@objc_assign_global' %t  | count 7 &&
// RUN: grep -F '@objc_assign_ivar' %t  | count 4 &&
// RUN: grep -F '@objc_assign_strongCast' %t  | count 4 &&
// RUN: true

extern id **somefunc(void);
extern id *somefunc2(void);


// Globals

id W, *X, **Y;

void func(id a, id *b, id **c) {
   static id w, *x, **y;
   W = a;  /* { dg-warning "global\\/static variable assignment" } */
   w = a;  /* { dg-warning "global\\/static variable assignment" } */
   X = b;  /* { dg-warning "global\\/static variable assignment" } */
   x = b;  /* { dg-warning "global\\/static variable assignment" } */
   Y = c;  /* { dg-warning "global\\/static variable assignment" } */
   y = c;  /* { dg-warning "global\\/static variable assignment" } */
}

// Instances

@interface something {
    id w, *x, **y;
}
@end

@implementation something
- (void)amethod {
    id badIdea = *somefunc2();
    w = badIdea;   /* { dg-warning "instance variable assignment" } */
    x = &badIdea;  /* { dg-warning "instance variable assignment" } */
    y = &x;        /* { dg-warning "instance variable assignment" } */
}
@end

typedef struct {
    int junk;
    id  alfred;
} AStruct;

void funct2(AStruct *aptr) {
    id **ppptr = somefunc();
    aptr->alfred = 0;  /* { dg-warning "strong\\-cast assignment" } */
    **ppptr = aptr->alfred;       /* { dg-warning "strong\\-cast assignment" } */
    *ppptr = somefunc2();         /* { dg-warning "strong\\-cast assignment" } */
}

