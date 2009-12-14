// RUN: clang -cc1 -triple=i686-apple-darwin9 -emit-llvm -o %t %s
// RUN: grep -e "\^{Innermost=CC}" %t | count 1
// RUN: grep -e "{Derived=#ib32b8b3b8sb16b8b8b2b8ccb6}" %t | count 1
// RUN: grep -e "{B1=#@c}" %t | count 1
// RUN: grep -e "v12@0:4\[3\[4@]]8" %t | count 1
// RUN: grep -e "r\^{S=i}" %t | count 1
// RUN: grep -e "\^{Object=#}" %t | count 1

@class Int1;

struct Innermost {
  unsigned char a, b;
};

@interface Int1 {
  signed char a, b;
  struct Innermost *innermost;
}
@end

@implementation Int1
@end

@interface Base
{
    struct objc_class *isa;
    int full;
    int full2: 32;
    int _refs: 8;
    int field2: 3;
    unsigned f3: 8;
    short cc;
    unsigned g: 16;
    int r2: 8;
    int r3: 8;
    int r4: 2;
    int r5: 8;
    char c;
}
@end

@interface Derived: Base
{
    char d;
    int _field3: 6;
}
@end

@implementation Base
@end

@implementation Derived
@end

@interface B1 
{
    struct objc_class *isa;
    Int1 *sBase;
    char c;
}
@end

@implementation B1
@end

@interface Test 
{
	int ivar;
         __attribute__((objc_gc(weak))) SEL selector;
}
-(void) test3: (Test*  [3] [4])b ; 
- (SEL**) meth : (SEL) arg : (SEL*****) arg1 : (SEL*)arg2 : (SEL**) arg3;
@end

@implementation Test
-(void) test3: (Test* [3] [4])b {}
- (SEL**) meth : (SEL) arg : (SEL*****) arg1 : (SEL*)arg2 : (SEL**) arg3 {}
@end

struct S { int iS; };

@interface Object
{
 Class isa;
}
@end
typedef Object MyObj;

int main()
{
	const char *en = @encode(Derived);
	const char *eb = @encode(B1);
        const char *es = @encode(const struct S *);
        const char *ec = @encode(const struct S);
        const char *ee = @encode(MyObj *const);
}

