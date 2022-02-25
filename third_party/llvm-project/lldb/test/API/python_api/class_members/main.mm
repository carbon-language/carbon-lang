#import <objc/NSObject.h>

class Base {
public:
    int foo(int x, int y) { return 1; }
    char bar(int x, char y) { return 2; }
    void dat() {}
    static int sfunc(char, int, float) { return 3; }
};

class Derived: public Base {
protected:
    int dImpl() { return 1; }
public:
    float baz(float b) { return b + 1.0; }
};

@interface Thingy: NSObject {
}
- (id)init;
- (id)fooWithBar: (int)bar andBaz:(id)baz;
@end

@implementation Thingy {
}
- (id)init {
    return (self = [super init]);
}
- (id)fooWithBar: (int)bar andBaz:(id)baz {
    return nil;
}
@end

int main() {
    Derived d;
    Thingy *thingy = [[Thingy alloc] init];
    return 0; // set breakpoint here
}
