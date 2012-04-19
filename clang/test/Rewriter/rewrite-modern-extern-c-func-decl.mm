// RUN: %clang_cc1 -fms-extensions -rewrite-objc -x objective-c++ -fblocks -o %t-rw.cpp %s
// RUN: %clang_cc1 -fsyntax-only -Werror -Wno-address-of-temporary -Wno-attributes -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://11131490

extern "C" __declspec(dllexport) void BreakTheRewriter(void) {
        __block int aBlockVariable = 0;
        void (^aBlock)(void) = ^ {
                aBlockVariable = 42;
        };
        aBlockVariable++;
        void (^bBlocks)(void) = ^ {
                aBlockVariable = 43;
        };
        void (^c)(void) = ^ {
                aBlockVariable = 44;
        };

}
__declspec(dllexport) extern "C" void AnotherBreakTheRewriter(int *p1, double d) {

        __block int bBlockVariable = 0;
        void (^aBlock)(void) = ^ {
                bBlockVariable = 42;
        };
        bBlockVariable++;
        void (^bBlocks)(void) = ^ {
                bBlockVariable = 43;
        };
        void (^c)(void) = ^ {
                bBlockVariable = 44;
        };

}

int

__declspec (dllexport)

main (int argc, char *argv[])
{
        __block int bBlockVariable = 0;
        void (^aBlock)(void) = ^ {
                bBlockVariable = 42;
        };
}

// rdar://11275241
static char stringtype;
char CFStringGetTypeID();
void x(void (^)());

static void initStatics(int arg, ...) {
    x(^{
        stringtype = CFStringGetTypeID();
    });
}
static void initStatics1(...) {
    x(^{
        stringtype = CFStringGetTypeID();
    });
}
static void initStatics2() {
    x(^{
        stringtype = CFStringGetTypeID();
    });
}

