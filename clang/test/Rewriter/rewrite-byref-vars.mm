// RUN: %clang_cc1 -fms-extensions -rewrite-objc -x objective-c++ -fblocks -o - %s
// radar 7540194

extern "C" __declspec(dllexport) void BreakTheRewriter(int i) {
        __block int aBlockVariable = 0;
        void (^aBlock)(void) = ^ {
                aBlockVariable = 42;
        };
        aBlockVariable++;
	if (i) {
	  __block int bbBlockVariable = 0;
	  void (^aBlock)(void) = ^ {
                bbBlockVariable = 42;
          };
        }
}

__declspec(dllexport) extern "C" __declspec(dllexport) void XXXXBreakTheRewriter(void) {

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

// $CLANG -cc1 -fms-extensions -rewrite-objc -x objective-c++ -fblocks bug.mm
// g++ -c -D"__declspec(X)=" bug.cpp
