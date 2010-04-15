// RUN: %clang_cc1 -fblocks %s -emit-llvm -o %t

extern "C" int printf(const char*, ...);

template<typename T> class range {
public:
T _i;
        range(T i) {_i = i;};
        T get() {return _i;};
};

int main() {

        // works
        void (^bl)(range<int> ) = ^(range<int> i){printf("Hello Blocks %d\n", i.get()); };

        //crashes in godegen?
        void (^bl2)(range<int>& ) = ^(range<int>& i){printf("Hello Blocks %d\n", i.get()); };
        return 0;
}

