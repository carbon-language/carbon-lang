// RUN: %clang_cc1 -fblocks %s -emit-llvm -o %t

extern "C" int printf(const char*, ...);

template<typename T> class range {
public:
T _i;
        range(T i) {_i = i;};
        T get() {return _i;};
};

// rdar: // 7495203
class A {
    public:
	A() : field(10), d1(3.14) {}
	void F();
	void S() {
	  printf(" field = %d\n", field);
	  printf(" field = %f\n", d1);
	}
	int field;
	double d1;
};

void A::F()
    {
	__block A &tlc = *this;
	// crashed in code gen (radar 7495203)
        ^{ tlc.S(); }();
    }

int main() {

        // works
        void (^bl)(range<int> ) = ^(range<int> i){printf("Hello Blocks %d\n", i.get()); };

        //crashes in godegen?
        void (^bl2)(range<int>& ) = ^(range<int>& i){printf("Hello Blocks %d\n", i.get()); };

	A *a = new A;
	a->F();
        return 0;
}

// rdar://8382559
namespace radar8382559 {
  void func(bool& outHasProperty);

  void test() {
    __attribute__((__blocks__(byref))) bool hasProperty = false;
    func(hasProperty);
  }
}
