struct S {
	int I;

	void incr() __attribute__((always_inline)) { I++; }
	void incr(int Add) __attribute__((always_inline)) { I += Add; }

	typedef int SInt;

	struct Nested {
		double D;

		template<typename T> void init(T Val) { D = double(Val); }
	};

	Nested D;

public:
	int foo() { return I; }
};

typedef S AliasForS;

namespace N {
class C {
	AliasForS S;
};
}

namespace N {
namespace N {
class C {
        int S;
};
}
}

namespace {
	class AnonC {
	};
}

union U {
	class C {} C;
	struct S {} S;
};

inline int func() {
	struct CInsideFunc { int i; };
	auto functor = []() { CInsideFunc dummy; return dummy.i; };
	return functor();
}
