// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify %s
// expected-no-diagnostics

class ButterFly {
private:
  ButterFly() { }
public:
	int triggerderef() {
		return 0;
	}
};
ButterFly *getInP();
class X{
	ButterFly *p;
	void setP(ButterFly *inP) {
		if(inP)
      ;
		p = inP;
	};
	void subtest1() {
		ButterFly *inP = getInP();
		setP(inP);
	}
	int subtest2() {
		int c = p->triggerderef(); // no-warning
		return c;
	}
	int test() {
		subtest1();
		return subtest2();
	}
};