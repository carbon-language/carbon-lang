#include "odr-types.h"

int bar() {
	S s;
	s.incr(42);
	return s.foo();
}
