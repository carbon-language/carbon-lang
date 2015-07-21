#include "odr-types.h"

int bar() {
	S s;
	s.incr();
	return s.foo();
}
