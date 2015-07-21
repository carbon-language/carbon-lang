#include "odr-types.h"

double baz() {
	S::Nested d;
	d.init(0.0);
	return d.D;
}
