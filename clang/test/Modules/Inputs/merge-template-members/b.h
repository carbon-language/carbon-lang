#include "a.h"

// Add update record for definition of A<int> and constructors.
// We need an eagerly-emitted function here to get the problematic
// deserialization ordering.
void foobar() { N::A<int> x; }
