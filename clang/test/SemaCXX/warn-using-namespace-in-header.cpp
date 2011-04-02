// RUN: %clang_cc1 -fsyntax-only -Wheader-hygiene -verify %s

#include "warn-using-namespace-in-header.h"

namespace dont_warn {}
using namespace dont_warn;

// Warning is actually in the header but only the cpp file gets scanned.
// expected-warning {{using namespace directive in global context in header}}









// Warn inside linkage specs too.
// expected-warning {{using namespace directive in global context in header}}






// expected-warning {{using namespace directive in global context in header}}






















// expected-warning {{using namespace directive in global context in header}}

// |using namespace| through a macro shouldn't warn if the instantiation is in a
// cc file.
USING_MACRO
