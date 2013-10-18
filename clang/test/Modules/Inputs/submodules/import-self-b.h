@import import_self.c;
#include "import-self-d.h"

// FIXME: This should not work; names from 'a' should not be visible here.
MyTypeA import_self_test_a;

// FIXME: This should work but does not; names from 'b' are not actually visible here.
//MyTypeC import_self_test_c;

MyTypeD import_self_test_d;
