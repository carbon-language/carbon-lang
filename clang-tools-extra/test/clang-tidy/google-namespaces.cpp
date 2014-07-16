// RUN: clang-tidy %s -checks='-*,google-build-namespaces,google-build-using-namespace' -header-filter='.*' -- | FileCheck %s -implicit-check-not="{{warning|error}}:"
#include "Inputs/google-namespaces.h"
// CHECK: warning: do not use unnamed namespaces in header files.

using namespace spaaaace;
// CHECK: :[[@LINE-1]]:1: warning: do not use namespace using-directives. Use using-declarations instead.

using spaaaace::core; // no-warning
