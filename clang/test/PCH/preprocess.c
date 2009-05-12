// RUN: clang-cc -emit-pch -o %t %S/preprocess.h &&
// RUN: clang-cc -include-pch %t -E -o - %s | grep -c "a_typedef" | count 1
#include "preprocess.h"

int a_value;
