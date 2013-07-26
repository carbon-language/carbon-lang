// The following block tests the following:
//   - Only 1 file is generated per translation unit and header file
//   - Replacements are written in YAML that matches the expected YAML file
// RUN: rm -rf %t/Test
// RUN: mkdir -p %t/Test
// RUN: cp %S/main.cpp %S/common.cpp %S/common.h %t/Test
// RUN: cpp11-migrate -loop-convert -headers -include=%t/Test %t/Test/main.cpp %t/Test/common.cpp --
// Check that only 1 file is generated per translation unit and header file.
// RUN: ls %t/Test/main.cpp_common.h_*.yaml | wc -l | grep "^1$"
// RUN: ls %t/Test/common.cpp_common.h_*.yaml | wc -l | grep "^1$"
// We need to remove the path from FileName in the generated YAML file because it will have a path in the temp directory
// RUN: sed -i -e 's/^\(FileName:\).*[\/\\]\(.*\)"$/\1 "\2"/g' %t/Test/main.cpp_common.h_*.yaml
// RUN: sed -i -e 's/^\(FileName:\).*[\/\\]\(.*\)"$/\1 "\2"/g' %t/Test/common.cpp_common.h_*.yaml
// RUN: diff -b %S/common.h.yaml %t/Test/main.cpp_common.h_*.yaml
// RUN: diff -b %S/common.h.yaml %t/Test/common.cpp_common.h_*.yaml
//
// The following block tests the following:
//   - YAML files are written only when -headers is used
// RUN: rm -rf %t/Test
// RUN: mkdir -p %t/Test
// RUN: cp %S/main.cpp %S/common.cpp %S/common.h %t/Test
// RUN: cpp11-migrate -loop-convert -headers -include=%t/Test %t/Test/main.cpp --
// RUN: cpp11-migrate -loop-convert %t/Test/common.cpp --
// Check that only one YAML file is generated from main.cpp and common.h and not from common.cpp and common.h since -header is not specified
// RUN: ls %t/Test/main.cpp_common.h_*.yaml | wc -l | grep "^1$"
// RUN: ls %t/Test/common.cpp_common.h_*.yaml | wc -l | grep "^0$"
// We need to remove the path from FileName in the generated YAML file because it will have a path in the temp directory
// RUN: sed -i -e 's/^\(FileName:\).*[\/\\]\(.*\)"$/\1 "\2"/g' %t/Test/main.cpp_common.h_*.yaml
// RUN: diff -b %S/common.h.yaml %t/Test/main.cpp_common.h_*.yaml

#include "common.h"

void test_header_replacement() {
  dostuff();
  func2();
}
