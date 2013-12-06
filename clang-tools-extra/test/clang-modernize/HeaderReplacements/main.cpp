// The following block tests the following:
//   - Only 1 file is generated per translation unit
//   - Replacements are written in YAML that matches the expected YAML file
// The test is run in %T/SerializeTest so it's easy to create a clean test
// directory.
//
// RUN: rm -rf %T/SerializeTest
// RUN: mkdir -p %T/SerializeTest
// RUN: cp %S/main.cpp %S/common.cpp %S/common.h %T/SerializeTest
// RUN: clang-modernize -loop-convert -serialize-replacements -serialize-dir=%T/SerializeTest -include=%T/SerializeTest %T/SerializeTest/main.cpp %T/SerializeTest/common.cpp --
// Check that only 1 file is generated per translation unit
// RUN: ls -1 %T/SerializeTest | FileCheck %s --check-prefix=MAIN_CPP
// RUN: ls -1 %T/SerializeTest | FileCheck %s --check-prefix=COMMON_CPP
// We need to put the build path to the expected YAML file to diff against the generated one.
// RUN: sed -e 's#$(path)#%/T/SerializeTest#g' -e "s#[^[:space:]]'[^[:space:]]#''#g" -e "s#'\([-a-zA-Z0-9_/^., \t]*\)'#\1#g" %S/main_expected.yaml > %T/SerializeTest/main_expected.yaml
// RUN: sed -i -e 's#\\#/#g' %T/SerializeTest/main.cpp_*.yaml
// RUN: diff -b %T/SerializeTest/main_expected.yaml %T/SerializeTest/main.cpp_*.yaml
// RUN: sed -e 's#$(path)#%/T/SerializeTest#g' -e "s#[^[:space:]]'[^[:space:]]#''#g" -e "s#'\([-a-zA-Z0-9_/^., \t]*\)'#\1#g" %S/common_expected.yaml > %T/SerializeTest/common_expected.yaml
// RUN: sed -i -e 's#\\#/#g' %T/SerializeTest/common.cpp_*.yaml
// RUN: diff -b %T/SerializeTest/common_expected.yaml %T/SerializeTest/common.cpp_*.yaml
//
// The following are for FileCheck when used on output of 'ls'. See above.
// MAIN_CPP: {{^main.cpp_.*.yaml$}}
// MAIN_CPP-NOT: {{main.cpp_.*.yaml}}
//
// COMMON_CPP:     {{^common.cpp_.*.yaml$}}
// COMMON_CPP-NOT: {{common.cpp_.*.yaml}}

#include "common.h"

void test_header_replacement() {
  dostuff();
  func2();
}
