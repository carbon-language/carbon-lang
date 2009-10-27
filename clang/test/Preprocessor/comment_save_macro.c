// RUN: clang-cc -E -C %s | FileCheck -strict-whitespace %s &&
// CHECK: boo bork bar // zot

// RUN: clang-cc -E -CC %s | FileCheck -strict-whitespace %s &&
// CHECK: boo bork /* blah*/ bar // zot

// RUN: clang-cc -E %s | FileCheck -strict-whitespace %s
// CHECK: boo bork bar


#define FOO bork // blah
boo FOO bar // zot

