// RUN: cd %S && %clang -### -g %s -c 2>&1 | FileCheck -check-prefix=CHECK-PWD %s
// CHECK-PWD: {{"-fdebug-compilation-dir" ".*Driver.*"}}

// RUN: env PWD=/foo %clang -### -g %s -c 2>&1 | FileCheck -check-prefix=CHECK-FOO %s
// CHECK-FOO: "-fdebug-compilation-dir" "/foo"

// This test uses grep instead of FileCheck so that we get %S -> dirname
// substitution.

// "PWD=/foo gcc" wouldn't necessarily work. You would need to pick a different
// path to the same directory (try a symlink).
