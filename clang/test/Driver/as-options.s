// PR21000: Test that -I is passed to assembler. 
// RUN: %clang -### -c -no-integrated-as %s -Ifoo_dir 2>&1 | FileCheck --check-prefix=BARE_INCLUDE1 %s
// BARE_INCLUDE1: "-I" "foo_dir"

// RUN: %clang -### -c -no-integrated-as %s -I foo_dir 2>&1 | FileCheck --check-prefix=BARE_INCLUDE2 %s
// BARE_INCLUDE2: "-I" "foo_dir"

// RUN: %clang -### -c -integrated-as %s -Ifoo_dir 2>&1 | FileCheck --check-prefix=BARE_INT_INCLUDE1 %s
// BARE_INT_INCLUDE1: "-I" "foo_dir"

// RUN: %clang -### -c -integrated-as %s -I foo_dir 2>&1 | FileCheck --check-prefix=BARE_INT_INCLUDE2 %s
// BARE_INT_INCLUDE2: "-I" "foo_dir"
