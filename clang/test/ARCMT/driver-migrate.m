// RUN: %clang -### -ccc-arcmt-migrate /foo/bar -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: "-arcmt-migrate" "-arcmt-migrate-directory" "/foo/bar"
