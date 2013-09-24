// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /fallback /Dfoo=bar /Ubaz /Ifoo /O0 /Ox /GR /GR- /LD /LDd \
// RUN:     /MD /MDd /MTd /MT -### -- %s 2>&1 | FileCheck %s
// CHECK: "-fdiagnostics-format" "msvc-fallback"
// CHECK: ||
// CHECK: cl.exe
// CHECK: "/c"
// CHECK: "/W0"
// CHECK: "-D" "foo=bar"
// CHECK: "-U" "baz"
// CHECK: "-I" "foo"
// CHECK: "-O3"
// CHECK: "/GR-"
// CHECK: "/LD"
// CHECK: "/LDd"
// CHECK: "/MT"
// CHECK: "/Tc" "{{.*cl-fallback.c}}"
// CHECK: "/Fo{{.*cl-fallback.*.obj}}"
