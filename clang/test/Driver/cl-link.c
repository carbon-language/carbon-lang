// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by -- or bound to another option, otherwise it may
// be interpreted as a command-line option, e.g. on Mac where %s is commonly
// under /Users.

// RUN: %clang_cl /Tc%s -### /link foo bar baz 2>&1 | FileCheck %s
// CHECK: link.exe
// CHECK: "foo"
// CHECK: "bar"
// CHECK: "baz"
