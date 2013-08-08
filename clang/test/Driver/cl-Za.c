// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// DEFAULT: "--dependent-lib=oldnames"

// RUN: %clang_cl /Za -### -- %s 2>&1 | FileCheck -check-prefix=Za %s
// Za-NOT: "--dependent-lib=oldnames"
