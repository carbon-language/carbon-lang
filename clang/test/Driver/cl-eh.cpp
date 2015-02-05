// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /c /EHsc -### -- %s 2>&1 | FileCheck -check-prefix=EHsc %s
// EHsc: "-fcxx-exceptions"
// EHsc: "-fexceptions"

// RUN: %clang_cl /c /EHs-c- -### -- %s 2>&1 | FileCheck -check-prefix=EHs_c_ %s
// EHs_c_-NOT: "-fcxx-exceptions"
// EHs_c_-NOT: "-fexceptions"

// RUN: %clang_cl /c /EHs- /EHc- -### -- %s 2>&1 | FileCheck -check-prefix=EHs_EHc_ %s
// EHs_EHc_-NOT: "-fcxx-exceptions"
// EHs_EHc_-NOT: "-fexceptions"

// RUN: %clang_cl /c /EHs- /EHs -### -- %s 2>&1 | FileCheck -check-prefix=EHs_EHs %s
// EHs_EHs: "-fcxx-exceptions"
// EHs_EHs: "-fexceptions"

// RUN: %clang_cl /c /EHs- /EHsa -### -- %s 2>&1 | FileCheck -check-prefix=EHs_EHa %s
// EHs_EHa: "-fcxx-exceptions"
// EHs_EHa: "-fexceptions"

// RUN: %clang_cl /c /EHinvalid -### -- %s 2>&1 | FileCheck -check-prefix=EHinvalid %s
// EHinvalid: error: invalid value 'invalid' in '/EH'
// EHinvalid-NOT: error:
