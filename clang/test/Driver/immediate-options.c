// RUN: %clang --help | FileCheck %s -check-prefix=HELP
// HELP: isystem
// HELP-NOT: ast-dump
// HELP-NOT: ccc-cxx

// RUN: %clang --help-hidden | FileCheck %s -check-prefix=HELP-HIDDEN
// HELP-HIDDEN: ccc-cxx

// RUN: %clang -dumpversion | FileCheck %s -check-prefix=DUMPVERSION
// DUMPVERSION: 4.2.1

// RUN: %clang -print-search-dirs | FileCheck %s -check-prefix=PRINT-SEARCH-DIRS
// PRINT-SEARCH-DIRS: programs: ={{.*}}
// PRINT-SEARCH-DIRS: libraries: ={{.*}}
