// REQUIRES: shell, arm-registered-target



// RUN: ln -fs %clang %T/clang++
// RUN: ln -fs %clang %T/clang++3.5.0
// RUN: ln -fs %clang %T/clang++-3.5
// RUN: ln -fs %clang %T/clang++-tot
// RUN: ln -fs %clang %T/clang-c++
// RUN: ln -fs %clang %T/clang-g++
// RUN: ln -fs %clang %T/c++
// RUN: ln -fs %clang %T/foo-clang++
// RUN: ln -fs %clang %T/foo-clang++-3.5
// RUN: ln -fs %clang %T/foo-clang++3.5
// RUN: %T/clang++          -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %T/clang++3.5.0     -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %T/clang++-3.5      -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %T/clang++-tot      -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %T/clang-c++        -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %T/clang-g++        -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %T/c++              -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %T/foo-clang++      -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %T/foo-clang++-3.5  -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// RUN: %T/foo-clang++3.5   -### %s 2>&1 | FileCheck -check-prefix=CXXMODE %s
// CXXMODE: "-x" "c++"


// RUN: ln -fs %clang %T/clang-cl
// RUN: ln -fs %clang %T/cl
// RUN: ln -fs %clang %T/cl.exe
// RUN: ln -fs %clang %T/clang-cl3.5
// RUN: ln -fs %clang %T/clang-cl-3.5
// Note: use -- in front of the filename so it's not mistaken for an option on
// filesystems that use slashes for dir separators.
// RUN: %T/clang-cl         -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// RUN: %T/cl               -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// RUN: %T/cl.exe           -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// RUN: %T/clang-cl3.5      -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// RUN: %T/clang-cl-3.5     -### -- %s 2>&1 | FileCheck -check-prefix=CLMODE %s
// CLMODE: "-fdiagnostics-format" "msvc"


// RUN: ln -fs %clang %T/clang-cpp
// RUN: ln -fs %clang %T/cpp
// RUN: %T/clang-cpp        -### %s 2>&1 | FileCheck -check-prefix=CPPMODE %s
// RUN: %T/cpp              -### %s 2>&1 | FileCheck -check-prefix=CPPMODE %s
// CPPMODE: "-E"


// RUN: ln -fs %clang %T/cl-clang
// RUN: %T/cl-clang        -### %s 2>&1 | FileCheck -check-prefix=CMODE %s
// CMODE: "-x" "c"
// CMODE-NOT: "-fdiagnostics-format" "msvc"


// RUN: ln -fs %clang %T/arm-linux-gnueabi-clang
// RUN: %T/arm-linux-gnueabi-clang -### %s 2>&1 | FileCheck -check-prefix=TARGET %s
// TARGET: Target: arm--linux-gnueabi
