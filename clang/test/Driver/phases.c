// Basic compilation for various types of files.
// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-phases -x c %s -x objective-c %s -x c++ %s -x objective-c++ -x assembler %s -x assembler-with-cpp %s -x none %s 2> %t &&
// RUN: grep '0: input, ".*phases.c", c' %t &&
// RUN: grep -F '1: preprocessor, {0}, cpp-output' %t &&
// RUN: grep -F '2: compiler, {1}, assembler' %t &&
// RUN: grep -F '3: assembler, {2}, object' %t &&
// RUN: grep '4: input, ".*phases.c", objective-c' %t &&
// RUN: grep -F '5: preprocessor, {4}, objective-c-cpp-output' %t &&
// RUN: grep -F '6: compiler, {5}, assembler' %t &&
// RUN: grep -F '7: assembler, {6}, object' %t &&
// RUN: grep '8: input, ".*phases.c", c++' %t &&
// RUN: grep -F '9: preprocessor, {8}, c++-cpp-output' %t &&
// RUN: grep -F '10: compiler, {9}, assembler' %t &&
// RUN: grep -F '11: assembler, {10}, object' %t &&
// RUN: grep '12: input, ".*phases.c", assembler' %t &&
// RUN: grep -F '13: assembler, {12}, object' %t &&
// RUN: grep '14: input, ".*phases.c", assembler-with-cpp' %t &&
// RUN: grep -F '15: preprocessor, {14}, assembler' %t &&
// RUN: grep -F '16: assembler, {15}, object' %t &&
// RUN: grep '17: input, ".*phases.c", c' %t &&
// RUN: grep -F '18: preprocessor, {17}, cpp-output' %t &&
// RUN: grep -F '19: compiler, {18}, assembler' %t &&
// RUN: grep -F '20: assembler, {19}, object' %t &&
// RUN: grep -F '21: linker, {3, 7, 11, 13, 16, 20}, image' %t &&

// Universal linked image.
// RUN: clang -ccc-host-triple i386-apple-darwin9 -ccc-print-phases -x c %s -arch ppc -arch i386 2> %t &&
// RUN: grep '0: input, ".*phases.c", c' %t &&
// RUN: grep -F '1: preprocessor, {0}, cpp-output' %t &&
// RUN: grep -F '2: compiler, {1}, assembler' %t &&
// RUN: grep -F '3: assembler, {2}, object' %t &&
// RUN: grep -F '4: linker, {3}, image' %t &&
// RUN: grep -F '5: bind-arch, "ppc", {4}, image' %t &&
// RUN: grep -F '6: bind-arch, "i386", {4}, image' %t &&
// RUN: grep -F '7: lipo, {5, 6}, image' %t &&

// Universal object file.
// RUN: clang -ccc-host-triple i386-apple-darwin9 -ccc-print-phases -c -x c %s -arch ppc -arch i386 2> %t &&
// RUN: grep '0: input, ".*phases.c", c' %t &&
// RUN: grep -F '1: preprocessor, {0}, cpp-output' %t &&
// RUN: grep -F '2: compiler, {1}, assembler' %t &&
// RUN: grep -F '3: assembler, {2}, object' %t &&
// RUN: grep -F '4: bind-arch, "ppc", {3}, object' %t &&
// RUN: grep -F '5: bind-arch, "i386", {3}, object' %t &&
// RUN: grep -F '6: lipo, {4, 5}, object' %t &&

// Arch defaulting
// RUN: clang -ccc-host-triple i386-apple-darwin9 -ccc-print-phases -c -x assembler %s 2> %t &&
// RUN: grep -F '2: bind-arch, "i386", {1}, object' %t &&
// RUN: clang -ccc-host-triple i386-apple-darwin9 -ccc-print-phases -c -x assembler %s -m32 -m64 2> %t &&
// RUN: grep -F '2: bind-arch, "x86_64", {1}, object' %t &&
// RUN: clang -ccc-host-triple x86_64-apple-darwin9 -ccc-print-phases -c -x assembler %s 2> %t &&
// RUN: grep -F '2: bind-arch, "x86_64", {1}, object' %t &&
// RUN: clang -ccc-host-triple x86_64-apple-darwin9 -ccc-print-phases -c -x assembler %s -m64 -m32 2> %t &&
// RUN: grep -F '2: bind-arch, "i386", {1}, object' %t &&

// Analyzer
// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-phases --analyze %s 2> %t &&
// RUN: grep '0: input, ".*phases.c", c' %t &&
// RUN: grep -F '1: preprocessor, {0}, cpp-output' %t &&
// RUN: grep -F '2: analyzer, {1}, plist' %t &&

// Precompiler
// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-phases -x c-header %s 2> %t &&
// RUN: grep '0: input, ".*phases.c", c-header' %t &&
// RUN: grep -F '1: preprocessor, {0}, c-header-cpp-output' %t &&
// RUN: grep -F '2: precompiler, {1}, precompiled-header' %t &&

// Darwin overrides the handling for .s
// RUN: touch %t.s &&
// RUN: clang -ccc-host-triple i386-unknown-unknown -ccc-print-phases -c %t.s 2> %t &&
// RUN: grep '0: input, ".*\.s", assembler' %t &&
// RUN: grep -F '1: assembler, {0}, object' %t &&
// RUN: clang -ccc-host-triple i386-apple-darwin9 -ccc-print-phases -c %t.s 2> %t &&
// RUN: grep '0: input, ".*\.s", assembler-with-cpp' %t &&
// RUN: grep -F '1: preprocessor, {0}, assembler' %t &&
// RUN: grep -F '2: assembler, {1}, object' %t &&

// RUN: true
