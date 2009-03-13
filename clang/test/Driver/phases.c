// Basic compilation for various types of files.
// RUN: clang-driver -ccc-host-triple i386-unknown-unknown -ccc-print-phases -x c %s -x objective-c %s -x c++ %s -x objective-c++ -x assembler %s -x assembler-with-cpp %s -x none %s &> %t &&
// RUN: grep -F '0: input, "phases.c", c' %t &&
// RUN: grep -F '1: preprocessor, {0}, cpp-output' %t &&
// RUN: grep -F '2: compiler, {1}, assembler' %t &&
// RUN: grep -F '3: assembler, {2}, object' %t &&
// RUN: grep -F '4: input, "phases.c", objective-c' %t &&
// RUN: grep -F '5: preprocessor, {4}, objective-c-cpp-output' %t &&
// RUN: grep -F '6: compiler, {5}, assembler' %t &&
// RUN: grep -F '7: assembler, {6}, object' %t &&
// RUN: grep -F '8: input, "phases.c", c++' %t &&
// RUN: grep -F '9: preprocessor, {8}, c++-cpp-output' %t &&
// RUN: grep -F '10: compiler, {9}, assembler' %t &&
// RUN: grep -F '11: assembler, {10}, object' %t &&
// RUN: grep -F '12: input, "phases.c", assembler' %t &&
// RUN: grep -F '13: assembler, {12}, object' %t &&
// RUN: grep -F '14: input, "phases.c", assembler-with-cpp' %t &&
// RUN: grep -F '15: preprocessor, {14}, assembler' %t &&
// RUN: grep -F '16: assembler, {15}, object' %t &&
// RUN: grep -F '17: input, "phases.c", c' %t &&
// RUN: grep -F '18: preprocessor, {17}, cpp-output' %t &&
// RUN: grep -F '19: compiler, {18}, assembler' %t &&
// RUN: grep -F '20: assembler, {19}, object' %t &&
// RUN: grep -F '21: linker, {3, 7, 11, 13, 16, 20}, image' %t &&

// RUN: true
