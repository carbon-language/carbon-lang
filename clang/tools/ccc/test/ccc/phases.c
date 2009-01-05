// One C file.
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases a.c > %t &&
// RUN: grep '0: input, "a.c", c' %t &&
// RUN: grep '1: preprocessor, {0}, cpp-output' %t &&
// RUN: grep '2: compiler, {1}, assembler' %t &&
// RUN: grep '3: assembler, {2}, object' %t &&
// RUN: grep '4: linker, {3}, image' %t &&

// PCH.
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases -x c-header a.h > %t &&
// RUN: grep '0: input, "a.h", c-header' %t &&
// RUN: grep '1: preprocessor, {0}, c-header-cpp-output' %t &&
// RUN: grep '2: precompiler, {1}, precompiled-header' %t &&

// Assembler w/ and w/o preprocessor.
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases -x assembler a.s > %t &&
// RUN: grep '0: input, "a.s", assembler' %t &&
// RUN: grep '1: assembler, {0}, object' %t &&
// RUN: grep '2: linker, {1}, image' %t &&
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases -x assembler-with-cpp a.s > %t &&
// RUN: grep '0: input, "a.s", assembler-with-cpp' %t &&
// RUN: grep '1: preprocessor, {0}, assembler' %t &&
// RUN: grep '2: assembler, {1}, object' %t &&
// RUN: grep '3: linker, {2}, image' %t &&

// Check the various ways of early termination.
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases -E a.c > %t &&
// RUN: not grep ': compiler, ' %t &&
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases -fsyntax-only a.c > %t &&
// RUN: grep ': compiler, {1}, nothing' %t &&
// RUN: not grep ': assembler, ' %t &&
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases -S a.c > %t &&
// RUN: not grep ': assembler, ' %t &&
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases -c a.c > %t &&
// RUN: not grep ': linker, ' %t &&

// Multiple output files.
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases -c a.c b.c > %t &&
// RUN: grep ': assembler,' %t | count 2 &&

// FIXME: Only for darwin.
// Treat -filelist as a linker input.
// RUN: xcc -ccc-no-driver-driver -ccc-print-phases -filelist /dev/null > %t &&
// RUN: grep '1: linker, {0}, image' %t &&

// RUN: true
