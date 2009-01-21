// One C file.
// RUN: touch %t.c &&
// RUN: xcc -ccc-host-system unknown -ccc-print-phases %t.c > %t &&
// RUN: grep '0: input, "%t.c", c' %t &&
// RUN: grep '1: preprocessor, {0}, cpp-output' %t &&
// RUN: grep '2: compiler, {1}, assembler' %t &&
// RUN: grep '3: assembler, {2}, object' %t &&
// RUN: grep '4: linker, {3}, image' %t &&

// PCH.
// RUN: touch %t.h &&
// RUN: xcc -ccc-host-system unknown -ccc-print-phases -x c-header %t.h > %t &&
// RUN: grep '0: input, "%t.h", c-header' %t &&
// RUN: grep '1: preprocessor, {0}, c-header-cpp-output' %t &&
// RUN: grep '2: precompiler, {1}, precompiled-header' %t &&

// Assembler w/ and w/o preprocessor.
// RUN: touch %t.s &&
// RUN: xcc -ccc-host-system unknown -ccc-print-phases -x assembler %t.s > %t &&
// RUN: grep '0: input, "%t.s", assembler' %t &&
// RUN: grep '1: assembler, {0}, object' %t &&
// RUN: grep '2: linker, {1}, image' %t &&
// RUN: xcc -ccc-host-system unknown -ccc-print-phases -x assembler-with-cpp %t.s > %t &&
// RUN: grep '0: input, "%t.s", assembler-with-cpp' %t &&
// RUN: grep '1: preprocessor, {0}, assembler' %t &&
// RUN: grep '2: assembler, {1}, object' %t &&
// RUN: grep '3: linker, {2}, image' %t &&

// Check the various ways of early termination.
// RUN: xcc -ccc-host-system unknown -ccc-print-phases -E %s > %t &&
// RUN: not grep ': compiler, ' %t &&
// RUN: xcc -ccc-host-system unknown -ccc-print-phases -fsyntax-only %s > %t &&
// RUN: grep ': syntax-only, {1}, nothing' %t &&
// RUN: not grep ': assembler, ' %t &&
// RUN: xcc -ccc-host-system unknown -ccc-print-phases -S %s > %t &&
// RUN: not grep ': assembler, ' %t &&
// RUN: xcc -ccc-host-system unknown -ccc-print-phases -c %s > %t &&
// RUN: not grep ': linker, ' %t &&

// Multiple output files.
// RUN: touch %t.1.c &&
// RUN: touch %t.2.c &&
// RUN: xcc -ccc-host-system unknown -ccc-print-phases -c %t.1.c %t.2.c > %t &&
// RUN: grep ': assembler,' %t | count 2 &&

// FIXME: Only for darwin.
// Treat -filelist as a linker input.
// RUN: xcc -ccc-host-system unknown -ccc-print-phases -filelist /dev/null > %t &&
// RUN: grep '1: linker, {0}, image' %t &&

// RUN: true
