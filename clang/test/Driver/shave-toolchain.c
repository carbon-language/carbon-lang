// Ensure that '-target shave' picks a different compiler.
// Also check that '-I' is turned into '-i:' for the assembler.

// Note that since we don't know where movi tools are installed,
// the driver may or may not find a full path to them.
// That is, the 0th argument will be "/path/to/my/moviCompile"
// or just "moviCompile" depending on whether moviCompile is found.
// As such, we test only for a trailing quote in its rendering.
// The same goes for "moviAsm".

// RUN: %clang -target shave -c -### %s -Icommon 2>&1 \
// RUN:   | FileCheck %s -check-prefix=movicompile
// movicompile: moviCompile" "-DMYRIAD2"
// movicompile: moviAsm" "-no6thSlotCompression" "-cv:myriad2" "-noSPrefixing" "-a" "-i:common" "-elf"

// RUN: %clang -target shave -c -### %s -DEFINE_ME -UNDEFINE_ME 2>&1 \
// RUN:   | FileCheck %s -check-prefix=defines
// defines: "-D" "EFINE_ME" "-U" "NDEFINE_ME"

// RUN: %clang -target shave -c -### %s -Icommon -iquote quotepath -isystem syspath 2>&1 \
// RUN:   | FileCheck %s -check-prefix=includes
// includes: "-iquote" "quotepath" "-isystem" "syspath"
