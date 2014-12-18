// -flto causes a switch to llvm-bc object files.
// RUN: %clang -ccc-print-phases -c %s -flto 2> %t.log
// RUN: grep '2: compiler, {1}, lto-bc' %t.log

// RUN: %clang -ccc-print-phases %s -flto 2> %t.log
// RUN: grep '0: input, ".*lto.c", c' %t.log
// RUN: grep '1: preprocessor, {0}, cpp-output' %t.log
// RUN: grep '2: compiler, {1}, lto-bc' %t.log
// RUN: grep '3: linker, {2}, image' %t.log

// llvm-bc and llvm-ll outputs need to match regular suffixes
// (unfortunately).
// RUN: %clang %s -flto -save-temps -### 2> %t.log
// RUN: grep '"-o" ".*lto\.i" "-x" "c" ".*lto\.c"' %t.log
// RUN: grep '"-o" ".*lto\.o" .*".*lto\.i"' %t.log
// RUN: grep '".*a.out" .*".*lto\.o"' %t.log

// RUN: %clang %s -flto -S -### 2> %t.log
// RUN: grep '"-o" ".*lto\.s" "-x" "c" ".*lto\.c"' %t.log

// RUN: not %clang %s -emit-llvm 2>&1 | FileCheck --check-prefix=LLVM-LINK %s
// LLVM-LINK: -emit-llvm cannot be used when linking
