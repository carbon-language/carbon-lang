// To create the covmapping for this file on Linux, copy this file to /tmp
// cd into /tmp. Use llvm-cov convert-for-testing to extract the covmapping.
// This test is Windows-only. It checks that all paths, which are generated
// in the index and source coverage reports, are native path. For example,
// on Windows all '/' are converted to '\'.
// REQUIRES: system-windows

// RUN: llvm-profdata merge %S/Inputs/double_dots.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/native_separators.covmapping -instr-profile=%t.profdata -o %t.dir
// RUN: FileCheck -check-prefixes=TEXT-INDEX -input-file=%t.dir/index.txt %s
// RUN: llvm-cov show -format=html %S/Inputs/native_separators.covmapping -instr-profile=%t.profdata -path-equivalence=/tmp,%S %S/../llvm-"config"/../llvm-"cov"/native_separators.c -o %t.dir
// RUN: FileCheck -check-prefixes=HTML-INDEX -input-file=%t.dir/index.html %s
// RUN: llvm-cov show -format=html %S/Inputs/native_separators.covmapping -instr-profile=%t.profdata -path-equivalence=/tmp,%S %s -o %t.dir
// RUN: FileCheck -check-prefixes=HTML -input-file=%t.dir/coverage/tmp/native_separators.c.html %s

// TEXT-INDEX: \tmp\native_separators.c
// HTML-INDEX: >tmp\native_separators.c</a>
// HTML: <pre>\tmp\native_separators.c</pre>

int main() {}
