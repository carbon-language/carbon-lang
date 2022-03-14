// REQUIRES: osx-ld64-live_support

// Compiling with PGO/code coverage on Darwin should raise no warnings or errors
// when using an exports list.

// 1) Check that using PGO/code coverage flags with an export list containing
// just "_main" produces no warnings or errors.
//
// RUN: echo "_main" > %t.exports
// RUN: %clang_pgogen -Werror -Wl,-exported_symbols_list,%t.exports -o %t %s 2>&1 | tee %t.log
// RUN: %clang_profgen -Werror -fcoverage-mapping -Wl,-exported_symbols_list,%t.exports -o %t %s 2>&1 | tee -a %t.log
// RUN: cat %t.log | count 0

// 2) Ditto (1), but for GCOV.
//
// RUN: %clang -Werror -Wl,-exported_symbols_list,%t.exports --coverage -o %t.gcov %s | tee -a %t.gcov.log
// RUN: cat %t.gcov.log | count 0

// 3) The default set of weak external symbols should match the set of symbols
// exported by clang. See Darwin::addProfileRTLibs. This requirement was put in
// place to support tapi binary verification.
//
// RUN: %clang_pgogen -Werror -o %t.default %s
// RUN: nm -jUg %t.default | grep -v __mh_execute_header > %t.default.exports
// RUN: nm -jUg %t > %t.clang.exports
// RUN: diff %t.default.exports %t.clang.exports

// 4) Ditto (3), but for GCOV.
//
// RUN: %clang -Werror --coverage -o %t.gcov.default %s
// RUN: nm -jUg %t.gcov | grep -v __mh_execute_header > %t.gcov.exports
// RUN: nm -jUg %t.gcov.default | grep -v __mh_execute_header > %t.gcov.default.exports
// RUN: diff %t.gcov.default.exports %t.gcov.exports

int main() {}
