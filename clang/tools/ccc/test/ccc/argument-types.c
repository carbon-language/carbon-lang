// Input argument:
// RUN: xcc -ccc-print-options %s | grep 'Name: "<input>", Values: {"%s"}' | count 1 &&

// Joined or separate arguments:
// RUN: xcc -ccc-print-options -xc -x c | grep 'Name: "-x", Values: {"c"}' | count 2 &&

// Joined and separate arguments:
// RUN: xcc -ccc-print-options -Xarch_mips -run | grep 'Name: "-Xarch_", Values: {"mips", "-run"}' | count 1 &&

// Multiple arguments:
// RUN: xcc -ccc-print-options -sectorder 1 2 3 | grep 'Name: "-sectorder", Values: {"1", "2", "3"}' | count 1 &&

// Unknown argument:
// RUN: xcc -ccc-print-options -=== | grep 'Name: "<unknown>", Values: {"-==="}' | count 1 && 

// RUN: true
