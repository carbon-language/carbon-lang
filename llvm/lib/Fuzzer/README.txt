===============================
Fuzzer -- a library for coverage-guided fuzz testing.
===============================

This library is intended primarily for in-process coverage-guided fuzz testing
(fuzzing) of other libraries. The typical workflow looks like this:

  * Build the Fuzzer library as a static archive (or just a set of .o files).
    Note that the Fuzzer contains the main() function.
    Preferably do *not* use sanitizers while building the Fuzzer.
  * Build the library you are going to test with -fsanitize-coverage=[234]
    and one of the sanitizers. We recommend to build the library in several
    different modes (e.g. asan, msan, lsan, ubsan, etc) and even using different
    optimizations options (e.g. -O0, -O1, -O2) to diversify testing.
  * Build a test driver using the same options as the library.
    The test driver is a C/C++ file containing interesting calls to the library
    inside a single function:
    extern "C" void TestOneInput(const uint8_t *Data, size_t Size);
  * Link the Fuzzer, the library and the driver together into an executable
    using the same sanitizer options as for the library.
  * Collect the initial corpus of inputs for the
    fuzzer (a directory with test inputs, one file per input).
    The better your inputs are the faster you will find something interesting.
    Also try to keep your inputs small, otherwise the Fuzzer will run too slow.
  * Run the fuzzer with the test corpus. As new interesting test cases are
    discovered they will be added to the corpus. If a bug is discovered by
    the sanitizer (asan, etc) it will be reported as usual and the reproducer
    will be written to disk.
    Each Fuzzer process is single-threaded (unless the library starts its own
    threads). You can run the Fuzzer on the same corpus in multiple processes.
    in parallel. For run-time options run the Fuzzer binary with '-help=1'.


The Fuzzer is similar in concept to AFL (http://lcamtuf.coredump.cx/afl/),
but uses in-process Fuzzing, which is more fragile, more restrictive, but
potentially much faster as it has no overhead for process start-up.
It uses LLVM's "Sanitizer Coverage" instrumentation to get in-process
coverage-feedback https://code.google.com/p/address-sanitizer/wiki/AsanCoverage

The code resides in the LLVM repository and is (or will be) used by various
parts of LLVM, but the Fuzzer itself does not (and should not) depend on any
part of LLVM and can be used for other projects. Ideally, the Fuzzer's code
should not have any external dependencies. Right now it uses STL, which may need
to be fixed later.

Examples of usage in LLVM:
  * clang-format-fuzzer. The inputs are random pieces of C++-like text.
  * TODO: add more

Toy example (see SimpleTest.cpp):
a simple function that does something interesting if it receives bytes "Hi!".
  # Build the Fuzzer with asan:
  % clang++ -std=c++11 -fsanitize=address -fsanitize-coverage=3 -O1 -g \
     Fuzzer*.cpp test/SimpleTest.cpp
  # Run the fuzzer with no corpus (assuming on empty input)
  % ./a.out
