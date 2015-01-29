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
to be fixed later. See also F.A.Q. below.

Examples of usage in LLVM:
  * clang-format-fuzzer. The inputs are random pieces of C++-like text.
  * Build (make sure to use fresh clang as the host compiler):
    cmake -GNinja  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_SANITIZER=Address -DLLVM_USE_SANITIZE_COVERAGE=YES \
    /path/to/llvm -DCMAKE_BUILD_TYPE=Release
    ninja clang-format-fuzzer
  * Optionally build other kinds of binaries (asan+Debug, msan, ubsan, etc)
  * TODO: commit the pre-fuzzed corpus to svn (?).
  * Run:
      clang-format-fuzzer CORPUS_DIR

Toy example (see SimpleTest.cpp):
a simple function that does something interesting if it receives bytes "Hi!".
  # Build the Fuzzer with asan:
  % clang++ -std=c++11 -fsanitize=address -fsanitize-coverage=3 -O1 -g \
     Fuzzer*.cpp test/SimpleTest.cpp
  # Run the fuzzer with no corpus (assuming on empty input)
  % ./a.out

===============================================================================
F.A.Q.

Q. Why Fuzzer does not use any of the LLVM support?
A. There are two reasons.
First, we want this library to be used outside of the LLVM w/o users having to
build the rest of LLVM. This may sound unconvincing for many LLVM folks,
but in practice the need for building the whole LLVM frightens many potential
users -- and we want more users to use this code.
Second, there is a subtle technical reason not to rely on the rest of LLVM, or
any other large body of code (maybe not even STL). When coverage instrumentation
is enabled, it will also instrument the LLVM support code which will blow up the
coverage set of the process (since the fuzzer is in-process). In other words, by
using more external dependencies we will slow down the fuzzer while the main
reason for it to exist is extreme speed.

Q. What about Windows then? The Fuzzer contains code that does not build on
Windows.
A. The sanitizer coverage support does not work on Windows either as of 01/2015.
Once it's there, we'll need to re-implement OS-specific parts (I/O, signals).

Q. When this Fuzzer is not a good solution for a problem?
A.
  * If the test inputs are validated by the target library and the validator
    asserts/crashes on invalid inputs, the in-process fuzzer is not applicable
    (we could use fork() w/o exec, but it comes with extra overhead).
  * Bugs in the target library may accumulate w/o being detected. E.g. a memory
    corruption that goes undetected at first and then leads to a crash while
    testing another input. This is why it is highly recommended to run this
    in-process fuzzer with all sanitizers to detect most bugs on the spot.
  * It is harder to protect the in-process fuzzer from excessive memory
    consumption and infinite loops in the target library (still possible).
  * The target library should not have significant global state that is not
    reset between the runs.
  * Many interesting target libs are not designed in a way that supports
    the in-process fuzzer interface (e.g. require a file path instead of a
    byte array).
  * If a single test run takes a considerable fraction of a second (or
    more) the speed benefit from the in-process fuzzer is negligible.
  * If the target library runs persistent threads (that outlive
    execution of one test) the fuzzing results will be unreliable.
