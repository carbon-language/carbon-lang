int foo(int a) {
  return a + 1;
}

int main(int argc, char *argv[]) {
  return foo(argc);
}

// Build instructions:
// 1) clang++ -### -O2 -gsplit-dwarf.cc split-dwarf-test.cc -o split-dwarf-test
// 2) Replace the value "-fdebug-compilation-dir" flag to "Output"
//      (this is the temp directory used by lit).
// 3) Manually run clang-cc1, objcopy and ld invocations.
// 4) Copy the binary and .dwo file to the Inputs directory. Make sure the
//    .dwo file will be available for symbolizer (use test RUN-lines to copy
//    the .dwo file to a directory
//    <execution_directory>/<directory_provided_in_fdebug_compilation_dir>.
