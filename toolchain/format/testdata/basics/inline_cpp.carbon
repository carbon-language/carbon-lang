// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/inline_cpp.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/inline_cpp.carbon

// --- test.carbon

// The `import Cpp inline` form's body is C++ and is reformatted even though it
// has no `'''cpp` file type indicator. Pointer declarators bind to the type
// (`int* p`), matching Carbon's own C++ style.
import Cpp inline '''
int  square ( int  n ) {return  n*n ;}
int *lookup(int *table, int i) { return &table[i]; }
''';

// The `inline Cpp` form is reformatted the same way.
inline Cpp '''
void  G ( ) {
  Carbon::A::F( ) ;
}
''';

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // The `import Cpp inline` form's body is C++ and is reformatted even though it
// CHECK:STDOUT: // has no `'''cpp` file type indicator. Pointer declarators bind to the type
// CHECK:STDOUT: // (`int* p`), matching Carbon's own C++ style.
// CHECK:STDOUT: import Cpp inline '''
// CHECK:STDOUT: int square(int n) { return n * n; }
// CHECK:STDOUT: int* lookup(int* table, int i) { return &table[i]; }
// CHECK:STDOUT: ''';
// CHECK:STDOUT:
// CHECK:STDOUT: // The `inline Cpp` form is reformatted the same way.
// CHECK:STDOUT: inline Cpp '''
// CHECK:STDOUT: void G() { Carbon::A::F(); }
// CHECK:STDOUT: ''';
