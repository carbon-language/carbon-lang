// RUN: clang -### -S -x c /dev/null -fblocks -fbuiltin -fmath-errno -fcommon -fpascal-strings -fno-blocks -fno-builtin -fno-math-errno -fno-common -fno-pascal-strings -fblocks -fbuiltin -fmath-errno -fcommon -fpascal-strings %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS1 %s
// RUN: clang -### -S -x c /dev/null -fblocks -fbuiltin -fmath-errno -fcommon -fpascal-strings -fno-blocks -fno-builtin -fno-math-errno -fno-common -fno-pascal-strings -fno-show-source-location -fshort-wchar %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS2 %s
// RUN: clang -fshort-enums -x c /dev/null 2>&1 | FileCheck -check-prefix=CHECK-SHORT-ENUMS %s

// CHECK-OPTIONS1: -fblocks
// CHECK-OPTIONS1: -fpascal-strings

// CHECK-OPTIONS2: -fno-builtin
// CHECK-OPTIONS2: -fno-common
// CHECK-OPTIONS2: -fno-math-errno
// CHECK-OPTIONS2: -fno-show-source-location
// CHECL-OPTIONS2: -fshort-wchar

// CHECK-SHORT-ENUMS: compiler does not support '-fshort-enums'
