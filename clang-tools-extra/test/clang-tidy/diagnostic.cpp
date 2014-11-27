// RUN: clang-tidy -checks='-*,misc-use-override' %s.nonexistent.cpp -- | FileCheck -check-prefix=CHECK1 %s
// RUN: clang-tidy -checks='-*,clang-diagnostic-*,google-explicit-constructor' %s -- -fan-unknown-option | FileCheck -check-prefix=CHECK2 %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor,clang-diagnostic-literal-conversion' %s -- -fan-unknown-option | FileCheck -check-prefix=CHECK3 %s
// RUN: clang-tidy -checks='-*,misc-use-override,clang-diagnostic-macro-redefined' %s -- -DMACRO_FROM_COMMAND_LINE | FileCheck -check-prefix=CHECK4 %s

// CHECK1-NOT: warning
// CHECK2-NOT: warning
// CHECK3-NOT: warning

// CHECK1: error: error reading '{{.*}}.nonexistent.cpp' [clang-diagnostic-error]
// CHECK2: error: unknown argument: '-fan-unknown-option' [clang-diagnostic-error]

// CHECK2: :[[@LINE+2]]:9: warning: implicit conversion from 'double' to 'int' changes value from 1.5 to 1 [clang-diagnostic-literal-conversion]
// CHECK3: :[[@LINE+1]]:9: warning: implicit conversion from 'double' to 'int' changes value
int a = 1.5;

// CHECK2: :[[@LINE+2]]:11: warning: single-argument constructors must be explicit [google-explicit-constructor]
// CHECK3: :[[@LINE+1]]:11: warning: single-argument constructors must be explicit [google-explicit-constructor]
class A { A(int) {} };

// CHECK2-NOT: warning:
// CHECK3-NOT: warning:

#define MACRO_FROM_COMMAND_LINE
// CHECK4: :[[@LINE-1]]:9: warning: 'MACRO_FROM_COMMAND_LINE' macro redefined
