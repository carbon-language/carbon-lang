// REQUIRES: lld

// RUN: %clang %s -gdwarf-5 -gpubnames -gsplit-dwarf -c -emit-llvm -o - --target=x86_64-pc-linux -DONE | \
// RUN:   llc -filetype=obj -split-dwarf-file=%t-1.dwo -o %t-1.o
// RUN: llvm-objcopy --split-dwo=%t-1.dwo %t-1.o
// RUN: %clang %s -gdwarf-5 -gpubnames -gsplit-dwarf -c -emit-llvm -o - --target=x86_64-pc-linux -DTWO | \
// RUN:   llc -filetype=obj -split-dwarf-file=%t-2.dwo -o %t-2.o
// RUN: llvm-objcopy --split-dwo=%t-2.dwo %t-2.o
// RUN: ld.lld %t-1.o %t-2.o -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix NAMES
// RUN: lldb-test symbols --name=foo --find=variable %t | FileCheck %s

// NAMES: Name: .debug_names

// CHECK: Found 2 variables:
#ifdef ONE
namespace one {
int foo;
// CHECK-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = find-variable-dwo.cpp:[[@LINE-1]]
} // namespace one

extern "C" void _start() {}
#else
namespace two {
int foo;
// CHECK-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = find-variable-dwo.cpp:[[@LINE-1]]
} // namespace two
#endif
