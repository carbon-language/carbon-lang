// clang-format off
// REQUIRES: lld, system-windows

// RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/stack_unwinding01.lldbinit 2>&1 | FileCheck %s


struct Struct {
  void simple_method(int a, int b) {
     a += 1;
     simple_method(a, b);
  }
};



int main(int argc, char **argv) {
  Struct s;
  s.simple_method(1,2);
  return 0;
}


// CHECK: (lldb) thread backtrace
// CHECK-NEXT: * thread #1, stop reason = breakpoint 1.1
// CHECK-NEXT:   * frame #0: {{.*}} stack_unwinding01.cpp.tmp.exe`Struct::simple_method at stack_unwinding01.cpp:12
// CHECK-NEXT:     frame #1: {{.*}} stack_unwinding01.cpp.tmp.exe`main(argc={{.*}}, argv={{.*}}) at stack_unwinding01.cpp:20


// CHECK: (lldb) thread backtrace
// CHECK-NEXT: * thread #1, stop reason = breakpoint 1.1
// CHECK-NEXT:   * frame #0: {{.*}} stack_unwinding01.cpp.tmp.exe`Struct::simple_method at stack_unwinding01.cpp:12
// CHECK-NEXT:     frame #1: {{.*}} stack_unwinding01.cpp.tmp.exe`Struct::simple_method at stack_unwinding01.cpp:12
// CHECK-NEXT:     frame #2: {{.*}} stack_unwinding01.cpp.tmp.exe`main(argc={{.*}}, argv={{.*}}) at stack_unwinding01.cpp:20

// CHECK: (lldb) thread backtrace
// CHECK-NEXT: * thread #1, stop reason = breakpoint 1.1
// CHECK-NEXT:   * frame #0: {{.*}} stack_unwinding01.cpp.tmp.exe`Struct::simple_method at stack_unwinding01.cpp:12
// CHECK-NEXT:     frame #1: {{.*}} stack_unwinding01.cpp.tmp.exe`Struct::simple_method at stack_unwinding01.cpp:12
// CHECK-NEXT:     frame #2: {{.*}} stack_unwinding01.cpp.tmp.exe`Struct::simple_method at stack_unwinding01.cpp:12
// CHECK-NEXT:     frame #3: {{.*}} stack_unwinding01.cpp.tmp.exe`main(argc={{.*}}, argv={{.*}}) at stack_unwinding01.cpp:20
