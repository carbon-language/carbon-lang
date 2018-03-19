// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

class A;
class B;

def A0 : A;

class C<string name> {
  B b = !cast<B>(name);
}

// CHECK: error: Invalid value of type 'A' is found when setting 'b' of type 'B'
def Test : C<"A0">;
