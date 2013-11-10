// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: WorldHelloCC
// CHECK-NOT: WorldHelloCC

class C<string n> {
  string name = n;
}

multiclass Names<string n, string m> {
   def CC : C<n>;
   def World#NAME#CC : C<m>;
}

defm Hello : Names<"hello", "world">;
