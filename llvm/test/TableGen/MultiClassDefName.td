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

// Ensure that the same anonymous name is used as the prefix for all defs in an
// anonymous multiclass.

class Outer<C i> {
  C Inner = i;
}

multiclass MC<string name> {
  def hi : C<name>;
  def there : Outer<!cast<C>(!strconcat(NAME, "hi"))>;
}

defm : MC<"foo">;

