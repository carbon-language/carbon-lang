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

multiclass MC2<string name> {
  def there : Outer<C<name> >;
}

// Ensure that we've correctly captured the reference to name from the implicit
// anonymous C def in the template parameter list of Outer.
// CHECK-NOT: MC2::name

defm : MC2<"bar">;

