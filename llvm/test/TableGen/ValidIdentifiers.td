// Ensure that anonymous names are valid identifiers via the ctags index
// RUN: llvm-tblgen -gen-ctags %s | grep -v '^!' | not grep -viE '^[a-z_][a-z0-9_]*\t'
// Test validation
// RUN: llvm-tblgen -gen-ctags %s | grep '^anonymous' > /dev/null
// XFAIL: vg_leak

class foo<int X> { int THEVAL = X; }

def : foo<2>;

def X {
  foo Y = foo<1>;
}
