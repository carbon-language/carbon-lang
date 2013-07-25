// RUN: llvm-tblgen -gen-intrinsic %s | FileCheck %s

class IntrinsicProperty;

class ValueType<int size, int value> {
  string Namespace = "MVT";
  int Size = size;
  int Value = value;
}

class LLVMType<ValueType vt> {
  ValueType VT = vt;
}

class Intrinsic<string name, list<LLVMType> param_types = []> {
  string LLVMName = name;
  bit isTarget = 0;
  string TargetPrefix = "";
  list<LLVMType> RetTypes = [];
  list<LLVMType> ParamTypes = param_types;
  list<IntrinsicProperty> Properties = [];
}

def iAny : ValueType<0, 254>;
def llvm_anyint_ty : LLVMType<iAny>;


// Make sure an intrinsic name that is a prefix of another is checked after the
// other.

// CHECK: if (NameR.startswith("oo.bar.")) return Intrinsic::foo_bar;
// CHECK: if (NameR.startswith("oo.")) return Intrinsic::foo;

def int_foo : Intrinsic<"llvm.foo", [llvm_anyint_ty]>;
def int_foo_bar : Intrinsic<"llvm.foo.bar", [llvm_anyint_ty]>;
