// RUN: llvm-tblgen -I../../include -gen-dag-isel %s | FileCheck %s
// XFAIL: vg_leak

include "llvm/Target/Target.td"

// Make sure the higher complexity pattern comes first
// CHECK: TARGET_VAL(::ADD0)
// CHECK: Complexity = {{[^-]}}
// Make sure the ADD1 pattern has a negative complexity
// CHECK: TARGET_VAL(::ADD1)
// CHECK: Complexity = -{{[0-9]+}}

def TestRC : RegisterClass<"TEST", [i32], 32, (add)>;

def TestInstrInfo : InstrInfo;

def Test : Target {
  let InstructionSet = TestInstrInfo;
}

def ADD0 : Instruction {
  let OutOperandList = (outs TestRC:$dst);
  let InOperandList = (ins TestRC:$src0, TestRC:$src1);
}

def ADD1 : Instruction {
  let OutOperandList = (outs TestRC:$dst);
  let InOperandList = (ins TestRC:$src0, TestRC:$src1);
}

def : Pat <
  (add i32:$src0, i32:$src1),
  (ADD1 $src0, $src1)
> {
  let AddedComplexity = -1000;
}

def : Pat <
   (add i32:$src0, i32:$src1),
   (ADD0 $src0, $src1)
>;
