//===-- llvm/CodeGen/MachineCodeForInstruction.h -----------------*- C++ -*--=//
//
//   Representation of the sequence of machine instructions created
//   for a single VM instruction.  Additionally records information
//   about hidden and implicit values used by the machine instructions:
//   about hidden values used by the machine instructions:
// 
//   "Temporary values" are intermediate values used in the machine
//   instruction sequence, but not in the VM instruction
//   Note that such values should be treated as pure SSA values with
//   no interpretation of their operands (i.e., as a TmpInstruction
//   object which actually represents such a value).
// 
//   (2) "Implicit uses" are values used in the VM instruction but not in
//       the machine instruction sequence
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECODE_FOR_INSTRUCTION_H
#define LLVM_CODEGEN_MACHINECODE_FOR_INSTRUCTION_H

#include "llvm/Annotation.h"
#include <vector>
class MachineInstr;
class Instruction;
class Value;

class MachineCodeForInstruction 
                  : public Annotation, public std::vector<MachineInstr*> {
  std::vector<Value*> tempVec;         // used by m/c instr but not VM instr
  
public:
  MachineCodeForInstruction();
  ~MachineCodeForInstruction();
  
  static MachineCodeForInstruction &get(const Instruction *I);
  static void destroy(const Instruction *I);

  const std::vector<Value*> &getTempValues() const { return tempVec; }
        std::vector<Value*> &getTempValues()       { return tempVec; }
  
  inline MachineCodeForInstruction &addTemp(Value *tmp) {
    tempVec.push_back(tmp);
    return *this;
  }
};

#endif
