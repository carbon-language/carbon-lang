//===-- llvm/CodeGen/MachineCodeForInstruction.h ----------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Representation of the sequence of machine instructions created for a single
// VM instruction.  Additionally records information about hidden and implicit
// values used by the machine instructions: about hidden values used by the
// machine instructions:
// 
// "Temporary values" are intermediate values used in the machine instruction
// sequence, but not in the VM instruction Note that such values should be
// treated as pure SSA values with no interpretation of their operands (i.e., as
// a TmpInstruction object which actually represents such a value).
// 
// (2) "Implicit uses" are values used in the VM instruction but not in
//     the machine instruction sequence
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECODE_FOR_INSTRUCTION_H
#define LLVM_CODEGEN_MACHINECODE_FOR_INSTRUCTION_H

#include "Support/Annotation.h"
#include <vector>

class MachineInstr;
class Instruction;
class Value;
class CallArgsDescriptor;

extern AnnotationID MCFI_AID;

class MachineCodeForInstruction : public Annotation {
  std::vector<Value*> tempVec;          // used by m/c instr but not VM instr
  std::vector<MachineInstr*> Contents;  // the machine instr for this VM instr
  CallArgsDescriptor* callArgsDesc;     // only used for CALL instructions
public:
  MachineCodeForInstruction() : Annotation(MCFI_AID), callArgsDesc(NULL) {}
  ~MachineCodeForInstruction();
  
  static MachineCodeForInstruction &get(const Instruction *I) {
    assert(I != NULL);
    return *(MachineCodeForInstruction*)
      ((Annotable*)I)->getOrCreateAnnotation(MCFI_AID);
  }
  static void destroy(const Instruction *I) {
    ((Annotable*)I)->deleteAnnotation(MCFI_AID);
  }

  // Access to underlying machine instructions...
  typedef std::vector<MachineInstr*>::iterator iterator;
  typedef std::vector<MachineInstr*>::const_iterator const_iterator;

  unsigned size() const { return Contents.size(); }
  bool empty() const { return Contents.empty(); }
  MachineInstr *front() const { return Contents.front(); }
  MachineInstr *back() const { return Contents.back(); }
  MachineInstr *&operator[](unsigned i) { return Contents[i]; }
  MachineInstr *operator[](unsigned i) const { return Contents[i]; }
  void pop_back() { Contents.pop_back(); }

  iterator begin() { return Contents.begin(); }
  iterator end()   { return Contents.end(); }
  const_iterator begin() const { return Contents.begin(); }
  const_iterator end()   const { return Contents.end(); }

  template<class InIt>
  void insert(iterator where, InIt first, InIt last) {
    Contents.insert(where, first, last);
  }
  iterator erase(iterator where) { return Contents.erase(where); }
  iterator erase(iterator s, iterator e) { return Contents.erase(s, e); }
  

  // dropAllReferences() - This function drops all references within
  // temporary (hidden) instructions created in implementing the original
  // VM intruction.  This ensures there are no remaining "uses" within
  // these hidden instructions, before the values of a method are freed.
  //
  void dropAllReferences();

  const std::vector<Value*> &getTempValues() const { return tempVec; }
        std::vector<Value*> &getTempValues()       { return tempVec; }
  
  MachineCodeForInstruction &addTemp(Value *tmp) {
    tempVec.push_back(tmp);
    return *this;
  }

  void setCallArgsDescriptor(CallArgsDescriptor* desc) { callArgsDesc = desc; }
  CallArgsDescriptor* getCallArgsDescriptor() const    { return callArgsDesc; }
};

#endif
