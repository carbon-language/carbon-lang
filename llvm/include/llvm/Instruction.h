//===-- llvm/Instruction.h - Instruction class definition --------*- C++ -*--=//
//
// This file contains the declaration of the Instruction class, which is the
// base class for all of the VM instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INSTRUCTION_H
#define LLVM_INSTRUCTION_H

#include "llvm/User.h"

class Type;
class BasicBlock;
class Method;
class MachineInstr;		// do not include header file MachineInstr.h
class MachineCodeForVMInstr;

class Instruction : public User {
  BasicBlock *Parent;
  unsigned iType;      // InstructionType

  MachineCodeForVMInstr* machineInstrVec;
  friend class ValueHolder<Instruction,BasicBlock,Method>;
  inline void setParent(BasicBlock *P) { Parent = P; }

public:
  Instruction(const Type *Ty, unsigned iType, const string &Name = "");
  virtual ~Instruction();  // Virtual dtor == good.

  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name);
  
  // clone() - Create a copy of 'this' instruction that is identical in all ways
  // except the following:
  //   * The instruction has no parent
  //   * The instruction has no name
  //
  virtual Instruction *clone() const = 0;
  
  // Add a machine instruction used to implement this instruction
  //
  void addMachineInstruction(MachineInstr* minstr);
  
  // Accessor methods...
  //
  inline const BasicBlock *getParent() const { return Parent; }
  inline       BasicBlock *getParent()       { return Parent; }
  virtual bool hasSideEffects() const { return false; }  // Memory & Call insts

  // ---------------------------------------------------------------------------
  // Machine code accessors...
  //
  inline MachineCodeForVMInstr &getMachineInstrVec() {
    return *machineInstrVec; 
  }
  
  // ---------------------------------------------------------------------------
  // Subclass classification... getInstType() returns a member of 
  // one of the enums that is coming soon (down below)...
  //
  virtual const char *getOpcodeName() const = 0;
  unsigned getOpcode() const { return iType; }

  // getInstType is deprecated, use getOpcode() instead.
  unsigned getInstType() const { return iType; }

  inline bool isTerminator() const {   // Instance of TerminatorInst?
    return iType >= FirstTermOp && iType < NumTermOps; 
  }
  inline bool isDefinition() const { return !isTerminator(); }
  inline bool isUnaryOp() const {
    return iType >= FirstUnaryOp && iType < NumUnaryOps;
  }
  inline bool isBinaryOp() const {
    return iType >= FirstBinaryOp && iType < NumBinaryOps;
  }

  // isPHINode() - This is used frequently enough to allow it to exist
  inline bool isPHINode() const { return iType == PHINode; }

  // dropAllReferences() - This function is in charge of "letting go" of all
  // objects that this Instruction refers to.  This first lets go of all
  // references to hidden values generated code for this instruction,
  // and then drops all references to its operands.
  // 
  void dropAllReferences();
  
  //----------------------------------------------------------------------
  // Exported enumerations...
  //
  enum TermOps {       // These terminate basic blocks
    FirstTermOp = 1,
    Ret = 1, Br, Switch, 
    NumTermOps         // Must remain at end of enum
  };

  enum UnaryOps {
    FirstUnaryOp = NumTermOps,
    Not          = NumTermOps,      // Binary inverse

    NumUnaryOps        // Must remain at end of enum
  };

  enum BinaryOps {
    // Standard binary operators...
    FirstBinaryOp = NumUnaryOps,
    Add = NumUnaryOps, Sub, Mul, Div, Rem,

    // Logical operators...
    And, Or, Xor,

    // Binary comparison operators...
    SetEQ, SetNE, SetLE, SetGE, SetLT, SetGT,

    NumBinaryOps
  };

  enum MemoryOps {
    FirstMemoryOp = NumBinaryOps,
    Malloc = NumBinaryOps, Free,     // Heap management instructions
    Alloca,                          // Stack management instruction

    Load, Store,                     // Memory manipulation instructions.
    GetElementPtr,                   // Get addr of Structure or Array element

    NumMemoryOps
  };

  enum OtherOps {
    FirstOtherOp = NumMemoryOps,
    PHINode      = NumMemoryOps,     // PHI node instruction
    Cast,                            // Type cast...
    Call,                            // Call a function

    Shl, Shr,                        // Shift operations...

    NumOps,                          // Must be the last 'op' defined.
    UserOp1, UserOp2                 // May be used internally to a pass...
  };
};

#endif
