//===-- llvm/Instruction.h - Instruction class definition --------*- C++ -*--=//
//
// This file contains the declaration of the Instruction class, which is the
// base class for all of the VM instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INSTRUCTION_H
#define LLVM_INSTRUCTION_H

#include "llvm/User.h"

class Instruction : public User {
  BasicBlock *Parent;

  friend class ValueHolder<Instruction,BasicBlock,Method>;
  inline void setParent(BasicBlock *P) { Parent = P; }
protected:
  unsigned iType;      // InstructionType
public:
  Instruction(const Type *Ty, unsigned iType, const std::string &Name = "");
  virtual ~Instruction() {
    assert(Parent == 0 && "Instruction still embedded in basic block!");
  }

  // Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);
  
  // clone() - Create a copy of 'this' instruction that is identical in all ways
  // except the following:
  //   * The instruction has no parent
  //   * The instruction has no name
  //
  virtual Instruction *clone() const = 0;
  
  // Accessor methods...
  //
  inline const BasicBlock *getParent() const { return Parent; }
  inline       BasicBlock *getParent()       { return Parent; }
  virtual bool hasSideEffects() const { return false; }  // Memory & Call insts

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

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::InstructionVal;
  }
  
  //----------------------------------------------------------------------
  // Exported enumerations...
  //
  enum TermOps {       // These terminate basic blocks
#define  FIRST_TERM_INST(N)             FirstTermOp = N,
#define HANDLE_TERM_INST(N, OPC, CLASS) OPC = N,
#define   LAST_TERM_INST(N)             NumTermOps = N+1,
#include "llvm/Instruction.def"
  };

  enum UnaryOps {
#define  FIRST_UNARY_INST(N)             FirstUnaryOp = N,
#define HANDLE_UNARY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_UNARY_INST(N)             NumUnaryOps = N+1,
#include "llvm/Instruction.def"
  };

  enum BinaryOps {
#define  FIRST_BINARY_INST(N)             FirstBinaryOp = N,
#define HANDLE_BINARY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_BINARY_INST(N)             NumBinaryOps = N+1,
#include "llvm/Instruction.def"
  };

  enum MemoryOps {
#define  FIRST_MEMORY_INST(N)             FirstMemoryOp = N,
#define HANDLE_MEMORY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_MEMORY_INST(N)             NumMemoryOps = N+1,
#include "llvm/Instruction.def"
  };

  enum OtherOps {
#define  FIRST_OTHER_INST(N)             FirstOtherOp = N,
#define HANDLE_OTHER_INST(N, OPC, CLASS) OPC = N,
#define   LAST_OTHER_INST(N)             NumOtherOps = N+1,
#include "llvm/Instruction.def"
  };
};

#endif
