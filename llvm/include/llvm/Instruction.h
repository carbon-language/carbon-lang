//===-- llvm/Instruction.h - Instruction class definition -------*- C++ -*-===//
//
// This file contains the declaration of the Instruction class, which is the
// base class for all of the LLVM instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INSTRUCTION_H
#define LLVM_INSTRUCTION_H

#include "llvm/User.h"
template<typename SC> struct ilist_traits;
template<typename ValueSubClass, typename ItemParentClass, typename SymTabClass,
         typename SubClass> class SymbolTableListTraits;

class Instruction : public User {
  BasicBlock *Parent;
  Instruction *Prev, *Next; // Next and Prev links for our intrusive linked list

  void setNext(Instruction *N) { Next = N; }
  void setPrev(Instruction *N) { Prev = N; }

  friend class SymbolTableListTraits<Instruction, BasicBlock, Function,
                                     ilist_traits<Instruction> >;
  void setParent(BasicBlock *P);
protected:
  unsigned iType;      // InstructionType: The opcode of the instruction

  Instruction(const Type *Ty, unsigned iType, const std::string &Name = "",
              Instruction *InsertBefore = 0);
public:
  virtual ~Instruction() {
    assert(Parent == 0 && "Instruction still embedded in basic block!");
  }

  // Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);
  
  /// clone() - Create a copy of 'this' instruction that is identical in all
  /// ways except the following:
  ///   * The instruction has no parent
  ///   * The instruction has no name
  ///
  virtual Instruction *clone() const = 0;
  
  // Accessor methods...
  //
  inline const BasicBlock *getParent() const { return Parent; }
  inline       BasicBlock *getParent()       { return Parent; }

  // getNext/Prev - Return the next or previous instruction in the list.  The
  // last node in the list is a terminator instruction.
        Instruction *getNext()       { return Next; }
  const Instruction *getNext() const { return Next; }
        Instruction *getPrev()       { return Prev; }
  const Instruction *getPrev() const { return Prev; }

  /// mayWriteToMemory - Return true if this instruction may modify memory.
  ///
  virtual bool mayWriteToMemory() const { return false; }

  // ---------------------------------------------------------------------------
  /// Subclass classification... getOpcode() returns a member of 
  /// one of the enums that is coming soon (down below)...
  ///
  unsigned getOpcode() const { return iType; }
  virtual const char *getOpcodeName() const {
    return getOpcodeName(getOpcode());
  }
  static const char* getOpcodeName(unsigned OpCode);

  inline bool isTerminator() const {   // Instance of TerminatorInst?
    return iType >= TermOpsBegin && iType < TermOpsEnd;
  }
  inline bool isBinaryOp() const {
    return iType >= BinaryOpsBegin && iType < BinaryOpsEnd;
  }

  /// isAssociative - Return true if the instruction is associative:
  ///
  ///   Associative operators satisfy:  x op (y op z) === (x op y) op z)
  ///
  /// In LLVM, the Add, Mul, And, Or, and Xor operators are associative, when
  /// not applied to floating point types.
  ///
  bool isAssociative() const { return isAssociative(getOpcode(), getType()); }
  static bool isAssociative(unsigned op, const Type *Ty);

  /// isCommutative - Return true if the instruction is commutative:
  ///
  ///   Commutative operators satistify: (x op y) === (y op x)
  ///
  /// In LLVM, these are the associative operators, plus SetEQ and SetNE, when
  /// applied to any type.
  ///
  bool isCommutative() const { return isCommutative(getOpcode()); }
  static bool isCommutative(unsigned op);


  virtual void print(std::ostream &OS) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::InstructionVal;
  }
  
  //----------------------------------------------------------------------
  // Exported enumerations...
  //
  enum TermOps {       // These terminate basic blocks
#define  FIRST_TERM_INST(N)             TermOpsBegin = N,
#define HANDLE_TERM_INST(N, OPC, CLASS) OPC = N,
#define   LAST_TERM_INST(N)             TermOpsEnd = N+1,
#include "llvm/Instruction.def"
  };

  enum BinaryOps {
#define  FIRST_BINARY_INST(N)             BinaryOpsBegin = N,
#define HANDLE_BINARY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_BINARY_INST(N)             BinaryOpsEnd = N+1,
#include "llvm/Instruction.def"
  };

  enum MemoryOps {
#define  FIRST_MEMORY_INST(N)             MemoryOpsBegin = N,
#define HANDLE_MEMORY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_MEMORY_INST(N)             MemoryOpsEnd = N+1,
#include "llvm/Instruction.def"
  };

  enum OtherOps {
#define  FIRST_OTHER_INST(N)             OtherOpsBegin = N,
#define HANDLE_OTHER_INST(N, OPC, CLASS) OPC = N,
#define   LAST_OTHER_INST(N)             OtherOpsEnd = N+1,
#include "llvm/Instruction.def"
  };
};

#endif
