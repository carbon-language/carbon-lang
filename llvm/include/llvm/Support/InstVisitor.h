//===- llvm/Support/InstVisitor.h - Define instruction visitors --*- C++ -*--=//
//
// This template class is used to define instruction visitors in a typesafe
// manner without having to use lots of casts and a big switch statement (in
// your code that is).  The win here is that if instructions are added in the
// future, they will be added to the InstVisitor<T> class, allowing you to
// automatically support them (if you handle on of their superclasses).
//
// Note that this library is specifically designed as a template to avoid
// virtual function call overhead.  Defining and using an InstVisitor is just as
// efficient as having your own switch statement over the instruction opcode.
//
// InstVisitor Usage:
//   You define InstVisitors from inheriting from the InstVisitor base class
// and "overriding" functions in your class.  I say "overriding" because this
// class is defined in terms of statically resolved overloading, not virtual
// functions.  As an example, here is a visitor that counts the number of malloc
// instructions processed:
//
//  // Declare the class.  Note that we derive from InstVisitor instantiated
//  // with _our new subclasses_ type.
//  //
//  struct CountMallocVisitor : public InstVisitor<CountMallocVisitor> {
//    unsigned Count;
//    CountMallocVisitor() : Count(0) {}
//
//    void visitMallocInst(MallocInst *MI) { ++Count; }
//  };
//
//  And this class would be used like this:
//    CountMallocVistor CMV;
//    CMV.visit(function);
//    NumMallocs = CMV.Count;
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_INSTVISITOR_H
#define LLVM_SUPPORT_INSTVISITOR_H

#include "llvm/Instruction.h"

// We operate on opaque instruction classes, so forward declare all instruction
// types now...
//
#define HANDLE_INST(NUM, OPCODE, CLASS)   class CLASS;
#include "llvm/Instruction.def"

// Forward declare the intermediate types...
class TerminatorInst; class UnaryOperator; class BinaryOperator;
class AllocationInst; class MemAccessInst;

template<typename SubClass, typename RetTy=void>
struct InstVisitor {
  virtual ~InstVisitor() {}           // We are meant to be derived from

  //===--------------------------------------------------------------------===//
  // Interface code - This is the public interface of the InstVisitor that you
  // use to visit instructions...
  //

  // Generic visit method - Allow visitation to all instructions in a range
  template<class Iterator>
  void visit(Iterator Start, Iterator End) {
    while (Start != End)
      visit(*Start++);
  }

  // Define visitors for modules, functions and basic blocks...
  //
  void visit(Module *M) {
    ((SubClass*)this)->visitModule(M);
    visit(M->begin(), M->end());
  }
  void visit(Function *F) {
    ((SubClass*)this)->visitFunction(F);
    visit(F->begin(), F->end());
  }
  void visit(BasicBlock *BB) {
    ((SubClass*)this)->visitBasicBlock(BB);
    visit(BB->begin(), BB->end());
  }

  // visit - Finally, code to visit an instruction...
  //
  RetTy visit(Instruction *I) {
    switch (I->getOpcode()) {
      // Build the switch statement using the Instruction.def file...
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE: return ((SubClass*)this)->visit##CLASS((CLASS*)I);
#include "llvm/Instruction.def"

    default: assert(0 && "Unknown instruction type encountered!");
    }
  }

  //===--------------------------------------------------------------------===//
  // Visitation functions... these functions provide default fallbacks in case
  // the user does not specify what to do for a particular instruction type.
  // The default behavior is to generalize the instruction type to its subtype
  // and try visiting the subtype.  All of this should be inlined perfectly,
  // because there are no virtual functions to get in the way.
  //

  // When visiting a module, function or basic block directly, these methods get
  // called to indicate when transitioning into a new unit.
  //
  void visitModule    (Module *M) {}
  void visitFunction  (Function *F) {}
  void visitBasicBlock(BasicBlock *BB) {}
  
  // Specific Instruction type classes... note that all of the casts are
  // neccesary because we use the instruction classes as opaque types...
  //
  RetTy visitReturnInst(ReturnInst *I)               { return ((SubClass*)this)->visitTerminatorInst((TerminatorInst*)I); }
  RetTy visitBranchInst(BranchInst *I)               { return ((SubClass*)this)->visitTerminatorInst((TerminatorInst*)I); }
  RetTy visitSwitchInst(SwitchInst *I)               { return ((SubClass*)this)->visitTerminatorInst((TerminatorInst*)I); }
  RetTy visitInvokeInst(InvokeInst *I)               { return ((SubClass*)this)->visitTerminatorInst((TerminatorInst*)I); }
  RetTy visitGenericUnaryInst(GenericUnaryInst  *I)  { return ((SubClass*)this)->visitUnaryOperator((UnaryOperator*)I); }
  RetTy visitGenericBinaryInst(GenericBinaryInst *I) { return ((SubClass*)this)->visitBinaryOperator((BinaryOperator*)I); }
  RetTy visitSetCondInst(SetCondInst *I)             { return ((SubClass*)this)->visitBinaryOperator((BinaryOperator *)I); }
  RetTy visitMallocInst(MallocInst *I)               { return ((SubClass*)this)->visitAllocationInst((AllocationInst *)I); }
  RetTy visitAllocaInst(AllocaInst *I)               { return ((SubClass*)this)->visitAllocationInst((AllocationInst *)I); }
  RetTy visitFreeInst(FreeInst   *I)                 { return ((SubClass*)this)->visitInstruction((Instruction *)I); }
  RetTy visitLoadInst(LoadInst   *I)                 { return ((SubClass*)this)->visitMemAccessInst((MemAccessInst *)I); }
  RetTy visitStoreInst(StoreInst  *I)                { return ((SubClass*)this)->visitMemAccessInst((MemAccessInst *)I); }
  RetTy visitGetElementPtrInst(GetElementPtrInst *I) { return ((SubClass*)this)->visitMemAccessInst((MemAccessInst *)I); }
  RetTy visitPHINode(PHINode    *I)                  { return ((SubClass*)this)->visitInstruction((Instruction *)I); }
  RetTy visitCastInst(CastInst   *I)                 { return ((SubClass*)this)->visitInstruction((Instruction *)I); }
  RetTy visitCallInst(CallInst   *I)                 { return ((SubClass*)this)->visitInstruction((Instruction *)I); }
  RetTy visitShiftInst(ShiftInst  *I)                { return ((SubClass*)this)->visitInstruction((Instruction *)I); }

  // Next level propogators... if the user does not overload a specific
  // instruction type, they can overload one of these to get the whole class
  // of instructions...
  //
  RetTy visitTerminatorInst(TerminatorInst *I) { return ((SubClass*)this)->visitInstruction((Instruction*)I); }
  RetTy visitUnaryOperator (UnaryOperator  *I) { return ((SubClass*)this)->visitInstruction((Instruction*)I); }
  RetTy visitBinaryOperator(BinaryOperator *I) { return ((SubClass*)this)->visitInstruction((Instruction*)I); }
  RetTy visitAllocationInst(AllocationInst *I) { return ((SubClass*)this)->visitInstruction((Instruction*)I); }
  RetTy visitMemAccessInst (MemAccessInst  *I) { return ((SubClass*)this)->visitInstruction((Instruction*)I); }

  // If the user wants a 'default' case, they can choose to override this
  // function.  If this function is not overloaded in the users subclass, then
  // this instruction just gets ignored.
  //
  // Note that you MUST override this function if your return type is not void.
  //
  void visitInstruction(Instruction *I) {}  // Ignore unhandled instructions
};

#endif
