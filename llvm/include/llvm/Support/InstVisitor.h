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
//    CMV.visit(method);
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

template<typename SubClass>
struct InstVisitor {
  ~InstVisitor() {}           // We are meant to be derived from

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

  // Define visitors for modules, methods and basic blocks...
  //
  void visit(Module *M) { visit(M->begin(), M->end()); }
  void visit(Method *M) { visit(M->begin(), M->end()); }
  void visit(BasicBlock *BB) { visit(BB->begin(), BB->end()); }

  // visit - Finally, code to visit an instruction...
  //
  void visit(Instruction *I) {
    switch (I->getOpcode()) {
      // Build the switch statement using the Instruction.def file...
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE: ((SubClass*)this)->visit##CLASS((CLASS*)I); return;
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
  
  // Specific Instruction type classes... note that all of the casts are
  // neccesary because we use the instruction classes as opaque types...
  //
  void visitReturnInst(ReturnInst *I)               { ((SubClass*)this)->visitTerminatorInst((TerminatorInst*)I); }
  void visitBranchInst(BranchInst *I)               { ((SubClass*)this)->visitTerminatorInst((TerminatorInst*)I); }
  void visitSwitchInst(SwitchInst *I)               { ((SubClass*)this)->visitTerminatorInst((TerminatorInst*)I); }
  void visitInvokeInst(InvokeInst *I)               { ((SubClass*)this)->visitTerminatorInst((TerminatorInst*)I); }
  void visitGenericUnaryInst(GenericUnaryInst  *I)  { ((SubClass*)this)->visitUnaryOperator((UnaryOperator*)I); }
  void visitGenericBinaryInst(GenericBinaryInst *I) { ((SubClass*)this)->visitBinaryOperator((BinaryOperator*)I); }
  void visitSetCondInst(SetCondInst *I)             { ((SubClass*)this)->visitBinaryOperator((BinaryOperator *)I); }
  void visitMallocInst(MallocInst *I)               { ((SubClass*)this)->visitAllocationInst((AllocationInst *)I); }
  void visitAllocaInst(AllocaInst *I)               { ((SubClass*)this)->visitAllocationInst((AllocationInst *)I); }
  void visitFreeInst(FreeInst   *I)                 { ((SubClass*)this)->visitInstruction((Instruction *)I); }
  void visitLoadInst(LoadInst   *I)                 { ((SubClass*)this)->visitMemAccessInst((MemAccessInst *)I); }
  void visitStoreInst(StoreInst  *I)                { ((SubClass*)this)->visitMemAccessInst((MemAccessInst *)I); }
  void visitGetElementPtrInst(GetElementPtrInst *I) { ((SubClass*)this)->visitMemAccessInst((MemAccessInst *)I); }
  void visitPHINode(PHINode    *I)                  { ((SubClass*)this)->visitInstruction((Instruction *)I); }
  void visitCastInst(CastInst   *I)                 { ((SubClass*)this)->visitInstruction((Instruction *)I); }
  void visitCallInst(CallInst   *I)                 { ((SubClass*)this)->visitInstruction((Instruction *)I); }
  void visitShiftInst(ShiftInst  *I)                { ((SubClass*)this)->visitInstruction((Instruction *)I); }

  // Next level propogators... if the user does not overload a specific
  // instruction type, they can overload one of these to get the whole class
  // of instructions...
  //
  void visitTerminatorInst(TerminatorInst *I) { ((SubClass*)this)->visitInstruction((Instruction*)I); }
  void visitUnaryOperator (UnaryOperator  *I) { ((SubClass*)this)->visitInstruction((Instruction*)I); }
  void visitBinaryOperator(BinaryOperator *I) { ((SubClass*)this)->visitInstruction((Instruction*)I); }
  void visitAllocationInst(AllocationInst *I) { ((SubClass*)this)->visitInstruction((Instruction*)I); }
  void visitMemAccessInst (MemAccessInst  *I) { ((SubClass*)this)->visitInstruction((Instruction*)I); }

  // If the user wants a 'default' case, they can choose to override this
  // function.  If this function is not overloaded in the users subclass, then
  // this instruction just gets ignored.
  //
  void visitInstruction(Instruction *I) {}  // Ignore unhandled instructions
};

#endif
