//===-- llvm/IR/AsmWriter.h - Printing LLVM IR as an assembly file - C++ --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This files defines the interface for the AssemblyWriter class used to print
// LLVM IR and various helper classes that are used in printing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_ASSEMBLYWRITER_H
#define LLVM_IR_ASSEMBLYWRITER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {

class BasicBlock;
class Function;
class GlobalValue;
class Module;
class NamedMDNode;
class Value;
class SlotTracker;

/// Create a new SlotTracker for a Module 
SlotTracker *createSlotTracker(const Module *M);

//===----------------------------------------------------------------------===//
// TypePrinting Class: Type printing machinery
//===----------------------------------------------------------------------===//

class TypePrinting {
  TypePrinting(const TypePrinting &) LLVM_DELETED_FUNCTION;
  void operator=(const TypePrinting&) LLVM_DELETED_FUNCTION;
public:

  /// NamedTypes - The named types that are used by the current module.
  TypeFinder NamedTypes;

  /// NumberedTypes - The numbered types, along with their value.
  DenseMap<StructType*, unsigned> NumberedTypes;


  TypePrinting() {}
  ~TypePrinting() {}

  void incorporateTypes(const Module &M);

  void print(Type *Ty, raw_ostream &OS);

  void printStructBody(StructType *Ty, raw_ostream &OS);
};

class AssemblyWriter {
protected:
  formatted_raw_ostream &Out;
  const Module *TheModule;

private:
  OwningPtr<SlotTracker> ModuleSlotTracker;
  SlotTracker &Machine;
  TypePrinting TypePrinter;
  AssemblyAnnotationWriter *AnnotationWriter;

public:
  /// Construct an AssemblyWriter with an external SlotTracker
  AssemblyWriter(formatted_raw_ostream &o, SlotTracker &Mac,
                 const Module *M, AssemblyAnnotationWriter *AAW);

  /// Construct an AssemblyWriter with an internally allocated SlotTracker
  AssemblyWriter(formatted_raw_ostream &o, const Module *M,
                 AssemblyAnnotationWriter *AAW);

  virtual ~AssemblyWriter();

  void printMDNodeBody(const MDNode *MD);
  void printNamedMDNode(const NamedMDNode *NMD);

  void printModule(const Module *M);

  void writeOperand(const Value *Op, bool PrintType);
  void writeParamOperand(const Value *Operand, AttributeSet Attrs,unsigned Idx);
  void writeAtomic(AtomicOrdering Ordering, SynchronizationScope SynchScope);

  void writeAllMDNodes();
  void writeMDNode(unsigned Slot, const MDNode *Node);
  void writeAllAttributeGroups();

  void printTypeIdentities();
  void printGlobal(const GlobalVariable *GV);
  void printAlias(const GlobalAlias *GV);
  void printFunction(const Function *F);
  void printArgument(const Argument *FA, AttributeSet Attrs, unsigned Idx);
  void printBasicBlock(const BasicBlock *BB);
  void printInstructionLine(const Instruction &I);
  void printInstruction(const Instruction &I);

private:
  void init();

  // printInfoComment - Print a little comment after the instruction indicating
  // which slot it occupies.
  void printInfoComment(const Value &V);
};

} // namespace llvm

#endif //LLVM_IR_ASMWRITER_H
