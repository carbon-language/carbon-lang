//===-- llvm/Assembly/Writer.h - Printer for VM assembly files ---*- C++ -*--=//
//
// This functionality is implemented by the lib/Assembly/Writer library.
// This library is used to print VM assembly language files to an iostream. It
// can print VM code at a variety of granularities, ranging from a whole class
// down to an individual instruction.  This makes it useful for debugging.
//
// This file also defines functions that allow it to output files that a program
// called VCG can read.
//
// This library uses the Analysis library to figure out offsets for
// variables in the method tables...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_WRITER_H
#define LLVM_ASSEMBLY_WRITER_H

#include <iostream>
#include "llvm/Type.h"

class SlotCalculator;

// The only interface defined by this file... convert the internal 
// representation of an object into an ascii bytestream that the parser can 
// understand later... (the parser only understands whole classes though)
//
void WriteToAssembly(const Module  *Module, std::ostream &o);
void WriteToAssembly(const GlobalVariable *G, std::ostream &o);
void WriteToAssembly(const Function    *F , std::ostream &o);
void WriteToAssembly(const BasicBlock  *BB, std::ostream &o);
void WriteToAssembly(const Instruction *In, std::ostream &o);
void WriteToAssembly(const Constant     *V, std::ostream &o);

// WriteTypeSymbolic - This attempts to write the specified type as a symbolic
// type, iff there is an entry in the modules symbol table for the specified
// type or one of it's component types.  This is slower than a simple x << Type;
//
std::ostream &WriteTypeSymbolic(std::ostream &, const Type *, const Module *M);


// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.
//
std::ostream &WriteAsOperand(std::ostream &, const Value *, bool PrintTy = true,
                             bool PrintName = true, SlotCalculator *Table = 0);


// Define operator<< to work on the various classes that we can send to an 
// ostream...
//
inline std::ostream &operator<<(std::ostream &o, const Module *C) {
  WriteToAssembly(C, o); return o;
}

inline std::ostream &operator<<(std::ostream &o, const GlobalVariable *G) {
  WriteToAssembly(G, o); return o;
}

inline std::ostream &operator<<(std::ostream &o, const Function *F) {
  WriteToAssembly(F, o); return o;
}

inline std::ostream &operator<<(std::ostream &o, const BasicBlock *B) {
  WriteToAssembly(B, o); return o;
}

inline std::ostream &operator<<(std::ostream &o, const Instruction *I) {
  WriteToAssembly(I, o); return o;
}

inline std::ostream &operator<<(std::ostream &o, const Constant *I) {
  WriteToAssembly(I, o); return o;
}


inline std::ostream &operator<<(std::ostream &o, const Type *T) {
  if (!T) return o << "<null Type>";
  return o << T->getDescription();
}

inline std::ostream &operator<<(std::ostream &o, const Value *I) {
  switch (I->getValueType()) {
  case Value::TypeVal:       return o << cast<const Type>(I);
  case Value::ConstantVal:   WriteToAssembly(cast<Constant>(I)      , o); break;
  case Value::FunctionArgumentVal:
    return o << I->getType() << " " << I->getName();
  case Value::InstructionVal:WriteToAssembly(cast<Instruction>(I)   , o); break;
  case Value::BasicBlockVal: WriteToAssembly(cast<BasicBlock>(I)    , o); break;
  case Value::FunctionVal:   WriteToAssembly(cast<Function>(I)      , o); break;
  case Value::GlobalVariableVal:
                             WriteToAssembly(cast<GlobalVariable>(I), o); break;
  case Value::ModuleVal:     WriteToAssembly(cast<Module>(I)        , o); break;
  default: return o << "<unknown value type: " << I->getValueType() << ">";
  }
  return o;
}

#endif
