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

class Module;
class Method;
class BasicBlock;
class Instruction;
class SlotCalculator;

// The only interface defined by this file... convert the internal 
// representation of an object into an ascii bytestream that the parser can 
// understand later... (the parser only understands whole classes though)
//
void WriteToAssembly(const Module  *Module, ostream &o);
void WriteToAssembly(const Method  *Method, ostream &o);
void WriteToAssembly(const BasicBlock  *BB, ostream &o);
void WriteToAssembly(const Instruction *In, ostream &o);
void WriteToAssembly(const ConstPoolVal *V, ostream &o);

// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.
//
ostream &WriteAsOperand(ostream &o, const Value *V, bool PrintType = true,
                       	bool PrintName = true, SlotCalculator *Table = 0);


// WriteToVCG - Dump the specified structure to a VCG file.  If method is
// dumped, then the file named is created.  If a module is to be written, a
// family of files with a common base name is created, with a method name
// suffix.
//
void WriteToVCG(const Module *Module, const string &Filename);
void WriteToVCG(const Method *Method, const string &Filename);




// Define operator<< to work on the various classes that we can send to an 
// ostream...
//
inline ostream &operator<<(ostream &o, const Module *C) {
  WriteToAssembly(C, o); return o;
}

inline ostream &operator<<(ostream &o, const Method *M) {
  WriteToAssembly(M, o); return o;
}

inline ostream &operator<<(ostream &o, const BasicBlock *B) {
  WriteToAssembly(B, o); return o;
}

inline ostream &operator<<(ostream &o, const Instruction *I) {
  WriteToAssembly(I, o); return o;
}

inline ostream &operator<<(ostream &o, const ConstPoolVal *I) {
  WriteToAssembly(I, o); return o;
}


inline ostream &operator<<(ostream &o, const Type *T) {
  if (!T) return o << "<null Type>";
  return o << T->getName();
}

inline ostream &operator<<(ostream &o, const Value *I) {
  switch (I->getValueType()) {
  case Value::TypeVal:        return o << (const Type*)I;
  case Value::ConstantVal:    WriteToAssembly((const ConstPoolVal*)I, o); break;
  case Value::MethodArgumentVal: return o <<I->getType() << " " << I->getName();
  case Value::InstructionVal: WriteToAssembly((const Instruction *)I, o); break;
  case Value::BasicBlockVal:  WriteToAssembly((const BasicBlock  *)I, o); break;
  case Value::MethodVal:      WriteToAssembly((const Method      *)I, o); break;
  case Value::ModuleVal:      WriteToAssembly((const Module      *)I, o); break;
  default: return o << "<unknown value type: " << I->getValueType() << ">";
  }
  return o;
}

void DebugValue(const Value *V);

#endif
