//===-- llvm/Assembly/Writer.h - Printer for LLVM assembly files --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This functionality is implemented by lib/VMCore/AsmWriter.cpp.
// This library is used to print LLVM assembly language files to an iostream. It
// can print LLVM code at a variety of granularities, including Modules,
// BasicBlocks, and Instructions.  This makes it useful for debugging.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_WRITER_H
#define LLVM_ASSEMBLY_WRITER_H

#include <iosfwd>
#include <string>

namespace llvm {

class Type;
class Module;
class Value;
class raw_ostream;
template <typename T> class SmallVectorImpl;
  
/// TypePrinting - Type printing machinery.
class TypePrinting {
  void *TypeNames;  // A map to remember type names.
  TypePrinting(const TypePrinting &);   // DO NOT IMPLEMENT
  void operator=(const TypePrinting&);  // DO NOT IMPLEMENT
public:
  TypePrinting();
  ~TypePrinting();
  
  void clear();
  
  void print(const Type *Ty, raw_ostream &OS, bool IgnoreTopLevelName = false);
  
  void printAtLeastOneLevel(const Type *Ty, raw_ostream &OS) {
    print(Ty, OS, true);
  }
  
  /// hasTypeName - Return true if the type has a name in TypeNames, false
  /// otherwise.
  bool hasTypeName(const Type *Ty) const;
  
  /// addTypeName - Add a name for the specified type if it doesn't already have
  /// one.  This name will be printed instead of the structural version of the
  /// type in order to make the output more concise.
  void addTypeName(const Type *Ty, const std::string &N);
  
private:
  void CalcTypeName(const Type *Ty, SmallVectorImpl<const Type *> &TypeStack,
                    raw_ostream &OS, bool IgnoreTopLevelName = false);
};

// WriteTypeSymbolic - This attempts to write the specified type as a symbolic
// type, if there is an entry in the Module's symbol table for the specified
// type or one of its component types.
//
void WriteTypeSymbolic(raw_ostream &, const Type *, const Module *M);

// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.  If you specify a Module for context,
// then even constants get pretty-printed; for example, the type of a null
// pointer is printed symbolically.
//
void WriteAsOperand(std::ostream &, const Value *, bool PrintTy = true,
                    const Module *Context = 0);
void WriteAsOperand(raw_ostream &, const Value *, bool PrintTy = true,
                    const Module *Context = 0);

} // End llvm namespace

#endif
