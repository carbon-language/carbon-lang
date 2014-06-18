//===- ConstantPool.h - Keep track of assembler-generated  ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ConstantPool and AssemblerConstantPools classes.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_MC_CONSTANTPOOL_H
#define LLVM_MC_CONSTANTPOOL_H

#include "llvm/ADT/SmallVector.h"
namespace llvm {
class MCContext;
class MCExpr;
class MCSection;
class MCStreamer;
class MCSymbol;
// A class to keep track of assembler-generated constant pools that are use to
// implement the ldr-pseudo.
class ConstantPool {
  typedef SmallVector<std::pair<MCSymbol *, const MCExpr *>, 4> EntryVecTy;
  EntryVecTy Entries;

public:
  // Initialize a new empty constant pool
  ConstantPool() {}

  // Add a new entry to the constant pool in the next slot.
  // \param Value is the new entry to put in the constant pool.
  //
  // \returns a MCExpr that references the newly inserted value
  const MCExpr *addEntry(const MCExpr *Value, MCContext &Context);

  // Emit the contents of the constant pool using the provided streamer.
  void emitEntries(MCStreamer &Streamer);

  // Return true if the constant pool is empty
  bool empty();
};

class AssemblerConstantPools {
  // Map type used to keep track of per-Section constant pools used by the
  // ldr-pseudo opcode. The map associates a section to its constant pool. The
  // constant pool is a vector of (label, value) pairs. When the ldr
  // pseudo is parsed we insert a new (label, value) pair into the constant pool
  // for the current section and add MCSymbolRefExpr to the new label as
  // an opcode to the ldr. After we have parsed all the user input we
  // output the (label, value) pairs in each constant pool at the end of the
  // section.
  //
  // We use the MapVector for the map type to ensure stable iteration of
  // the sections at the end of the parse. We need to iterate over the
  // sections in a stable order to ensure that we have print the
  // constant pools in a deterministic order when printing an assembly
  // file.
  typedef MapVector<const MCSection *, ConstantPool> ConstantPoolMapTy;
  ConstantPoolMapTy ConstantPools;

public:
  AssemblerConstantPools() {}
  ~AssemblerConstantPools() {}

  void emitAll(MCStreamer &Streamer);
  void emitForCurrentSection(MCStreamer &Streamer);
  const MCExpr *addEntry(MCStreamer &Streamer, const MCExpr *Expr);

private:
  ConstantPool *getConstantPool(const MCSection *Section);
  ConstantPool &getOrCreateConstantPool(const MCSection *Section);
};
} // end namespace llvm

#endif
