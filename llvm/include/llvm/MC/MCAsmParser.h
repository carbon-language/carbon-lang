//===-- llvm/MC/MCAsmParser.h - Abstract Asm Parser Interface ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMPARSER_H
#define LLVM_MC_MCASMPARSER_H

namespace llvm {
class MCAsmParser;
class MCInst;
class Target;
class TargetAsmParser;

/// MCAsmParser - Generic assembler parser interface, for use by target specific
/// assembly parsers.
class MCAsmParser {
  MCAsmParser(const MCAsmParser &);   // DO NOT IMPLEMENT
  void operator=(const MCAsmParser &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  MCAsmParser();
 
public:
  virtual ~MCAsmParser();
};

} // End llvm namespace

#endif
