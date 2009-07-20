//===-- llvm/MC/MCAsmLexer.h - Abstract Asm Lexer Interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMLEXER_H
#define LLVM_MC_MCASMLEXER_H

namespace llvm {
class MCAsmLexer;
class MCInst;
class Target;

/// MCAsmLexer - Generic assembler lexer interface, for use by target specific
/// assembly lexers.
class MCAsmLexer {
  MCAsmLexer(const MCAsmLexer &);   // DO NOT IMPLEMENT
  void operator=(const MCAsmLexer &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  MCAsmLexer();
 
public:
  virtual ~MCAsmLexer();
};

} // End llvm namespace

#endif
