//===-- llvm/Target/TargetAsmLexer.h - Target Assembly Lexer ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETASMLEXER_H
#define LLVM_TARGET_TARGETASMLEXER_H

namespace llvm {
class Target;
  
/// TargetAsmLexer - Generic interface to target specific assembly lexers.
class TargetAsmLexer {
  TargetAsmLexer(const TargetAsmLexer &);   // DO NOT IMPLEMENT
  void operator=(const TargetAsmLexer &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  TargetAsmLexer(const Target &);
  
  /// TheTarget - The Target that this machine was created for.
  const Target &TheTarget;
  
public:
  virtual ~TargetAsmLexer();
  
  const Target &getTarget() const { return TheTarget; }
  
  
};

} // End llvm namespace

#endif
