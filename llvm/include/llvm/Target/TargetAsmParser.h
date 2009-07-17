//===-- llvm/Target/TargetAsmParser.h - Target Assembly Parser --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETPARSER_H
#define LLVM_TARGET_TARGETPARSER_H

namespace llvm {
class Target;

/// TargetAsmParser - Generic interface to target specific assembly parsers.
class TargetAsmParser {
  TargetAsmParser(const TargetAsmParser &);   // DO NOT IMPLEMENT
  void operator=(const TargetAsmParser &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  TargetAsmParser(const Target &);
 
  /// TheTarget - The Target that this machine was created for.
  const Target &TheTarget;

public:
  virtual ~TargetAsmParser();

  const Target &getTarget() const { return TheTarget; }
};

} // End llvm namespace

#endif
