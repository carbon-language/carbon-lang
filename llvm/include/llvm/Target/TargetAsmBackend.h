//===-- llvm/Target/TargetAsmBackend.h - Target Asm Backend -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETASMBACKEND_H
#define LLVM_TARGET_TARGETASMBACKEND_H

namespace llvm {
class Target;

/// TargetAsmBackend - Generic interface to target specific assembler backends.
class TargetAsmBackend {
  TargetAsmBackend(const TargetAsmBackend &);   // DO NOT IMPLEMENT
  void operator=(const TargetAsmBackend &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  TargetAsmBackend(const Target &);

  /// TheTarget - The Target that this machine was created for.
  const Target &TheTarget;

public:
  virtual ~TargetAsmBackend();

  const Target &getTarget() const { return TheTarget; }

};

} // End llvm namespace

#endif
