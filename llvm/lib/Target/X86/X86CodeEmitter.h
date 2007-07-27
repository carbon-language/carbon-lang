//===-- X86CodeEmitter.h - X86 DAG Lowering Interface -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duncan Sands and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for X86 code emission.
//
//===----------------------------------------------------------------------===//

#ifndef X86CODEEMITTER_H
#define X86CODEEMITTER_H

/// N86 namespace - Native X86 Register numbers... used by X86 backend.
///
namespace N86 {
  enum {
    EAX = 0, ECX = 1, EDX = 2, EBX = 3, ESP = 4, EBP = 5, ESI = 6, EDI = 7
  };
}

#endif    // X86CODEEMITTER_H
