//===-- llvm/Support/CodeGen.h - CodeGen Concepts ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file define some types which define code generation concepts. For
// example, relocation model.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CODEGEN_H
#define LLVM_SUPPORT_CODEGEN_H

namespace llvm {

  // Relocation model types.
  namespace Reloc {
    enum Model { Default, Static, PIC_, DynamicNoPIC };
  }

  // Code model types.
  namespace CodeModel {
    enum Model { Default, JITDefault, Small, Kernel, Medium, Large };
  }

}  // end llvm namespace

#endif
