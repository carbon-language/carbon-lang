//=====-- X86TargetAsmInfo.h - X86 asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the X86TargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86TARGETASMINFO_H
#define X86TARGETASMINFO_H

#include "X86TargetMachine.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/COFFTargetAsmInfo.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

  extern const char *const x86_asm_table[];

  template <class BaseTAI>
  struct X86TargetAsmInfo : public BaseTAI {
    explicit X86TargetAsmInfo(const X86TargetMachine &TM) : BaseTAI(TM) {
      BaseTAI::AsmTransCBE = x86_asm_table;
      BaseTAI::AssemblerDialect =TM.getSubtarget<X86Subtarget>().getAsmFlavor();
    }
  };

  EXTERN_TEMPLATE_INSTANTIATION(class X86TargetAsmInfo<TargetAsmInfo>);

  struct X86DarwinTargetAsmInfo : public X86TargetAsmInfo<DarwinTargetAsmInfo> {
    explicit X86DarwinTargetAsmInfo(const X86TargetMachine &TM);
  };

  struct X86ELFTargetAsmInfo : public X86TargetAsmInfo<ELFTargetAsmInfo> {
    explicit X86ELFTargetAsmInfo(const X86TargetMachine &TM);
  };

  typedef X86TargetAsmInfo<COFFTargetAsmInfo> X86COFFTargetAsmInfo;

  struct X86WinTargetAsmInfo : public X86TargetAsmInfo<TargetAsmInfo> {
    explicit X86WinTargetAsmInfo(const X86TargetMachine &TM);
  };

} // namespace llvm

#endif
