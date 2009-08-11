//=====-- ARMTargetAsmInfo.h - ARM asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ARMTARGETASMINFO_H
#define LLVM_ARMTARGETASMINFO_H

#include "llvm/Target/DarwinTargetAsmInfo.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

  extern const char *const arm_asm_table[];

  template <class BaseTAI>
  struct ARMTargetAsmInfo : public BaseTAI {
    explicit ARMTargetAsmInfo() {
      BaseTAI::AsmTransCBE = arm_asm_table;

      BaseTAI::AlignmentIsInBytes = false;
      BaseTAI::Data64bitsDirective = 0;
      BaseTAI::CommentString = "@";
      BaseTAI::COMMDirectiveTakesAlignment = false;
      BaseTAI::InlineAsmStart = "@ InlineAsm Start";
      BaseTAI::InlineAsmEnd = "@ InlineAsm End";
    }
  };

  EXTERN_TEMPLATE_INSTANTIATION(class ARMTargetAsmInfo<DarwinTargetAsmInfo>);
  EXTERN_TEMPLATE_INSTANTIATION(class ARMTargetAsmInfo<TargetAsmInfo>);

  struct ARMDarwinTargetAsmInfo : public ARMTargetAsmInfo<DarwinTargetAsmInfo> {
    explicit ARMDarwinTargetAsmInfo();
  };

  struct ARMELFTargetAsmInfo : public ARMTargetAsmInfo<TargetAsmInfo> {
    explicit ARMELFTargetAsmInfo();
  };

} // namespace llvm

#endif
