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
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"
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

  typedef X86TargetAsmInfo<TargetAsmInfo> X86GenericTargetAsmInfo;

  EXTERN_TEMPLATE_INSTANTIATION(class X86TargetAsmInfo<TargetAsmInfo>);

  struct X86DarwinTargetAsmInfo : public X86TargetAsmInfo<DarwinTargetAsmInfo> {
    explicit X86DarwinTargetAsmInfo(const X86TargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
    virtual const char *getEHGlobalPrefix() const;
  };

  struct X86ELFTargetAsmInfo : public X86TargetAsmInfo<ELFTargetAsmInfo> {
    explicit X86ELFTargetAsmInfo(const X86TargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
  };

  struct X86COFFTargetAsmInfo : public X86GenericTargetAsmInfo {
    explicit X86COFFTargetAsmInfo(const X86TargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
    virtual const char *
    getSectionPrefixForUniqueGlobal(SectionKind::Kind kind) const;
    virtual std::string printSectionFlags(unsigned flags) const;
  };

  struct X86WinTargetAsmInfo : public X86GenericTargetAsmInfo {
    explicit X86WinTargetAsmInfo(const X86TargetMachine &TM);
  };

} // namespace llvm

#endif
