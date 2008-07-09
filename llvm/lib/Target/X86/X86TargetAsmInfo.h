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

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class X86TargetMachine;
  class GlobalVariable;

  struct X86TargetAsmInfo : public TargetAsmInfo {
    explicit X86TargetAsmInfo(const X86TargetMachine &TM);

    virtual bool ExpandInlineAsm(CallInst *CI) const;
    virtual std::string SectionForGlobal(const GlobalValue *GV) const;

  private:
    bool LowerToBSwap(CallInst *CI) const;
  protected:
    const X86TargetMachine* X86TM;
  };

  struct X86DarwinTargetAsmInfo : public X86TargetAsmInfo {
    explicit X86DarwinTargetAsmInfo(const X86TargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
    virtual std::string UniqueSectionForGlobal(const GlobalValue* GV,
                                               SectionKind::Kind kind) const;
  };

  struct X86ELFTargetAsmInfo : public X86TargetAsmInfo {
    explicit X86ELFTargetAsmInfo(const X86TargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;

    virtual std::string SelectSectionForGlobal(const GlobalValue *GV) const;
    virtual unsigned SectionFlagsForGlobal(const GlobalValue *GV,
                                           const char* name) const;
    virtual std::string PrintSectionFlags(unsigned flags) const;
    std::string MergeableConstSection(const GlobalVariable *GV) const;
    std::string MergeableStringSection(const GlobalVariable *GV) const;
  };

  struct X86COFFTargetAsmInfo : public X86TargetAsmInfo {
    explicit X86COFFTargetAsmInfo(const X86TargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
    virtual std::string UniqueSectionForGlobal(const GlobalValue* GV,
                                               SectionKind::Kind kind) const;
    virtual std::string PrintSectionFlags(unsigned flags) const;
  };

  struct X86WinTargetAsmInfo : public X86TargetAsmInfo {
    explicit X86WinTargetAsmInfo(const X86TargetMachine &TM);
  };
} // namespace llvm

#endif
