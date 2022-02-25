//===-- X86Subtarget.cpp - X86 Subtarget Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "X86Subtarget.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "X86.h"
#include "X86CallLowering.h"
#include "X86LegalizerInfo.h"
#include "X86MacroFusion.h"
#include "X86RegisterBankInfo.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

using namespace llvm;

#define DEBUG_TYPE "subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "X86GenSubtargetInfo.inc"

// Temporary option to control early if-conversion for x86 while adding machine
// models.
static cl::opt<bool>
X86EarlyIfConv("x86-early-ifcvt", cl::Hidden,
               cl::desc("Enable early if-conversion on X86"));


/// Classify a blockaddress reference for the current subtarget according to how
/// we should reference it in a non-pcrel context.
unsigned char X86Subtarget::classifyBlockAddressReference() const {
  return classifyLocalReference(nullptr);
}

/// Classify a global variable reference for the current subtarget according to
/// how we should reference it in a non-pcrel context.
unsigned char
X86Subtarget::classifyGlobalReference(const GlobalValue *GV) const {
  return classifyGlobalReference(GV, *GV->getParent());
}

unsigned char
X86Subtarget::classifyLocalReference(const GlobalValue *GV) const {
  // If we're not PIC, it's not very interesting.
  if (!isPositionIndependent())
    return X86II::MO_NO_FLAG;

  if (is64Bit()) {
    // 64-bit ELF PIC local references may use GOTOFF relocations.
    if (isTargetELF()) {
      switch (TM.getCodeModel()) {
      // 64-bit small code model is simple: All rip-relative.
      case CodeModel::Tiny:
        llvm_unreachable("Tiny codesize model not supported on X86");
      case CodeModel::Small:
      case CodeModel::Kernel:
        return X86II::MO_NO_FLAG;

      // The large PIC code model uses GOTOFF.
      case CodeModel::Large:
        return X86II::MO_GOTOFF;

      // Medium is a hybrid: RIP-rel for code, GOTOFF for DSO local data.
      case CodeModel::Medium:
        // Constant pool and jump table handling pass a nullptr to this
        // function so we need to use isa_and_nonnull.
        if (isa_and_nonnull<Function>(GV))
          return X86II::MO_NO_FLAG; // All code is RIP-relative
        return X86II::MO_GOTOFF;    // Local symbols use GOTOFF.
      }
      llvm_unreachable("invalid code model");
    }

    // Otherwise, this is either a RIP-relative reference or a 64-bit movabsq,
    // both of which use MO_NO_FLAG.
    return X86II::MO_NO_FLAG;
  }

  // The COFF dynamic linker just patches the executable sections.
  if (isTargetCOFF())
    return X86II::MO_NO_FLAG;

  if (isTargetDarwin()) {
    // 32 bit macho has no relocation for a-b if a is undefined, even if
    // b is in the section that is being relocated.
    // This means we have to use o load even for GVs that are known to be
    // local to the dso.
    if (GV && (GV->isDeclarationForLinker() || GV->hasCommonLinkage()))
      return X86II::MO_DARWIN_NONLAZY_PIC_BASE;

    return X86II::MO_PIC_BASE_OFFSET;
  }

  return X86II::MO_GOTOFF;
}

unsigned char X86Subtarget::classifyGlobalReference(const GlobalValue *GV,
                                                    const Module &M) const {
  // The static large model never uses stubs.
  if (TM.getCodeModel() == CodeModel::Large && !isPositionIndependent())
    return X86II::MO_NO_FLAG;

  // Absolute symbols can be referenced directly.
  if (GV) {
    if (Optional<ConstantRange> CR = GV->getAbsoluteSymbolRange()) {
      // See if we can use the 8-bit immediate form. Note that some instructions
      // will sign extend the immediate operand, so to be conservative we only
      // accept the range [0,128).
      if (CR->getUnsignedMax().ult(128))
        return X86II::MO_ABS8;
      else
        return X86II::MO_NO_FLAG;
    }
  }

  if (TM.shouldAssumeDSOLocal(M, GV))
    return classifyLocalReference(GV);

  if (isTargetCOFF()) {
    // ExternalSymbolSDNode like _tls_index.
    if (!GV)
      return X86II::MO_NO_FLAG;
    if (GV->hasDLLImportStorageClass())
      return X86II::MO_DLLIMPORT;
    return X86II::MO_COFFSTUB;
  }
  // Some JIT users use *-win32-elf triples; these shouldn't use GOT tables.
  if (isOSWindows())
    return X86II::MO_NO_FLAG;

  if (is64Bit()) {
    // ELF supports a large, truly PIC code model with non-PC relative GOT
    // references. Other object file formats do not. Use the no-flag, 64-bit
    // reference for them.
    if (TM.getCodeModel() == CodeModel::Large)
      return isTargetELF() ? X86II::MO_GOT : X86II::MO_NO_FLAG;
    return X86II::MO_GOTPCREL;
  }

  if (isTargetDarwin()) {
    if (!isPositionIndependent())
      return X86II::MO_DARWIN_NONLAZY;
    return X86II::MO_DARWIN_NONLAZY_PIC_BASE;
  }

  // 32-bit ELF references GlobalAddress directly in static relocation model.
  // We cannot use MO_GOT because EBX may not be set up.
  if (TM.getRelocationModel() == Reloc::Static)
    return X86II::MO_NO_FLAG;
  return X86II::MO_GOT;
}

unsigned char
X86Subtarget::classifyGlobalFunctionReference(const GlobalValue *GV) const {
  return classifyGlobalFunctionReference(GV, *GV->getParent());
}

unsigned char
X86Subtarget::classifyGlobalFunctionReference(const GlobalValue *GV,
                                              const Module &M) const {
  if (TM.shouldAssumeDSOLocal(M, GV))
    return X86II::MO_NO_FLAG;

  // Functions on COFF can be non-DSO local for three reasons:
  // - They are intrinsic functions (!GV)
  // - They are marked dllimport
  // - They are extern_weak, and a stub is needed
  if (isTargetCOFF()) {
    if (!GV)
      return X86II::MO_NO_FLAG;
    if (GV->hasDLLImportStorageClass())
      return X86II::MO_DLLIMPORT;
    return X86II::MO_COFFSTUB;
  }

  const Function *F = dyn_cast_or_null<Function>(GV);

  if (isTargetELF()) {
    if (is64Bit() && F && (CallingConv::X86_RegCall == F->getCallingConv()))
      // According to psABI, PLT stub clobbers XMM8-XMM15.
      // In Regcall calling convention those registers are used for passing
      // parameters. Thus we need to prevent lazy binding in Regcall.
      return X86II::MO_GOTPCREL;
    // If PLT must be avoided then the call should be via GOTPCREL.
    if (((F && F->hasFnAttribute(Attribute::NonLazyBind)) ||
         (!F && M.getRtLibUseGOT())) &&
        is64Bit())
       return X86II::MO_GOTPCREL;
    // Reference ExternalSymbol directly in static relocation model.
    if (!is64Bit() && !GV && TM.getRelocationModel() == Reloc::Static)
      return X86II::MO_NO_FLAG;
    return X86II::MO_PLT;
  }

  if (is64Bit()) {
    if (F && F->hasFnAttribute(Attribute::NonLazyBind))
      // If the function is marked as non-lazy, generate an indirect call
      // which loads from the GOT directly. This avoids runtime overhead
      // at the cost of eager binding (and one extra byte of encoding).
      return X86II::MO_GOTPCREL;
    return X86II::MO_NO_FLAG;
  }

  return X86II::MO_NO_FLAG;
}

/// Return true if the subtarget allows calls to immediate address.
bool X86Subtarget::isLegalToCallImmediateAddr() const {
  // FIXME: I386 PE/COFF supports PC relative calls using IMAGE_REL_I386_REL32
  // but WinCOFFObjectWriter::RecordRelocation cannot emit them.  Once it does,
  // the following check for Win32 should be removed.
  if (In64BitMode || isTargetWin32())
    return false;
  return isTargetELF() || TM.getRelocationModel() == Reloc::Static;
}

void X86Subtarget::initSubtargetFeatures(StringRef CPU, StringRef TuneCPU,
                                         StringRef FS) {
  if (CPU.empty())
    CPU = "generic";

  if (TuneCPU.empty())
    TuneCPU = "i586"; // FIXME: "generic" is more modern than llc tests expect.

  std::string FullFS = X86_MC::ParseX86Triple(TargetTriple);
  assert(!FullFS.empty() && "Failed to parse X86 triple");

  if (!FS.empty())
    FullFS = (Twine(FullFS) + "," + FS).str();

  // Parse features string and set the CPU.
  ParseSubtargetFeatures(CPU, TuneCPU, FullFS);

  // All CPUs that implement SSE4.2 or SSE4A support unaligned accesses of
  // 16-bytes and under that are reasonably fast. These features were
  // introduced with Intel's Nehalem/Silvermont and AMD's Family10h
  // micro-architectures respectively.
  if (hasSSE42() || hasSSE4A())
    IsUAMem16Slow = false;

  LLVM_DEBUG(dbgs() << "Subtarget features: SSELevel " << X86SSELevel
                    << ", 3DNowLevel " << X863DNowLevel << ", 64bit "
                    << HasX86_64 << "\n");
  if (In64BitMode && !HasX86_64)
    report_fatal_error("64-bit code requested on a subtarget that doesn't "
                       "support it!");

  // Stack alignment is 16 bytes on Darwin, Linux, kFreeBSD, NaCl, and for all
  // 64-bit targets.  On Solaris (32-bit), stack alignment is 4 bytes
  // following the i386 psABI, while on Illumos it is always 16 bytes.
  if (StackAlignOverride)
    stackAlignment = *StackAlignOverride;
  else if (isTargetDarwin() || isTargetLinux() || isTargetKFreeBSD() ||
           isTargetNaCl() || In64BitMode)
    stackAlignment = Align(16);

  // Consume the vector width attribute or apply any target specific limit.
  if (PreferVectorWidthOverride)
    PreferVectorWidth = PreferVectorWidthOverride;
  else if (Prefer128Bit)
    PreferVectorWidth = 128;
  else if (Prefer256Bit)
    PreferVectorWidth = 256;
}

X86Subtarget &X86Subtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef TuneCPU,
                                                            StringRef FS) {
  initSubtargetFeatures(CPU, TuneCPU, FS);
  return *this;
}

X86Subtarget::X86Subtarget(const Triple &TT, StringRef CPU, StringRef TuneCPU,
                           StringRef FS, const X86TargetMachine &TM,
                           MaybeAlign StackAlignOverride,
                           unsigned PreferVectorWidthOverride,
                           unsigned RequiredVectorWidth)
    : X86GenSubtargetInfo(TT, CPU, TuneCPU, FS),
      PICStyle(PICStyles::Style::None), TM(TM), TargetTriple(TT),
      StackAlignOverride(StackAlignOverride),
      PreferVectorWidthOverride(PreferVectorWidthOverride),
      RequiredVectorWidth(RequiredVectorWidth),
      InstrInfo(initializeSubtargetDependencies(CPU, TuneCPU, FS)),
      TLInfo(TM, *this), FrameLowering(*this, getStackAlignment()) {
  // Determine the PICStyle based on the target selected.
  if (!isPositionIndependent())
    setPICStyle(PICStyles::Style::None);
  else if (is64Bit())
    setPICStyle(PICStyles::Style::RIPRel);
  else if (isTargetCOFF())
    setPICStyle(PICStyles::Style::None);
  else if (isTargetDarwin())
    setPICStyle(PICStyles::Style::StubPIC);
  else if (isTargetELF())
    setPICStyle(PICStyles::Style::GOT);

  CallLoweringInfo.reset(new X86CallLowering(*getTargetLowering()));
  Legalizer.reset(new X86LegalizerInfo(*this, TM));

  auto *RBI = new X86RegisterBankInfo(*getRegisterInfo());
  RegBankInfo.reset(RBI);
  InstSelector.reset(createX86InstructionSelector(TM, *this, *RBI));
}

const CallLowering *X86Subtarget::getCallLowering() const {
  return CallLoweringInfo.get();
}

InstructionSelector *X86Subtarget::getInstructionSelector() const {
  return InstSelector.get();
}

const LegalizerInfo *X86Subtarget::getLegalizerInfo() const {
  return Legalizer.get();
}

const RegisterBankInfo *X86Subtarget::getRegBankInfo() const {
  return RegBankInfo.get();
}

bool X86Subtarget::enableEarlyIfConversion() const {
  return hasCMov() && X86EarlyIfConv;
}

void X86Subtarget::getPostRAMutations(
    std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(createX86MacroFusionDAGMutation());
}

bool X86Subtarget::isPositionIndependent() const {
  return TM.isPositionIndependent();
}
