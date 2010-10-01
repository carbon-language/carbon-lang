//===-- X86InstComments.cpp - Generate verbose-asm comments for instrs ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines functionality used to emit comments about X86 instructions to
// an output stream for -fverbose-asm.
//
//===----------------------------------------------------------------------===//

#include "X86InstComments.h"
#include "X86GenInstrNames.inc"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"
#include "../X86ShuffleDecode.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Top Level Entrypoint
//===----------------------------------------------------------------------===//

/// EmitAnyX86InstComments - This function decodes x86 instructions and prints
/// newline terminated strings to the specified string if desired.  This
/// information is shown in disassembly dumps when verbose assembly is enabled.
void llvm::EmitAnyX86InstComments(const MCInst *MI, raw_ostream &OS,
                                  const char *(*getRegName)(unsigned)) {
  // If this is a shuffle operation, the switch should fill in this state.
  SmallVector<unsigned, 8> ShuffleMask;
  const char *DestName = 0, *Src1Name = 0, *Src2Name = 0;

  switch (MI->getOpcode()) {
  case X86::INSERTPSrr:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    Src2Name = getRegName(MI->getOperand(2).getReg());
    DecodeINSERTPSMask(MI->getOperand(3).getImm(), ShuffleMask);
    break;

  case X86::MOVLHPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodeMOVLHPSMask(2, ShuffleMask);
    break;

  case X86::MOVHLPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodeMOVHLPSMask(2, ShuffleMask);
    break;

  case X86::PSHUFDri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSHUFDmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodePSHUFMask(4, MI->getOperand(MI->getNumOperands()-1).getImm(),
                    ShuffleMask);
    break;

  case X86::PSHUFHWri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSHUFHWmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodePSHUFHWMask(MI->getOperand(MI->getNumOperands()-1).getImm(),
                      ShuffleMask);
    break;
  case X86::PSHUFLWri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSHUFLWmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodePSHUFLWMask(MI->getOperand(MI->getNumOperands()-1).getImm(),
                      ShuffleMask);
    break;

  case X86::PUNPCKHBWrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKHBWrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKHMask(16, ShuffleMask);
    break;
  case X86::PUNPCKHWDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKHWDrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKHMask(8, ShuffleMask);
    break;
  case X86::PUNPCKHDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKHDQrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKHMask(4, ShuffleMask);
    break;
  case X86::PUNPCKHQDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKHQDQrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKHMask(2, ShuffleMask);
    break;

  case X86::PUNPCKLBWrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLBWrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKLMask(16, ShuffleMask);
    break;
  case X86::PUNPCKLWDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLWDrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKLMask(8, ShuffleMask);
    break;
  case X86::PUNPCKLDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLDQrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKLMask(4, ShuffleMask);
    break;
  case X86::PUNPCKLQDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLQDQrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKLMask(2, ShuffleMask);
    break;

  case X86::SHUFPDrri:
    DecodeSHUFPSMask(2, MI->getOperand(3).getImm(), ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    Src2Name = getRegName(MI->getOperand(2).getReg());
    break;

  case X86::SHUFPSrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::SHUFPSrmi:
    DecodeSHUFPSMask(4, MI->getOperand(3).getImm(), ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::UNPCKLPDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKLPDrm:
    DecodeUNPCKLPMask(2, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::UNPCKLPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKLPSrm:
    DecodeUNPCKLPMask(4, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::UNPCKHPDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKHPDrm:
    DecodeUNPCKHPMask(2, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::UNPCKHPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKHPSrm:
    DecodeUNPCKHPMask(4, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  }


  // If this was a shuffle operation, print the shuffle mask.
  if (!ShuffleMask.empty()) {
    if (DestName == 0) DestName = Src1Name;
    OS << (DestName ? DestName : "mem") << " = ";

    // If the two sources are the same, canonicalize the input elements to be
    // from the first src so that we get larger element spans.
    if (Src1Name == Src2Name) {
      for (unsigned i = 0, e = ShuffleMask.size(); i != e; ++i) {
        if ((int)ShuffleMask[i] >= 0 && // Not sentinel.
            ShuffleMask[i] >= e)        // From second mask.
          ShuffleMask[i] -= e;
      }
    }

    // The shuffle mask specifies which elements of the src1/src2 fill in the
    // destination, with a few sentinel values.  Loop through and print them
    // out.
    for (unsigned i = 0, e = ShuffleMask.size(); i != e; ++i) {
      if (i != 0)
        OS << ',';
      if (ShuffleMask[i] == SM_SentinelZero) {
        OS << "zero";
        continue;
      }

      // Otherwise, it must come from src1 or src2.  Print the span of elements
      // that comes from this src.
      bool isSrc1 = ShuffleMask[i] < ShuffleMask.size();
      const char *SrcName = isSrc1 ? Src1Name : Src2Name;
      OS << (SrcName ? SrcName : "mem") << '[';
      bool IsFirst = true;
      while (i != e &&
             (int)ShuffleMask[i] >= 0 &&
             (ShuffleMask[i] < ShuffleMask.size()) == isSrc1) {
        if (!IsFirst)
          OS << ',';
        else
          IsFirst = false;
        OS << ShuffleMask[i] % ShuffleMask.size();
        ++i;
      }
      OS << ']';
      --i;  // For loop increments element #.
    }
    //MI->print(OS, 0);
    OS << "\n";
  }

}
