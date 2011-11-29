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
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "Utils/X86ShuffleDecode.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"
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
    Src1Name = getRegName(MI->getOperand(0).getReg());
    Src2Name = getRegName(MI->getOperand(2).getReg());
    DecodeINSERTPSMask(MI->getOperand(3).getImm(), ShuffleMask);
    break;
  case X86::VINSERTPSrr:
    DestName = getRegName(MI->getOperand(0).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    Src2Name = getRegName(MI->getOperand(2).getReg());
    DecodeINSERTPSMask(MI->getOperand(3).getImm(), ShuffleMask);
    break;

  case X86::MOVLHPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodeMOVLHPSMask(2, ShuffleMask);
    break;
  case X86::VMOVLHPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVLHPSMask(2, ShuffleMask);
    break;

  case X86::MOVHLPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodeMOVHLPSMask(2, ShuffleMask);
    break;
  case X86::VMOVHLPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodeMOVHLPSMask(2, ShuffleMask);
    break;

  case X86::PSHUFDri:
  case X86::VPSHUFDri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSHUFDmi:
  case X86::VPSHUFDmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodePSHUFMask(4, MI->getOperand(MI->getNumOperands()-1).getImm(),
                    ShuffleMask);
    break;

  case X86::PSHUFHWri:
  case X86::VPSHUFHWri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSHUFHWmi:
  case X86::VPSHUFHWmi:
    DestName = getRegName(MI->getOperand(0).getReg());
    DecodePSHUFHWMask(MI->getOperand(MI->getNumOperands()-1).getImm(),
                      ShuffleMask);
    break;
  case X86::PSHUFLWri:
  case X86::VPSHUFLWri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::PSHUFLWmi:
  case X86::VPSHUFLWmi:
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
    DecodePUNPCKLBWMask(16, ShuffleMask);
    break;
  case X86::PUNPCKLWDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLWDrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKLWDMask(8, ShuffleMask);
    break;
  case X86::PUNPCKLDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLDQrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKLDQMask(4, ShuffleMask);
    break;
  case X86::PUNPCKLQDQrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::PUNPCKLQDQrm:
    Src1Name = getRegName(MI->getOperand(0).getReg());
    DecodePUNPCKLQDQMask(2, ShuffleMask);
    break;

  case X86::SHUFPDrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::SHUFPDrmi:
    DecodeSHUFPMask(MVT::v2f64, MI->getOperand(MI->getNumOperands()-1).getImm(),
                    ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VSHUFPDrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VSHUFPDrmi:
    DecodeSHUFPMask(MVT::v2f64, MI->getOperand(MI->getNumOperands()-1).getImm(),
                    ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VSHUFPDYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VSHUFPDYrmi:
    DecodeSHUFPMask(MVT::v4f64, MI->getOperand(MI->getNumOperands()-1).getImm(),
                    ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::SHUFPSrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::SHUFPSrmi:
    DecodeSHUFPMask(MVT::v4f32, MI->getOperand(MI->getNumOperands()-1).getImm(),
                    ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VSHUFPSrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VSHUFPSrmi:
    DecodeSHUFPMask(MVT::v4f32, MI->getOperand(MI->getNumOperands()-1).getImm(),
                    ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VSHUFPSYrri:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VSHUFPSYrmi:
    DecodeSHUFPMask(MVT::v8f32, MI->getOperand(MI->getNumOperands()-1).getImm(),
                    ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;

  case X86::UNPCKLPDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKLPDrm:
    DecodeUNPCKLPMask(MVT::v2f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKLPDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKLPDrm:
    DecodeUNPCKLPMask(MVT::v2f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKLPDYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKLPDYrm:
    DecodeUNPCKLPMask(MVT::v4f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::UNPCKLPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKLPSrm:
    DecodeUNPCKLPMask(MVT::v4f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKLPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKLPSrm:
    DecodeUNPCKLPMask(MVT::v4f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKLPSYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKLPSYrm:
    DecodeUNPCKLPMask(MVT::v8f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::UNPCKHPDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKHPDrm:
    DecodeUNPCKHPMask(MVT::v2f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKHPDrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKHPDrm:
    DecodeUNPCKHPMask(MVT::v2f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKHPDYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKHPDYrm:
    DecodeUNPCKLPMask(MVT::v4f64, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::UNPCKHPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::UNPCKHPSrm:
    DecodeUNPCKHPMask(MVT::v4f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKHPSrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKHPSrm:
    DecodeUNPCKHPMask(MVT::v4f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VUNPCKHPSYrr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VUNPCKHPSYrm:
    DecodeUNPCKHPMask(MVT::v8f32, ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERMILPSri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPERMILPSmi:
    DecodeVPERMILPSMask(4, MI->getOperand(MI->getNumOperands()-1).getImm(),
                        ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERMILPSYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPERMILPSYmi:
    DecodeVPERMILPSMask(8, MI->getOperand(MI->getNumOperands()-1).getImm(),
                        ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERMILPDri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPERMILPDmi:
    DecodeVPERMILPDMask(2, MI->getOperand(MI->getNumOperands()-1).getImm(),
                        ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERMILPDYri:
    Src1Name = getRegName(MI->getOperand(1).getReg());
    // FALL THROUGH.
  case X86::VPERMILPDYmi:
    DecodeVPERMILPDMask(4, MI->getOperand(MI->getNumOperands()-1).getImm(),
                        ShuffleMask);
    DestName = getRegName(MI->getOperand(0).getReg());
    break;
  case X86::VPERM2F128rr:
  case X86::VPERM2I128rr:
    Src2Name = getRegName(MI->getOperand(2).getReg());
    // FALL THROUGH.
  case X86::VPERM2F128rm:
  case X86::VPERM2I128rm:
    DecodeVPERM2F128Mask(MI->getOperand(MI->getNumOperands()-1).getImm(),
                         ShuffleMask);
    Src1Name = getRegName(MI->getOperand(1).getReg());
    DestName = getRegName(MI->getOperand(0).getReg());
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
