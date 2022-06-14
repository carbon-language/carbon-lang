//===--- X86InstPrinterCommon.cpp - X86 assembly instruction printing -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file includes common code for rendering MCInst instances as Intel-style
// and Intel-style assembly.
//
//===----------------------------------------------------------------------===//

#include "X86InstPrinterCommon.h"
#include "X86BaseInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

void X86InstPrinterCommon::printCondCode(const MCInst *MI, unsigned Op,
                                         raw_ostream &O) {
  int64_t Imm = MI->getOperand(Op).getImm();
  switch (Imm) {
  default: llvm_unreachable("Invalid condcode argument!");
  case    0: O << "o";  break;
  case    1: O << "no"; break;
  case    2: O << "b";  break;
  case    3: O << "ae"; break;
  case    4: O << "e";  break;
  case    5: O << "ne"; break;
  case    6: O << "be"; break;
  case    7: O << "a";  break;
  case    8: O << "s";  break;
  case    9: O << "ns"; break;
  case  0xa: O << "p";  break;
  case  0xb: O << "np"; break;
  case  0xc: O << "l";  break;
  case  0xd: O << "ge"; break;
  case  0xe: O << "le"; break;
  case  0xf: O << "g";  break;
  }
}

void X86InstPrinterCommon::printSSEAVXCC(const MCInst *MI, unsigned Op,
                                         raw_ostream &O) {
  int64_t Imm = MI->getOperand(Op).getImm();
  switch (Imm) {
  default: llvm_unreachable("Invalid ssecc/avxcc argument!");
  case    0: O << "eq"; break;
  case    1: O << "lt"; break;
  case    2: O << "le"; break;
  case    3: O << "unord"; break;
  case    4: O << "neq"; break;
  case    5: O << "nlt"; break;
  case    6: O << "nle"; break;
  case    7: O << "ord"; break;
  case    8: O << "eq_uq"; break;
  case    9: O << "nge"; break;
  case  0xa: O << "ngt"; break;
  case  0xb: O << "false"; break;
  case  0xc: O << "neq_oq"; break;
  case  0xd: O << "ge"; break;
  case  0xe: O << "gt"; break;
  case  0xf: O << "true"; break;
  case 0x10: O << "eq_os"; break;
  case 0x11: O << "lt_oq"; break;
  case 0x12: O << "le_oq"; break;
  case 0x13: O << "unord_s"; break;
  case 0x14: O << "neq_us"; break;
  case 0x15: O << "nlt_uq"; break;
  case 0x16: O << "nle_uq"; break;
  case 0x17: O << "ord_s"; break;
  case 0x18: O << "eq_us"; break;
  case 0x19: O << "nge_uq"; break;
  case 0x1a: O << "ngt_uq"; break;
  case 0x1b: O << "false_os"; break;
  case 0x1c: O << "neq_os"; break;
  case 0x1d: O << "ge_oq"; break;
  case 0x1e: O << "gt_oq"; break;
  case 0x1f: O << "true_us"; break;
  }
}

void X86InstPrinterCommon::printVPCOMMnemonic(const MCInst *MI,
                                              raw_ostream &OS) {
  OS << "vpcom";

  int64_t Imm = MI->getOperand(MI->getNumOperands() - 1).getImm();
  switch (Imm) {
  default: llvm_unreachable("Invalid vpcom argument!");
  case 0: OS << "lt"; break;
  case 1: OS << "le"; break;
  case 2: OS << "gt"; break;
  case 3: OS << "ge"; break;
  case 4: OS << "eq"; break;
  case 5: OS << "neq"; break;
  case 6: OS << "false"; break;
  case 7: OS << "true"; break;
  }

  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unexpected opcode!");
  case X86::VPCOMBmi:  case X86::VPCOMBri:  OS << "b\t";  break;
  case X86::VPCOMDmi:  case X86::VPCOMDri:  OS << "d\t";  break;
  case X86::VPCOMQmi:  case X86::VPCOMQri:  OS << "q\t";  break;
  case X86::VPCOMUBmi: case X86::VPCOMUBri: OS << "ub\t"; break;
  case X86::VPCOMUDmi: case X86::VPCOMUDri: OS << "ud\t"; break;
  case X86::VPCOMUQmi: case X86::VPCOMUQri: OS << "uq\t"; break;
  case X86::VPCOMUWmi: case X86::VPCOMUWri: OS << "uw\t"; break;
  case X86::VPCOMWmi:  case X86::VPCOMWri:  OS << "w\t";  break;
  }
}

void X86InstPrinterCommon::printVPCMPMnemonic(const MCInst *MI,
                                              raw_ostream &OS) {
  OS << "vpcmp";

  printSSEAVXCC(MI, MI->getNumOperands() - 1, OS);

  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unexpected opcode!");
  case X86::VPCMPBZ128rmi:  case X86::VPCMPBZ128rri:
  case X86::VPCMPBZ256rmi:  case X86::VPCMPBZ256rri:
  case X86::VPCMPBZrmi:     case X86::VPCMPBZrri:
  case X86::VPCMPBZ128rmik: case X86::VPCMPBZ128rrik:
  case X86::VPCMPBZ256rmik: case X86::VPCMPBZ256rrik:
  case X86::VPCMPBZrmik:    case X86::VPCMPBZrrik:
    OS << "b\t";
    break;
  case X86::VPCMPDZ128rmi:  case X86::VPCMPDZ128rri:
  case X86::VPCMPDZ256rmi:  case X86::VPCMPDZ256rri:
  case X86::VPCMPDZrmi:     case X86::VPCMPDZrri:
  case X86::VPCMPDZ128rmik: case X86::VPCMPDZ128rrik:
  case X86::VPCMPDZ256rmik: case X86::VPCMPDZ256rrik:
  case X86::VPCMPDZrmik:    case X86::VPCMPDZrrik:
  case X86::VPCMPDZ128rmib: case X86::VPCMPDZ128rmibk:
  case X86::VPCMPDZ256rmib: case X86::VPCMPDZ256rmibk:
  case X86::VPCMPDZrmib:    case X86::VPCMPDZrmibk:
    OS << "d\t";
    break;
  case X86::VPCMPQZ128rmi:  case X86::VPCMPQZ128rri:
  case X86::VPCMPQZ256rmi:  case X86::VPCMPQZ256rri:
  case X86::VPCMPQZrmi:     case X86::VPCMPQZrri:
  case X86::VPCMPQZ128rmik: case X86::VPCMPQZ128rrik:
  case X86::VPCMPQZ256rmik: case X86::VPCMPQZ256rrik:
  case X86::VPCMPQZrmik:    case X86::VPCMPQZrrik:
  case X86::VPCMPQZ128rmib: case X86::VPCMPQZ128rmibk:
  case X86::VPCMPQZ256rmib: case X86::VPCMPQZ256rmibk:
  case X86::VPCMPQZrmib:    case X86::VPCMPQZrmibk:
    OS << "q\t";
    break;
  case X86::VPCMPUBZ128rmi:  case X86::VPCMPUBZ128rri:
  case X86::VPCMPUBZ256rmi:  case X86::VPCMPUBZ256rri:
  case X86::VPCMPUBZrmi:     case X86::VPCMPUBZrri:
  case X86::VPCMPUBZ128rmik: case X86::VPCMPUBZ128rrik:
  case X86::VPCMPUBZ256rmik: case X86::VPCMPUBZ256rrik:
  case X86::VPCMPUBZrmik:    case X86::VPCMPUBZrrik:
    OS << "ub\t";
    break;
  case X86::VPCMPUDZ128rmi:  case X86::VPCMPUDZ128rri:
  case X86::VPCMPUDZ256rmi:  case X86::VPCMPUDZ256rri:
  case X86::VPCMPUDZrmi:     case X86::VPCMPUDZrri:
  case X86::VPCMPUDZ128rmik: case X86::VPCMPUDZ128rrik:
  case X86::VPCMPUDZ256rmik: case X86::VPCMPUDZ256rrik:
  case X86::VPCMPUDZrmik:    case X86::VPCMPUDZrrik:
  case X86::VPCMPUDZ128rmib: case X86::VPCMPUDZ128rmibk:
  case X86::VPCMPUDZ256rmib: case X86::VPCMPUDZ256rmibk:
  case X86::VPCMPUDZrmib:    case X86::VPCMPUDZrmibk:
    OS << "ud\t";
    break;
  case X86::VPCMPUQZ128rmi:  case X86::VPCMPUQZ128rri:
  case X86::VPCMPUQZ256rmi:  case X86::VPCMPUQZ256rri:
  case X86::VPCMPUQZrmi:     case X86::VPCMPUQZrri:
  case X86::VPCMPUQZ128rmik: case X86::VPCMPUQZ128rrik:
  case X86::VPCMPUQZ256rmik: case X86::VPCMPUQZ256rrik:
  case X86::VPCMPUQZrmik:    case X86::VPCMPUQZrrik:
  case X86::VPCMPUQZ128rmib: case X86::VPCMPUQZ128rmibk:
  case X86::VPCMPUQZ256rmib: case X86::VPCMPUQZ256rmibk:
  case X86::VPCMPUQZrmib:    case X86::VPCMPUQZrmibk:
    OS << "uq\t";
    break;
  case X86::VPCMPUWZ128rmi:  case X86::VPCMPUWZ128rri:
  case X86::VPCMPUWZ256rri:  case X86::VPCMPUWZ256rmi:
  case X86::VPCMPUWZrmi:     case X86::VPCMPUWZrri:
  case X86::VPCMPUWZ128rmik: case X86::VPCMPUWZ128rrik:
  case X86::VPCMPUWZ256rrik: case X86::VPCMPUWZ256rmik:
  case X86::VPCMPUWZrmik:    case X86::VPCMPUWZrrik:
    OS << "uw\t";
    break;
  case X86::VPCMPWZ128rmi:  case X86::VPCMPWZ128rri:
  case X86::VPCMPWZ256rmi:  case X86::VPCMPWZ256rri:
  case X86::VPCMPWZrmi:     case X86::VPCMPWZrri:
  case X86::VPCMPWZ128rmik: case X86::VPCMPWZ128rrik:
  case X86::VPCMPWZ256rmik: case X86::VPCMPWZ256rrik:
  case X86::VPCMPWZrmik:    case X86::VPCMPWZrrik:
    OS << "w\t";
    break;
  }
}

void X86InstPrinterCommon::printCMPMnemonic(const MCInst *MI, bool IsVCmp,
                                            raw_ostream &OS) {
  OS << (IsVCmp ? "vcmp" : "cmp");

  printSSEAVXCC(MI, MI->getNumOperands() - 1, OS);

  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unexpected opcode!");
  case X86::CMPPDrmi:       case X86::CMPPDrri:
  case X86::VCMPPDrmi:      case X86::VCMPPDrri:
  case X86::VCMPPDYrmi:     case X86::VCMPPDYrri:
  case X86::VCMPPDZ128rmi:  case X86::VCMPPDZ128rri:
  case X86::VCMPPDZ256rmi:  case X86::VCMPPDZ256rri:
  case X86::VCMPPDZrmi:     case X86::VCMPPDZrri:
  case X86::VCMPPDZ128rmik: case X86::VCMPPDZ128rrik:
  case X86::VCMPPDZ256rmik: case X86::VCMPPDZ256rrik:
  case X86::VCMPPDZrmik:    case X86::VCMPPDZrrik:
  case X86::VCMPPDZ128rmbi: case X86::VCMPPDZ128rmbik:
  case X86::VCMPPDZ256rmbi: case X86::VCMPPDZ256rmbik:
  case X86::VCMPPDZrmbi:    case X86::VCMPPDZrmbik:
  case X86::VCMPPDZrrib:    case X86::VCMPPDZrribk:
    OS << "pd\t";
    break;
  case X86::CMPPSrmi:       case X86::CMPPSrri:
  case X86::VCMPPSrmi:      case X86::VCMPPSrri:
  case X86::VCMPPSYrmi:     case X86::VCMPPSYrri:
  case X86::VCMPPSZ128rmi:  case X86::VCMPPSZ128rri:
  case X86::VCMPPSZ256rmi:  case X86::VCMPPSZ256rri:
  case X86::VCMPPSZrmi:     case X86::VCMPPSZrri:
  case X86::VCMPPSZ128rmik: case X86::VCMPPSZ128rrik:
  case X86::VCMPPSZ256rmik: case X86::VCMPPSZ256rrik:
  case X86::VCMPPSZrmik:    case X86::VCMPPSZrrik:
  case X86::VCMPPSZ128rmbi: case X86::VCMPPSZ128rmbik:
  case X86::VCMPPSZ256rmbi: case X86::VCMPPSZ256rmbik:
  case X86::VCMPPSZrmbi:    case X86::VCMPPSZrmbik:
  case X86::VCMPPSZrrib:    case X86::VCMPPSZrribk:
    OS << "ps\t";
    break;
  case X86::CMPSDrm:        case X86::CMPSDrr:
  case X86::CMPSDrm_Int:    case X86::CMPSDrr_Int:
  case X86::VCMPSDrm:       case X86::VCMPSDrr:
  case X86::VCMPSDrm_Int:   case X86::VCMPSDrr_Int:
  case X86::VCMPSDZrm:      case X86::VCMPSDZrr:
  case X86::VCMPSDZrm_Int:  case X86::VCMPSDZrr_Int:
  case X86::VCMPSDZrm_Intk: case X86::VCMPSDZrr_Intk:
  case X86::VCMPSDZrrb_Int: case X86::VCMPSDZrrb_Intk:
    OS << "sd\t";
    break;
  case X86::CMPSSrm:        case X86::CMPSSrr:
  case X86::CMPSSrm_Int:    case X86::CMPSSrr_Int:
  case X86::VCMPSSrm:       case X86::VCMPSSrr:
  case X86::VCMPSSrm_Int:   case X86::VCMPSSrr_Int:
  case X86::VCMPSSZrm:      case X86::VCMPSSZrr:
  case X86::VCMPSSZrm_Int:  case X86::VCMPSSZrr_Int:
  case X86::VCMPSSZrm_Intk: case X86::VCMPSSZrr_Intk:
  case X86::VCMPSSZrrb_Int: case X86::VCMPSSZrrb_Intk:
    OS << "ss\t";
    break;
  case X86::VCMPPHZ128rmi:  case X86::VCMPPHZ128rri:
  case X86::VCMPPHZ256rmi:  case X86::VCMPPHZ256rri:
  case X86::VCMPPHZrmi:     case X86::VCMPPHZrri:
  case X86::VCMPPHZ128rmik: case X86::VCMPPHZ128rrik:
  case X86::VCMPPHZ256rmik: case X86::VCMPPHZ256rrik:
  case X86::VCMPPHZrmik:    case X86::VCMPPHZrrik:
  case X86::VCMPPHZ128rmbi: case X86::VCMPPHZ128rmbik:
  case X86::VCMPPHZ256rmbi: case X86::VCMPPHZ256rmbik:
  case X86::VCMPPHZrmbi:    case X86::VCMPPHZrmbik:
  case X86::VCMPPHZrrib:    case X86::VCMPPHZrribk:
    OS << "ph\t";
    break;
  case X86::VCMPSHZrm:      case X86::VCMPSHZrr:
  case X86::VCMPSHZrm_Int:  case X86::VCMPSHZrr_Int:
  case X86::VCMPSHZrrb_Int: case X86::VCMPSHZrrb_Intk:
  case X86::VCMPSHZrm_Intk: case X86::VCMPSHZrr_Intk:
    OS << "sh\t";
    break;
  }
}

void X86InstPrinterCommon::printRoundingControl(const MCInst *MI, unsigned Op,
                                                raw_ostream &O) {
  int64_t Imm = MI->getOperand(Op).getImm();
  switch (Imm) {
  default:
    llvm_unreachable("Invalid rounding control!");
  case X86::TO_NEAREST_INT:
    O << "{rn-sae}";
    break;
  case X86::TO_NEG_INF:
    O << "{rd-sae}";
    break;
  case X86::TO_POS_INF:
    O << "{ru-sae}";
    break;
  case X86::TO_ZERO:
    O << "{rz-sae}";
    break;
  }
}

/// value (e.g. for jumps and calls). In Intel-style these print slightly
/// differently than normal immediates. For example, a $ is not emitted.
///
/// \p Address The address of the next instruction.
/// \see MCInstPrinter::printInst
void X86InstPrinterCommon::printPCRelImm(const MCInst *MI, uint64_t Address,
                                         unsigned OpNo, raw_ostream &O) {
  // Do not print the numberic target address when symbolizing.
  if (SymbolizeOperands)
    return;

  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    if (PrintBranchImmAsAddress) {
      uint64_t Target = Address + Op.getImm();
      if (MAI.getCodePointerSize() == 4)
        Target &= 0xffffffff;
      O << formatHex(Target);
    } else
      O << formatImm(Op.getImm());
  } else {
    assert(Op.isExpr() && "unknown pcrel immediate operand");
    // If a symbolic branch target was added as a constant expression then print
    // that address in hex.
    const MCConstantExpr *BranchTarget = dyn_cast<MCConstantExpr>(Op.getExpr());
    int64_t Address;
    if (BranchTarget && BranchTarget->evaluateAsAbsolute(Address)) {
      O << formatHex((uint64_t)Address);
    } else {
      // Otherwise, just print the expression.
      Op.getExpr()->print(O, &MAI);
    }
  }
}

void X86InstPrinterCommon::printOptionalSegReg(const MCInst *MI, unsigned OpNo,
                                               raw_ostream &O) {
  if (MI->getOperand(OpNo).getReg()) {
    printOperand(MI, OpNo, O);
    O << ':';
  }
}

void X86InstPrinterCommon::printInstFlags(const MCInst *MI, raw_ostream &O,
                                          const MCSubtargetInfo &STI) {
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  uint64_t TSFlags = Desc.TSFlags;
  unsigned Flags = MI->getFlags();

  if ((TSFlags & X86II::LOCK) || (Flags & X86::IP_HAS_LOCK))
    O << "\tlock\t";

  if ((TSFlags & X86II::NOTRACK) || (Flags & X86::IP_HAS_NOTRACK))
    O << "\tnotrack\t";

  if (Flags & X86::IP_HAS_REPEAT_NE)
    O << "\trepne\t";
  else if (Flags & X86::IP_HAS_REPEAT)
    O << "\trep\t";

  // These all require a pseudo prefix
  if ((Flags & X86::IP_USE_VEX) || (TSFlags & X86II::ExplicitVEXPrefix))
    O << "\t{vex}";
  else if (Flags & X86::IP_USE_VEX2)
    O << "\t{vex2}";
  else if (Flags & X86::IP_USE_VEX3)
    O << "\t{vex3}";
  else if (Flags & X86::IP_USE_EVEX)
    O << "\t{evex}";

  if (Flags & X86::IP_USE_DISP8)
    O << "\t{disp8}";
  else if (Flags & X86::IP_USE_DISP32)
    O << "\t{disp32}";

  // Determine where the memory operand starts, if present
  int MemoryOperand = X86II::getMemoryOperandNo(TSFlags);
  if (MemoryOperand != -1)
    MemoryOperand += X86II::getOperandBias(Desc);

  // Address-Size override prefix
  if (Flags & X86::IP_HAS_AD_SIZE &&
      !X86_MC::needsAddressSizeOverride(*MI, STI, MemoryOperand, TSFlags)) {
    if (STI.hasFeature(X86::Is16Bit) || STI.hasFeature(X86::Is64Bit))
      O << "\taddr32\t";
    else if (STI.hasFeature(X86::Is32Bit))
      O << "\taddr16\t";
  }
}

void X86InstPrinterCommon::printVKPair(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &OS) {
  // In assembly listings, a pair is represented by one of its members, any
  // of the two.  Here, we pick k0, k2, k4, k6, but we could as well
  // print K2_K3 as "k3".  It would probably make a lot more sense, if
  // the assembly would look something like:
  // "vp2intersect %zmm5, %zmm7, {%k2, %k3}"
  // but this can work too.
  switch (MI->getOperand(OpNo).getReg()) {
  case X86::K0_K1:
    printRegName(OS, X86::K0);
    return;
  case X86::K2_K3:
    printRegName(OS, X86::K2);
    return;
  case X86::K4_K5:
    printRegName(OS, X86::K4);
    return;
  case X86::K6_K7:
    printRegName(OS, X86::K6);
    return;
  }
  llvm_unreachable("Unknown mask pair register name");
}
