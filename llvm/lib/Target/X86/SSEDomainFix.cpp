//===- SSEDomainFix.cpp - Use proper int/float domain for SSE ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SSEDomainFix pass.
//
// Some SSE instructions like mov, and, or, xor are available in different
// variants for different operand types. These variant instructions are
// equivalent, but on Nehalem and newer cpus there is extra latency
// transferring data between integer and floating point domains.
//
// This pass changes the variant instructions to minimize domain crossings.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sse-domain-fix"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class SSEDomainFixPass : public MachineFunctionPass {
  static char ID;
  const X86InstrInfo *TII;

  MachineFunction *MF;
  MachineBasicBlock *MBB;
public:
  SSEDomainFixPass() : MachineFunctionPass(&ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual const char *getPassName() const {
    return "SSE execution domain fixup";
  }

private:
  void enterBasicBlock(MachineBasicBlock *MBB);
};
}

void SSEDomainFixPass::enterBasicBlock(MachineBasicBlock *mbb) {
  MBB = mbb;
  DEBUG(dbgs() << "Entering MBB " << MBB->getName() << "\n");
}

bool SSEDomainFixPass::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  TII = static_cast<const X86InstrInfo*>(MF->getTarget().getInstrInfo());

  MachineBasicBlock *Entry = MF->begin();
  SmallPtrSet<MachineBasicBlock*, 16> Visited;
  for (df_ext_iterator<MachineBasicBlock*,
         SmallPtrSet<MachineBasicBlock*, 16> >
         DFI = df_ext_begin(Entry, Visited), DFE = df_ext_end(Entry, Visited);
       DFI != DFE; ++DFI) {
    enterBasicBlock(*DFI);
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
        ++I) {
      MachineInstr *MI = I;
      const unsigned *equiv = 0;
      X86InstrInfo::SSEDomain domain = TII->GetSSEDomain(MI, equiv);
      DEBUG(dbgs() << "isd-"[domain] << (equiv ? "* " : "  ") << *MI);
    }
  }
  return false;
}

FunctionPass *llvm::createSSEDomainFixPass() {
  return new SSEDomainFixPass();
}

// These are the replaceable instructions. Some of these have _Int variants
// that we don't include here. We don't want to replace instructions selected
// by intrinsics.
static const unsigned ReplaceableInstrs[][3] = {
  //PackedInt          PackedSingle     PackedDouble
  { X86::MOVDQAmr,     X86::MOVAPSmr,   X86::MOVAPDmr   },
  { X86::MOVDQArm,     X86::MOVAPSrm,   X86::MOVAPDrm   },
  { X86::MOVDQArr,     X86::MOVAPSrr,   X86::MOVAPDrr   },
  { X86::MOVDQUmr,     X86::MOVUPSmr,   X86::MOVUPDmr   },
  { X86::MOVDQUrm,     X86::MOVUPSrm,   X86::MOVUPDrm   },
  { X86::MOVNTDQmr,    X86::MOVNTPSmr,  X86::MOVNTPDmr  },
  { X86::PANDNrm,      X86::ANDNPSrm,   X86::ANDNPDrm   },
  { X86::PANDNrr,      X86::ANDNPSrr,   X86::ANDNPDrr   },
  { X86::PANDrm,       X86::ANDPSrm,    X86::ANDPDrm    },
  { X86::PANDrr,       X86::ANDPSrr,    X86::ANDPDrr    },
  { X86::PORrm,        X86::ORPSrm,     X86::ORPDrm     },
  { X86::PORrr,        X86::ORPSrr,     X86::ORPDrr     },
  { X86::PUNPCKHQDQrm, X86::UNPCKHPSrm, X86::UNPCKHPDrm },
  { X86::PUNPCKHQDQrr, X86::UNPCKHPSrr, X86::UNPCKHPDrr },
  { X86::PUNPCKLQDQrm, X86::UNPCKLPSrm, X86::UNPCKLPDrm },
  { X86::PUNPCKLQDQrr, X86::UNPCKLPSrr, X86::UNPCKLPDrr },
  { X86::PXORrm,       X86::XORPSrm,    X86::XORPDrm    },
  { X86::PXORrr,       X86::XORPSrr,    X86::XORPDrr    },
};

void X86InstrInfo::populateSSEInstrDomainTable() {
  // Instructions that execute in the packed integer domain.
  static const unsigned PackedIntInstrs[] = {
    X86::LDDQUrm,
    X86::MASKMOVDQU,
    X86::MASKMOVDQU64,
    X86::MOVDI2PDIrm,
    X86::MOVDI2PDIrr,
    X86::MOVDQUmr_Int,
    X86::MOVDQUrm_Int,
    X86::MOVLQ128mr,
    X86::MOVNTDQArm,
    X86::MOVNTDQmr_Int,
    X86::MOVNTDQ_64mr,
    X86::MOVPDI2DImr,
    X86::MOVPDI2DIrr,
    X86::MOVPQI2QImr,
    X86::MOVPQIto64rr,
    X86::MOVQI2PQIrm,
    X86::MOVQxrxr,
    X86::MOVZDI2PDIrm,
    X86::MOVZDI2PDIrr,
    X86::MOVZPQILo2PQIrm,
    X86::MOVZPQILo2PQIrr,
    X86::MOVZQI2PQIrm,
    X86::MOVZQI2PQIrr,
    X86::MPSADBWrmi,
    X86::MPSADBWrri,
    X86::PABSBrm128,
    X86::PABSBrr128,
    X86::PABSDrm128,
    X86::PABSDrr128,
    X86::PABSWrm128,
    X86::PABSWrr128,
    X86::PACKSSDWrm,
    X86::PACKSSDWrr,
    X86::PACKSSWBrm,
    X86::PACKSSWBrr,
    X86::PACKUSDWrm,
    X86::PACKUSDWrr,
    X86::PACKUSWBrm,
    X86::PACKUSWBrr,
    X86::PADDBrm,
    X86::PADDBrr,
    X86::PADDDrm,
    X86::PADDDrr,
    X86::PADDQrm,
    X86::PADDQrr,
    X86::PADDSBrm,
    X86::PADDSBrr,
    X86::PADDSWrm,
    X86::PADDSWrr,
    X86::PADDUSBrm,
    X86::PADDUSBrr,
    X86::PADDUSWrm,
    X86::PADDUSWrr,
    X86::PADDWrm,
    X86::PADDWrr,
    X86::PALIGNR128rm,
    X86::PALIGNR128rr,
    X86::PAVGBrm,
    X86::PAVGBrr,
    X86::PAVGWrm,
    X86::PAVGWrr,
    X86::PBLENDVBrm0,
    X86::PBLENDVBrr0,
    X86::PBLENDWrmi,
    X86::PBLENDWrri,
    X86::PCMPEQBrm,
    X86::PCMPEQBrr,
    X86::PCMPEQDrm,
    X86::PCMPEQDrr,
    X86::PCMPEQQrm,
    X86::PCMPEQQrr,
    X86::PCMPEQWrm,
    X86::PCMPEQWrr,
    X86::PCMPESTRIArm,
    X86::PCMPESTRIArr,
    X86::PCMPESTRICrm,
    X86::PCMPESTRICrr,
    X86::PCMPESTRIOrm,
    X86::PCMPESTRIOrr,
    X86::PCMPESTRIrm,
    X86::PCMPESTRIrr,
    X86::PCMPESTRISrm,
    X86::PCMPESTRISrr,
    X86::PCMPESTRIZrm,
    X86::PCMPESTRIZrr,
    X86::PCMPESTRM128MEM,
    X86::PCMPESTRM128REG,
    X86::PCMPESTRM128rm,
    X86::PCMPESTRM128rr,
    X86::PCMPGTBrm,
    X86::PCMPGTBrr,
    X86::PCMPGTDrm,
    X86::PCMPGTDrr,
    X86::PCMPGTQrm,
    X86::PCMPGTQrr,
    X86::PCMPGTWrm,
    X86::PCMPGTWrr,
    X86::PCMPISTRIArm,
    X86::PCMPISTRIArr,
    X86::PCMPISTRICrm,
    X86::PCMPISTRICrr,
    X86::PCMPISTRIOrm,
    X86::PCMPISTRIOrr,
    X86::PCMPISTRIrm,
    X86::PCMPISTRIrr,
    X86::PCMPISTRISrm,
    X86::PCMPISTRISrr,
    X86::PCMPISTRIZrm,
    X86::PCMPISTRIZrr,
    X86::PCMPISTRM128MEM,
    X86::PCMPISTRM128REG,
    X86::PCMPISTRM128rm,
    X86::PCMPISTRM128rr,
    X86::PEXTRBmr,
    X86::PEXTRBrr,
    X86::PEXTRDmr,
    X86::PEXTRDrr,
    X86::PEXTRQmr,
    X86::PEXTRQrr,
    X86::PEXTRWmr,
    X86::PEXTRWri,
    X86::PHADDDrm128,
    X86::PHADDDrr128,
    X86::PHADDSWrm128,
    X86::PHADDSWrr128,
    X86::PHADDWrm128,
    X86::PHADDWrr128,
    X86::PHMINPOSUWrm128,
    X86::PHMINPOSUWrr128,
    X86::PHSUBDrm128,
    X86::PHSUBDrr128,
    X86::PHSUBSWrm128,
    X86::PHSUBSWrr128,
    X86::PHSUBWrm128,
    X86::PHSUBWrr128,
    X86::PINSRBrm,
    X86::PINSRBrr,
    X86::PINSRDrm,
    X86::PINSRDrr,
    X86::PINSRQrm,
    X86::PINSRQrr,
    X86::PINSRWrmi,
    X86::PINSRWrri,
    X86::PMADDUBSWrm128,
    X86::PMADDUBSWrr128,
    X86::PMADDWDrm,
    X86::PMADDWDrr,
    X86::PMAXSBrm,
    X86::PMAXSBrr,
    X86::PMAXSDrm,
    X86::PMAXSDrr,
    X86::PMAXSWrm,
    X86::PMAXSWrr,
    X86::PMAXUBrm,
    X86::PMAXUBrr,
    X86::PMAXUDrm,
    X86::PMAXUDrr,
    X86::PMAXUWrm,
    X86::PMAXUWrr,
    X86::PMINSBrm,
    X86::PMINSBrr,
    X86::PMINSDrm,
    X86::PMINSDrr,
    X86::PMINSWrm,
    X86::PMINSWrr,
    X86::PMINUBrm,
    X86::PMINUBrr,
    X86::PMINUDrm,
    X86::PMINUDrr,
    X86::PMINUWrm,
    X86::PMINUWrr,
    X86::PMOVSXBDrm,
    X86::PMOVSXBDrr,
    X86::PMOVSXBQrm,
    X86::PMOVSXBQrr,
    X86::PMOVSXBWrm,
    X86::PMOVSXBWrr,
    X86::PMOVSXDQrm,
    X86::PMOVSXDQrr,
    X86::PMOVSXWDrm,
    X86::PMOVSXWDrr,
    X86::PMOVSXWQrm,
    X86::PMOVSXWQrr,
    X86::PMOVZXBDrm,
    X86::PMOVZXBDrr,
    X86::PMOVZXBQrm,
    X86::PMOVZXBQrr,
    X86::PMOVZXBWrm,
    X86::PMOVZXBWrr,
    X86::PMOVZXDQrm,
    X86::PMOVZXDQrr,
    X86::PMOVZXWDrm,
    X86::PMOVZXWDrr,
    X86::PMOVZXWQrm,
    X86::PMOVZXWQrr,
    X86::PMULDQrm,
    X86::PMULDQrr,
    X86::PMULHRSWrm128,
    X86::PMULHRSWrr128,
    X86::PMULHUWrm,
    X86::PMULHUWrr,
    X86::PMULHWrm,
    X86::PMULHWrr,
    X86::PMULLDrm,
    X86::PMULLDrm_int,
    X86::PMULLDrr,
    X86::PMULLDrr_int,
    X86::PMULLWrm,
    X86::PMULLWrr,
    X86::PMULUDQrm,
    X86::PMULUDQrr,
    X86::PSADBWrm,
    X86::PSADBWrr,
    X86::PSHUFBrm128,
    X86::PSHUFBrr128,
    X86::PSHUFHWmi,
    X86::PSHUFHWri,
    X86::PSHUFLWmi,
    X86::PSHUFLWri,
    X86::PSIGNBrm128,
    X86::PSIGNBrr128,
    X86::PSIGNDrm128,
    X86::PSIGNDrr128,
    X86::PSIGNWrm128,
    X86::PSIGNWrr128,
    X86::PSLLDQri,
    X86::PSLLDri,
    X86::PSLLDrm,
    X86::PSLLDrr,
    X86::PSLLQri,
    X86::PSLLQrm,
    X86::PSLLQrr,
    X86::PSLLWri,
    X86::PSLLWrm,
    X86::PSLLWrr,
    X86::PSRADri,
    X86::PSRADrm,
    X86::PSRADrr,
    X86::PSRAWri,
    X86::PSRAWrm,
    X86::PSRAWrr,
    X86::PSRLDQri,
    X86::PSRLDri,
    X86::PSRLDrm,
    X86::PSRLDrr,
    X86::PSRLQri,
    X86::PSRLQrm,
    X86::PSRLQrr,
    X86::PSRLWri,
    X86::PSRLWrm,
    X86::PSRLWrr,
    X86::PSUBBrm,
    X86::PSUBBrr,
    X86::PSUBDrm,
    X86::PSUBDrr,
    X86::PSUBQrm,
    X86::PSUBQrr,
    X86::PSUBSBrm,
    X86::PSUBSBrr,
    X86::PSUBSWrm,
    X86::PSUBSWrr,
    X86::PSUBUSBrm,
    X86::PSUBUSBrr,
    X86::PSUBUSWrm,
    X86::PSUBUSWrr,
    X86::PSUBWrm,
    X86::PSUBWrr,
    X86::PUNPCKHBWrm,
    X86::PUNPCKHBWrr,
    X86::PUNPCKHWDrm,
    X86::PUNPCKHWDrr,
    X86::PUNPCKLBWrm,
    X86::PUNPCKLBWrr,
    X86::PUNPCKLWDrm,
    X86::PUNPCKLWDrr,
  };

  // Instructions that execute in the packed single domain.
  static const unsigned PackedSingleInstrs[] = {
    X86::ADDPSrm,
    X86::ADDPSrr,
    X86::ADDSUBPSrm,
    X86::ADDSUBPSrr,
    X86::BLENDPSrmi,
    X86::BLENDPSrri,
    X86::BLENDVPSrm0,
    X86::BLENDVPSrr0,
    X86::CMPPSrmi,
    X86::CMPPSrri,
    X86::DIVPSrm,
    X86::DIVPSrr,
    X86::DPPSrmi,
    X86::DPPSrri,
    X86::EXTRACTPSmr,
    X86::EXTRACTPSrr,
    X86::HADDPSrm,
    X86::HADDPSrr,
    X86::HSUBPSrm,
    X86::HSUBPSrr,
    X86::INSERTPSrm,
    X86::INSERTPSrr,
    X86::MAXPSrm,
    X86::MAXPSrm_Int,
    X86::MAXPSrr,
    X86::MAXPSrr_Int,
    X86::MINPSrm,
    X86::MINPSrm_Int,
    X86::MINPSrr,
    X86::MINPSrr_Int,
    X86::MOVHLPSrr,
    X86::MOVHPSmr,
    X86::MOVHPSrm,
    X86::MOVLHPSrr,
    X86::MOVLPSmr,
    X86::MOVLPSrm,
    X86::MOVMSKPSrr,
    X86::MOVNTPSmr_Int,
    X86::MOVSHDUPrm,
    X86::MOVSHDUPrr,
    X86::MOVSLDUPrm,
    X86::MOVSLDUPrr,
    X86::MOVUPSmr_Int,
    X86::MOVUPSrm_Int,
    X86::MULPSrm,
    X86::MULPSrr,
    X86::RCPPSm,
    X86::RCPPSm_Int,
    X86::RCPPSr,
    X86::RCPPSr_Int,
    X86::ROUNDPSm_Int,
    X86::ROUNDPSr_Int,
    X86::RSQRTPSm,
    X86::RSQRTPSm_Int,
    X86::RSQRTPSr,
    X86::RSQRTPSr_Int,
    X86::SQRTPSm,
    X86::SQRTPSm_Int,
    X86::SQRTPSr,
    X86::SQRTPSr_Int,
    X86::SUBPSrm,
    X86::SUBPSrr,
  };

  // Instructions that execute in the packed double domain.
  static const unsigned PackedDoubleInstrs[] = {
    X86::ADDPDrm,
    X86::ADDPDrr,
    X86::ADDSUBPDrm,
    X86::ADDSUBPDrr,
    X86::BLENDPDrmi,
    X86::BLENDPDrri,
    X86::BLENDVPDrm0,
    X86::BLENDVPDrr0,
    X86::CMPPDrmi,
    X86::CMPPDrri,
    X86::DIVPDrm,
    X86::DIVPDrr,
    X86::DPPDrmi,
    X86::DPPDrri,
    X86::HADDPDrm,
    X86::HADDPDrr,
    X86::HSUBPDrm,
    X86::HSUBPDrr,
    X86::MAXPDrm,
    X86::MAXPDrm_Int,
    X86::MAXPDrr,
    X86::MAXPDrr_Int,
    X86::MINPDrm,
    X86::MINPDrm_Int,
    X86::MINPDrr,
    X86::MINPDrr_Int,
    X86::MOVHPDmr,
    X86::MOVHPDrm,
    X86::MOVLPDmr,
    X86::MOVLPDrm,
    X86::MOVMSKPDrr,
    X86::MOVNTPDmr_Int,
    X86::MOVUPDmr_Int,
    X86::MOVUPDrm_Int,
    X86::MULPDrm,
    X86::MULPDrr,
    X86::ROUNDPDm_Int,
    X86::ROUNDPDr_Int,
    X86::SQRTPDm,
    X86::SQRTPDm_Int,
    X86::SQRTPDr,
    X86::SQRTPDr_Int,
    X86::SUBPDrm,
    X86::SUBPDrr,
  };

  // Add non-negative entries for forcing instructions.
  for (unsigned i = 0, e = array_lengthof(PackedIntInstrs); i != e; ++i)
    SSEInstrDomainTable.insert(std::make_pair(PackedIntInstrs[i],
                                              PackedInt));
  for (unsigned i = 0, e = array_lengthof(PackedSingleInstrs); i != e; ++i)
    SSEInstrDomainTable.insert(std::make_pair(PackedSingleInstrs[i],
                                              PackedSingle));
  for (unsigned i = 0, e = array_lengthof(PackedDoubleInstrs); i != e; ++i)
    SSEInstrDomainTable.insert(std::make_pair(PackedDoubleInstrs[i],
                                              PackedDouble));

  // Add row number + 1 for replaceable instructions.
  for (unsigned i = 0, e = array_lengthof(ReplaceableInstrs); i != e; ++i)
    for (unsigned c = 0; c != 3; ++c)
    SSEInstrDomainTable.insert(std::make_pair(ReplaceableInstrs[i][c],
                                              c + 4*(i+1)));
}

X86InstrInfo::SSEDomain X86InstrInfo::GetSSEDomain(const MachineInstr *MI,
                                                 const unsigned *&equiv) const {
  DenseMap<unsigned,unsigned>::const_iterator i =
    SSEInstrDomainTable.find(MI->getOpcode());
  if (i == SSEInstrDomainTable.end())
    return NotSSEDomain;
  unsigned value = i->second;
  if (value/4)
    equiv = ReplaceableInstrs[value/4 - 1];
  else
    equiv = 0;
  return SSEDomain(value & 3);
}
