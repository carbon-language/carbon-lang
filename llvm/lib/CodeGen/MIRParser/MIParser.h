//===- MIParser.h - Machine Instructions Parser ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the function that parses the machine instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_MIRPARSER_MIPARSER_H
#define LLVM_LIB_CODEGEN_MIRPARSER_MIPARSER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class BasicBlock;
class MachineBasicBlock;
class MachineInstr;
class MachineFunction;
struct SlotMapping;
class SMDiagnostic;
class SourceMgr;

struct PerFunctionMIParsingState {
  DenseMap<unsigned, MachineBasicBlock *> MBBSlots;
  DenseMap<unsigned, unsigned> VirtualRegisterSlots;
  DenseMap<unsigned, int> FixedStackObjectSlots;
  DenseMap<unsigned, int> StackObjectSlots;
  DenseMap<unsigned, unsigned> ConstantPoolSlots;
  DenseMap<unsigned, unsigned> JumpTableSlots;
};

bool parseMachineInstr(MachineInstr *&MI, SourceMgr &SM, MachineFunction &MF,
                       StringRef Src, const PerFunctionMIParsingState &PFS,
                       const SlotMapping &IRSlots, SMDiagnostic &Error);

bool parseMBBReference(MachineBasicBlock *&MBB, SourceMgr &SM,
                       MachineFunction &MF, StringRef Src,
                       const PerFunctionMIParsingState &PFS,
                       const SlotMapping &IRSlots, SMDiagnostic &Error);

bool parseNamedRegisterReference(unsigned &Reg, SourceMgr &SM,
                                 MachineFunction &MF, StringRef Src,
                                 const PerFunctionMIParsingState &PFS,
                                 const SlotMapping &IRSlots,
                                 SMDiagnostic &Error);

bool parseVirtualRegisterReference(unsigned &Reg, SourceMgr &SM,
                                   MachineFunction &MF, StringRef Src,
                                   const PerFunctionMIParsingState &PFS,
                                   const SlotMapping &IRSlots,
                                   SMDiagnostic &Error);

bool parseIRBlockReference(const BasicBlock *&BB, SourceMgr &SM,
                           MachineFunction &MF, StringRef Src,
                           const PerFunctionMIParsingState &PFS,
                           const SlotMapping &IRSlots, SMDiagnostic &Error);

} // end namespace llvm

#endif
