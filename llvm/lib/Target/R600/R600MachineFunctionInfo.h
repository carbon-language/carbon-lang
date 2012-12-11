//===-- R600MachineFunctionInfo.h - R600 Machine Function Info ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef R600MACHINEFUNCTIONINFO_H
#define R600MACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include <vector>

namespace llvm {

class R600MachineFunctionInfo : public MachineFunctionInfo {

public:
  R600MachineFunctionInfo(const MachineFunction &MF);
  std::vector<unsigned> ReservedRegs;
  SDNode *Outputs[16];
  SDNode *StreamOutputs[64][4];
  bool HasLinearInterpolation;
  bool HasPerspectiveInterpolation;

  unsigned GetIJLinearIndex() const;
  unsigned GetIJPerspectiveIndex() const;

};

} // End llvm namespace

#endif //R600MACHINEFUNCTIONINFO_H
