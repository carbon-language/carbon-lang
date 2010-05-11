//===-- BlackfinSelectionDAGInfo.h - Blackfin SelectionDAG Info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Blackfin subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef BLACKFINSELECTIONDAGINFO_H
#define BLACKFINSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class BlackfinTargetMachine;

class BlackfinSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit BlackfinSelectionDAGInfo(const BlackfinTargetMachine &TM);
  ~BlackfinSelectionDAGInfo();
};

}

#endif
