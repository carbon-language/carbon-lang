//===-- SystemZSelectionDAGInfo.h - SystemZ SelectionDAG Info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SystemZ subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZSELECTIONDAGINFO_H
#define SYSTEMZSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class SystemZSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  SystemZSelectionDAGInfo();
  ~SystemZSelectionDAGInfo();
};

}

#endif
