//===-- X86SelectionDAGInfo.h - X86 SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef X86SELECTIONDAGINFO_H
#define X86SELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class X86SelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  X86SelectionDAGInfo();
  ~X86SelectionDAGInfo();
};

}

#endif
