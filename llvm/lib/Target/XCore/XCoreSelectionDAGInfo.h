//===-- XCoreSelectionDAGInfo.h - XCore SelectionDAG Info -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the XCore subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef XCORESELECTIONDAGINFO_H
#define XCORESELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class XCoreSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  XCoreSelectionDAGInfo();
  ~XCoreSelectionDAGInfo();
};

}

#endif
