//==-- llvm/Target/TargetSelectionDAGInfo.h - SelectionDAG Info --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TargetSelectionDAGInfo class, which targets can
// subclass to parameterize the SelectionDAG lowering and instruction
// selection process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSELECTIONDAGINFO_H
#define LLVM_TARGET_TARGETSELECTIONDAGINFO_H

namespace llvm {

//===----------------------------------------------------------------------===//
/// TargetSelectionDAGLowering - Targets can subclass this to parameterize the
/// SelectionDAG lowering and instruction selection process.
///
class TargetSelectionDAGInfo {
  TargetSelectionDAGInfo(const TargetSelectionDAGInfo &); // DO NOT IMPLEMENT
  void operator=(const TargetSelectionDAGInfo &);         // DO NOT IMPLEMENT

public:
  TargetSelectionDAGInfo();
  virtual ~TargetSelectionDAGInfo();
};

} // end llvm namespace

#endif
