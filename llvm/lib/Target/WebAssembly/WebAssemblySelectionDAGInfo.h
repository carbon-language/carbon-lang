//=- WebAssemblySelectionDAGInfo.h - WebAssembly SelectionDAG Info -*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the WebAssembly subclass for
/// TargetSelectionDAGInfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class WebAssemblySelectionDAGInfo final : public TargetSelectionDAGInfo {
public:
  ~WebAssemblySelectionDAGInfo() override;
};

} // end namespace llvm

#endif
