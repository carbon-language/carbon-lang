//===--- Passes/StackPointerTracking.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "StackPointerTracking.h"

namespace llvm {
namespace bolt {

StackPointerTracking::StackPointerTracking(const BinaryContext &BC,
                                           BinaryFunction &BF)
    : StackPointerTrackingBase<StackPointerTracking>(BC, BF) {}

} // end namespace bolt
} // end namespace llvm

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &OS,
                                    const std::pair<int, int> &Val) {
  OS << Val.first << ", " << Val.second;
  return OS;
}
