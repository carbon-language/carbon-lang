//===--- ReorderSection.h - Profile based reordering of section data =========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_REORDER_DATA_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_REORDER_DATA_H

#include "BinaryPasses.h"
#include "BinarySection.h"

namespace llvm {
namespace bolt {

class ReorderData : public BinaryFunctionPass {
public:
  using DataOrder = std::vector<std::pair<BinaryData *, uint64_t>>;

private:
  DataOrder baseOrder(BinaryContext &BC,
                      const BinarySection &Section) const;

  /// Sort symbols by memory profiling data execution count.  The output
  /// is a vector of [address,count] pairs.
  std::pair<DataOrder, unsigned>
  sortedByCount(BinaryContext &BC, const BinarySection &Section) const;

  std::pair<DataOrder, unsigned>
  sortedByFunc(BinaryContext &BC,
               const BinarySection &Section,
               std::map<uint64_t, BinaryFunction> &BFs) const;

  void printOrder(const BinarySection &Section,
                  DataOrder::const_iterator Begin,
                  DataOrder::const_iterator End) const;

  /// Set the ordering of the section with \p SectionName.  \p NewOrder is a
  /// vector of [old address, size] pairs.  The new symbol order is implicit
  /// in the order of the vector.
  void setSectionOrder(BinaryContext &BC,
                       BinarySection &OutputSection,
                       DataOrder::iterator Begin,
                       DataOrder::iterator End);

  bool markUnmoveableSymbols(BinaryContext &BC,
                             BinarySection &Section) const;
public:
  explicit ReorderData() : BinaryFunctionPass(false) {}

  const char *getName() const override {
    return "reorder-data";
  }

  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
