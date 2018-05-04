//===--- Aligner.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Aligner.h"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltOptCategory;

cl::opt<bool>
UseCompactAligner("use-compact-aligner",
  cl::desc("Use compact approach for aligning functions"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
AlignFunctions("align-functions",
  cl::desc("align functions at a given value (relocation mode)"),
  cl::init(64),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
AlignFunctionsMaxBytes("align-functions-max-bytes",
  cl::desc("maximum number of bytes to use to align functions"),
  cl::init(32),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

} // end namespace opts

namespace llvm {
namespace bolt {

namespace {

// Align function to the specified byte-boundary (typically, 64) offsetting
// the fuction by not more than the corresponding value
void alignMaxBytes(BinaryFunction &Function) {
  Function.setAlignment(opts::AlignFunctions);
  Function.setMaxAlignmentBytes(opts::AlignFunctionsMaxBytes);
  Function.setMaxColdAlignmentBytes(opts::AlignFunctionsMaxBytes);
}

// Align function to the specified byte-boundary (typically, 64) offsetting
// the fuction by not more than the minimum over
// -- the size of the function
// -- the specified number of bytes
void alignCompact(BinaryContext &BC, BinaryFunction &Function) {
  size_t HotSize = 0;
  size_t ColdSize = 0;
  for (const auto *BB : Function.layout()) {
    if (BB->isCold())
      ColdSize += BC.computeCodeSize(BB->begin(), BB->end());
    else
      HotSize += BC.computeCodeSize(BB->begin(), BB->end());
  }

  Function.setAlignment(opts::AlignFunctions);
  if (HotSize > 0)
    Function.setMaxAlignmentBytes(
      std::min(size_t(opts::AlignFunctionsMaxBytes), HotSize));

  // using the same option, max-align-bytes, both for cold and hot parts of the
  // functions, as aligning cold functions typically does not affect performance
  if (ColdSize > 0)
    Function.setMaxColdAlignmentBytes(
      std::min(size_t(opts::AlignFunctionsMaxBytes), ColdSize));
}

} // end anonymous namespace

void AlignerPass::runOnFunctions(BinaryContext &BC,
                                 std::map<uint64_t, BinaryFunction> &BFs,
                                 std::set<uint64_t> &LargeFunctions) {
  if (!BC.HasRelocations)
    return;

  for (auto &It : BFs) {
    auto &Function = It.second;
    if (opts::UseCompactAligner)
      alignCompact(BC, Function);
    else
      alignMaxBytes(Function);
  }
}

} // end namespace bolt
} // end namespace llvm
