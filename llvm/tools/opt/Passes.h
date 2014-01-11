//===- Passes.h - Parsing, selection, and running of passes -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Interfaces for producing common pass manager configurations and parsing
/// textual pass specifications.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OPT_PASSES_H
#define LLVM_TOOLS_OPT_PASSES_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class ModulePassManager;

/// \brief Parse a textual pass pipeline description into a \c ModulePassManager.
///
/// The format of the textual pass pipeline description looks something like:
///
///   module(function(instcombine,sroa),dce,cgscc(inliner,function(...)),...)
///
/// Pass managers have ()s describing the nest structure of passes. All passes
/// are comma separated. As a special shortcut, if the very first pass is not
/// a module pass (as a module pass manager is), this will automatically form
/// the shortest stack of pass managers that allow inserting that first pass.
/// So, assuming function passes 'fpassN', CGSCC passes 'cgpassN', and loop passes
/// 'lpassN', all of these are valid:
///
///   fpass1,fpass2,fpass3
///   cgpass1,cgpass2,cgpass3
///   lpass1,lpass2,lpass3
///
/// And they are equivalent to the following (resp.):
///
///   module(function(fpass1,fpass2,fpass3))
///   module(cgscc(cgpass1,cgpass2,cgpass3))
///   module(function(loop(lpass1,lpass2,lpass3)))
///
/// This shortcut is especially useful for debugging and testing small pass
/// combinations. Note that these shortcuts don't introduce any other magic. If
/// the sequence of passes aren't all the exact same kind of pass, it will be
/// an error. You cannot mix different levels implicitly, you must explicitly
/// form a pass manager in which to nest passes.
bool parsePassPipeline(ModulePassManager &MPM, StringRef PipelineText);

}

#endif
