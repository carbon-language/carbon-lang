//===- NewPMDriver.h - Function to drive opt with the new PM ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// A single function which is called to drive the opt behavior for the new
/// PassManager.
///
/// This is only in a separate TU with a header to avoid including all of the
/// old pass manager headers and the new pass manager headers into the same
/// file. Eventually all of the routines here will get folded back into
/// opt.cpp.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OPT_NEW_PM_DRIVER_H
#define LLVM_TOOLS_OPT_NEW_PM_DRIVER_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class LLVMContext;
class Module;
class tool_output_file;

/// \brief Driver function to run the new pass manager over a module.
///
/// This function only exists factored away from opt.cpp in order to prevent
/// inclusion of the new pass manager headers and the old headers into the same
/// file. It's interface is consequentially somewhat ad-hoc, but will go away
/// when the transition finishes.
bool runPassPipeline(StringRef Arg0, LLVMContext &Context, Module &M,
                     tool_output_file *Out, StringRef PassPipeline,
                     bool NoOutput);
}

#endif
