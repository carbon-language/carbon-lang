//===-- WebAssemblyUtilities - WebAssembly Utility Functions ---*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the WebAssembly-specific
/// utility functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_UTILS_WEBASSEMBLYUTILITIES_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_UTILS_WEBASSEMBLYUTILITIES_H

#include "llvm/Support/CommandLine.h"

namespace llvm {

class MachineBasicBlock;
class MachineInstr;
class MachineOperand;
class MCContext;
class MCSymbolWasm;
class WebAssemblyFunctionInfo;
class WebAssemblySubtarget;

namespace WebAssembly {

bool isChild(const MachineInstr &MI, const WebAssemblyFunctionInfo &MFI);
bool mayThrow(const MachineInstr &MI);

// Exception handling / setjmp-longjmp handling command-line options
extern cl::opt<bool> WasmEnableEmEH;   // asm.js-style EH
extern cl::opt<bool> WasmEnableEmSjLj; // asm.js-style SjLJ
extern cl::opt<bool> WasmEnableEH;     // EH using Wasm EH instructions
extern cl::opt<bool> WasmEnableSjLj;   // SjLj using Wasm EH instructions

// Exception-related function names
extern const char *const ClangCallTerminateFn;
extern const char *const CxaBeginCatchFn;
extern const char *const CxaRethrowFn;
extern const char *const StdTerminateFn;
extern const char *const PersonalityWrapperFn;

/// Returns the operand number of a callee, assuming the argument is a call
/// instruction.
const MachineOperand &getCalleeOp(const MachineInstr &MI);

/// Returns the __indirect_function_table, for use in call_indirect and in
/// function bitcasts.
MCSymbolWasm *
getOrCreateFunctionTableSymbol(MCContext &Ctx,
                               const WebAssemblySubtarget *Subtarget);

/// Returns the __funcref_call_table, for use in funcref calls when lowered to
/// table.set + call_indirect.
MCSymbolWasm *
getOrCreateFuncrefCallTableSymbol(MCContext &Ctx,
                                  const WebAssemblySubtarget *Subtarget);

/// Find a catch instruction from an EH pad. Returns null if no catch
/// instruction found or the catch is in an invalid location.
MachineInstr *findCatch(MachineBasicBlock *EHPad);

} // end namespace WebAssembly

} // end namespace llvm

#endif
