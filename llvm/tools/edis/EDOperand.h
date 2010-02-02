//===-EDOperand.h - LLVM Enhanced Disassembler ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the Enhanced Disassembly library's 
// operand class.  The operand is responsible for allowing evaluation given a
// particular register context.
//
//===----------------------------------------------------------------------===//

#ifndef EDOperand_
#define EDOperand_

#include "llvm-c/EnhancedDisassembly.h"

/// EDOperand - Encapsulates a single operand, which can be evaluated by the
///   client
struct EDOperand {
  /// The parent disassembler
  const EDDisassembler &Disassembler;
  /// The parent instruction
  const EDInst &Inst;
  
  /// The index of the operand in the EDInst
  unsigned int OpIndex;
  /// The index of the first component of the operand in the MCInst
  unsigned int MCOpIndex;
  
  /// Constructor - Initializes an EDOperand
  ///
  /// @arg disassembler - The disassembler responsible for the operand
  /// @arg inst         - The instruction containing this operand
  /// @arg opIndex      - The index of the operand in inst
  /// @arg mcOpIndex    - The index of the operand in the original MCInst
  EDOperand(const EDDisassembler &disassembler,
            const EDInst &inst,
            unsigned int opIndex,
            unsigned int &mcOpIndex);
  ~EDOperand();
  
  /// evaluate - Returns the numeric value of an operand to the extent possible,
  ///   returning 0 on success or -1 if there was some problem (such as a 
  ///   register not being readable)
  ///
  /// @arg result   - A reference whose target is filled in with the value of
  ///                 the operand (the address if it is a memory operand)
  /// @arg callback - A function to call to obtain register values
  /// @arg arg      - An opaque argument to pass to callback
  int evaluate(uint64_t &result,
               EDRegisterReaderCallback callback,
               void *arg);
  
#ifdef __BLOCKS__
  /// evaluate - Like evaluate for a callback, but uses a block instead
  int evaluate(uint64_t &result,
               EDRegisterBlock_t regBlock);
#endif
};

#endif
