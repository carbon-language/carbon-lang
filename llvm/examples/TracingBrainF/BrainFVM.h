//===-- BrainFVM.h - BrainF interpreter header ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//

#ifndef BRAINF_VM_H
#define BRAINF_VM_H

#include "BrainF.h"
#include "stdint.h"
#include <cstring>

/// opcode_func_t - A function pointer signature for all opcode functions.
typedef void(*opcode_func_t)(size_t pc, uint8_t* data);

/// BytecodeArray - An array of function pointers representing the
/// source program.  Indexed by PC address.
extern opcode_func_t *BytecodeArray;

/// JumpMap - An array of on-the-side data used by the interpreter.
/// Indexed by PC address.
extern size_t *JumpMap;

/// executed - A flag indicating whether the preceding opcode was evaluated
/// within a compiled trace execution.  Used by the trace recorder.
extern uint8_t executed;

/// Recorder - The trace recording engine.
extern BrainFTraceRecorder *Recorder;

/// op_plus - Implements the '+' instruction.
void op_plus(size_t, uint8_t*);

/// op_minus - Implements the '-' instruction.
void op_minus(size_t, uint8_t*);

// op_left - Implements the '<' instruction.
void op_left(size_t, uint8_t*);

// op_right - Implements the '>' instruction.
void op_right(size_t, uint8_t*);

// op_put - Implements the '.' instruction.
void op_put(size_t, uint8_t*);

// op_get - Implements the ',' instruction.
void op_get(size_t, uint8_t*);

// op_if - Implements the '[' instruction.
void op_if(size_t, uint8_t*);

// op_back - Implements the ']' instruction.
void op_back(size_t, uint8_t*);

// op_end - Terminates an execution.
void op_end(size_t, uint8_t*);


#endif