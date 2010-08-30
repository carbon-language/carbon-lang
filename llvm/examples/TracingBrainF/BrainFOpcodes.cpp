//===-- BrainFOpcodes.cpp - BrainF interpreter opcodes ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//

#include "BrainFVM.h"
#include <cstdio>

opcode_func_t *BytecodeArray = 0;
size_t *JumpMap = 0;
uint8_t executed = 0;

BrainFTraceRecorder *Recorder = 0;

void op_plus(size_t pc, uint8_t *data) {
  Recorder->record_simple(pc, '+');
  *data += 1;
  BytecodeArray[pc+1](pc+1, data);
}

void op_minus(size_t pc, uint8_t *data) {
  Recorder->record_simple(pc, '-');
  *data -= 1;
  BytecodeArray[pc+1](pc+1, data);
}

void op_left(size_t pc, uint8_t *data) {
  Recorder->record_simple(pc, '<');
  BytecodeArray[pc+1](pc+1, data-1);
}

void op_right(size_t pc, uint8_t *data) {
  Recorder->record_simple(pc, '>');
  BytecodeArray[pc+1](pc+1, data+1);
}

void op_put(size_t pc, uint8_t *data) {
  Recorder->record_simple(pc, '.');
  putchar(*data);
  BytecodeArray[pc+1](pc+1, data);
}

void op_get(size_t pc, uint8_t *data) {
  Recorder->record_simple(pc, ',');
  *data = getchar();
  BytecodeArray[pc+1](pc+1, data);
}

void op_if(size_t pc, uint8_t *data) {
  Recorder->record(pc, '[');
  size_t new_pc = pc+1;
  if (!*data) new_pc = JumpMap[pc]+1;
  BytecodeArray[new_pc](new_pc, data);
}

void op_back(size_t pc, uint8_t *data) {
  Recorder->record_simple(pc, ']');
  size_t new_pc = JumpMap[pc];
  BytecodeArray[new_pc](new_pc, data);
}

void op_end(size_t, uint8_t *) {
  return;
}
