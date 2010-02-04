//===-EDOperand.cpp - LLVM Enhanced Disassembler --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Enhanced Disassembly library's operand class.  The
// operand is responsible for allowing evaluation given a particular register 
// context.
//
//===----------------------------------------------------------------------===//

#include "EDDisassembler.h"
#include "EDInst.h"
#include "EDOperand.h"

#include "llvm/MC/MCInst.h"

using namespace llvm;

EDOperand::EDOperand(const EDDisassembler &disassembler,
                     const EDInst &inst,
                     unsigned int opIndex,
                     unsigned int &mcOpIndex) :
  Disassembler(disassembler),
  Inst(inst),
  OpIndex(opIndex),
  MCOpIndex(mcOpIndex) {
  unsigned int numMCOperands = 0;
    
  if(Disassembler.Key.Arch == Triple::x86 ||
     Disassembler.Key.Arch == Triple::x86_64) {
    uint8_t operandFlags = inst.ThisInstInfo->operandFlags[opIndex];
    
    if (operandFlags & kOperandFlagImmediate) {
      numMCOperands = 1;
    }
    else if (operandFlags & kOperandFlagRegister) {
      numMCOperands = 1;
    }
    else if (operandFlags & kOperandFlagMemory) {
      if (operandFlags & kOperandFlagPCRelative) {
        numMCOperands = 1;
      }
      else {
        numMCOperands = 5;
      }
    }
    else if (operandFlags & kOperandFlagEffectiveAddress) {
      numMCOperands = 4;
    }
  }
    
  mcOpIndex += numMCOperands;
}

EDOperand::~EDOperand() {
}

int EDOperand::evaluate(uint64_t &result,
                        EDRegisterReaderCallback callback,
                        void *arg) {
  if (Disassembler.Key.Arch == Triple::x86 ||
      Disassembler.Key.Arch == Triple::x86_64) {
    uint8_t operandFlags = Inst.ThisInstInfo->operandFlags[OpIndex];
    
    if (operandFlags & kOperandFlagImmediate) {
      result = Inst.Inst->getOperand(MCOpIndex).getImm();
      return 0;
    }
    if (operandFlags & kOperandFlagRegister) {
      unsigned reg = Inst.Inst->getOperand(MCOpIndex).getReg();
      return callback(&result, reg, arg);
    }
    if (operandFlags & kOperandFlagMemory ||
        operandFlags & kOperandFlagEffectiveAddress){
      if(operandFlags & kOperandFlagPCRelative) {
        int64_t displacement = Inst.Inst->getOperand(MCOpIndex).getImm();
        
        uint64_t ripVal;
        
        // TODO fix how we do this
        
        if (callback(&ripVal, Disassembler.registerIDWithName("RIP"), arg))
          return -1;
        
        result = ripVal + displacement;
        return 0;
      }
      else {
        unsigned baseReg = Inst.Inst->getOperand(MCOpIndex).getReg();
        uint64_t scaleAmount = Inst.Inst->getOperand(MCOpIndex+1).getImm();
        unsigned indexReg = Inst.Inst->getOperand(MCOpIndex+2).getReg();
        int64_t displacement = Inst.Inst->getOperand(MCOpIndex+3).getImm();
        //unsigned segmentReg = Inst.Inst->getOperand(MCOpIndex+4).getReg();
      
        uint64_t addr = 0;
        
        if(baseReg) {
          uint64_t baseVal;
          if (callback(&baseVal, baseReg, arg))
            return -1;
          addr += baseVal;
        }
        
        if(indexReg) {
          uint64_t indexVal;
          if (callback(&indexVal, indexReg, arg))
            return -1;
          addr += (scaleAmount * indexVal);
        }
        
        addr += displacement;
        
        result = addr;
        return 0;
      }
    }
    return -1;
  }
  
  return -1;
}

int EDOperand::isRegister() {
  return(Inst.ThisInstInfo->operandFlags[OpIndex] & kOperandFlagRegister);
}

unsigned EDOperand::regVal() {
  return Inst.Inst->getOperand(MCOpIndex).getReg(); 
}

int EDOperand::isImmediate() {
  return(Inst.ThisInstInfo->operandFlags[OpIndex] & kOperandFlagImmediate);
}

uint64_t EDOperand::immediateVal() {
  return Inst.Inst->getOperand(MCOpIndex).getImm();
}

int EDOperand::isMemory() {
  return(Inst.ThisInstInfo->operandFlags[OpIndex] & kOperandFlagMemory);
}

#ifdef __BLOCKS__
struct RegisterReaderWrapper {
  EDRegisterBlock_t regBlock;
};

int readerWrapperCallback(uint64_t *value, 
                          unsigned regID, 
                          void *arg) {
  struct RegisterReaderWrapper *wrapper = (struct RegisterReaderWrapper *)arg;
  return wrapper->regBlock(value, regID);
}

int EDOperand::evaluate(uint64_t &result,
                        EDRegisterBlock_t regBlock) {
  struct RegisterReaderWrapper wrapper;
  wrapper.regBlock = regBlock;
  return evaluate(result, 
                  readerWrapperCallback, 
                  (void*)&wrapper);
}
#endif
