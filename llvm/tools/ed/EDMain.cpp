//===-EDMain.cpp - LLVM Enhanced Disassembly C API ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the enhanced disassembler's public C API.
//
//===----------------------------------------------------------------------===//

#include "EDDisassembler.h"
#include "EDInst.h"
#include "EDOperand.h"
#include "EDToken.h"

#include "llvm-c/EnhancedDisassembly.h"

int EDGetDisassembler(EDDisassemblerRef *disassembler,
                      const char *triple,
                      EDAssemblySyntax_t syntax) {
  EDDisassembler::initialize();
  
  EDDisassemblerRef ret = EDDisassembler::getDisassembler(triple,
                                                          syntax);
  
  if (ret) {
    *disassembler = ret;
    return 0;
  }
  else {
    return -1;
  }
}

int EDGetRegisterName(const char** regName,
                      EDDisassemblerRef disassembler,
                      unsigned regID) {
  const char* name = disassembler->nameWithRegisterID(regID);
  if(!name)
    return -1;
  *regName = name;
  return 0;
}

int EDRegisterIsStackPointer(EDDisassemblerRef disassembler,
                             unsigned regID) {
  return disassembler->registerIsStackPointer(regID) ? 1 : 0;
}

int EDRegisterIsProgramCounter(EDDisassemblerRef disassembler,
                               unsigned regID) {
  return disassembler->registerIsProgramCounter(regID) ? 1 : 0;
}

unsigned int EDCreateInsts(EDInstRef *insts,
                           unsigned int count,
                           EDDisassemblerRef disassembler,
                           EDByteReaderCallback byteReader,
                           uint64_t address,
                           void *arg) {
  unsigned int index;
  
  for (index = 0; index < count; index++) {
    EDInst *inst = disassembler->createInst(byteReader, address, arg);
    
    if(!inst)
      return index;
    
    insts[index] = inst;
    address += inst->byteSize();
  }
  
  return count;
}

void EDReleaseInst(EDInstRef inst) {
  delete inst;
}

int EDInstByteSize(EDInstRef inst) {
  return inst->byteSize();
}

int EDGetInstString(const char **buf,
                    EDInstRef inst) {
  return inst->getString(*buf);
}

int EDInstID(unsigned *instID, EDInstRef inst) {
  *instID = inst->instID();
  return 0;
}

int EDInstIsBranch(EDInstRef inst) {
  return inst->isBranch();
}

int EDInstIsMove(EDInstRef inst) {
  return inst->isMove();
}

int EDBranchTargetID(EDInstRef inst) {
  return inst->branchTargetID();
}

int EDMoveSourceID(EDInstRef inst) {
  return inst->moveSourceID();
}

int EDMoveTargetID(EDInstRef inst) {
  return inst->moveTargetID();
}

int EDNumTokens(EDInstRef inst) {
  return inst->numTokens();
}

int EDGetToken(EDTokenRef *token,
               EDInstRef inst,
               int index) {
  return inst->getToken(*token, index);
}

int EDGetTokenString(const char **buf,
                     EDTokenRef token) {
  return token->getString(*buf);
}

int EDOperandIndexForToken(EDTokenRef token) {
  return token->operandID();
}

int EDTokenIsWhitespace(EDTokenRef token) {
  if(token->type() == EDToken::kTokenWhitespace)
    return 1;
  else
    return 0;
}

int EDTokenIsPunctuation(EDTokenRef token) {
  if(token->type() == EDToken::kTokenPunctuation)
    return 1;
  else
    return 0;
}

int EDTokenIsOpcode(EDTokenRef token) {
  if(token->type() == EDToken::kTokenOpcode)
    return 1;
  else
    return 0;
}

int EDTokenIsLiteral(EDTokenRef token) {
  if(token->type() == EDToken::kTokenLiteral)
    return 1;
  else
    return 0;
}

int EDTokenIsRegister(EDTokenRef token) {
  if(token->type() == EDToken::kTokenRegister)
    return 1;
  else
    return 0;
}

int EDTokenIsNegativeLiteral(EDTokenRef token) {
  if(token->type() != EDToken::kTokenLiteral)
    return -1;
  
  return token->literalSign();
}

int EDLiteralTokenAbsoluteValue(uint64_t *value,
                                EDTokenRef token) {
  if(token->type() != EDToken::kTokenLiteral)
    return -1;
  
  return token->literalAbsoluteValue(*value);
}

int EDRegisterTokenValue(unsigned *registerID,
                         EDTokenRef token) {
  if(token->type() != EDToken::kTokenRegister)
    return -1;
  
  return token->registerID(*registerID);
}

int EDNumOperands(EDInstRef inst) {
  return inst->numOperands();
}

int EDGetOperand(EDOperandRef *operand,
                 EDInstRef inst,
                 int index) {
  return inst->getOperand(*operand, index);
}

int EDEvaluateOperand(uint64_t *result,
                      EDOperandRef operand,
                      EDRegisterReaderCallback regReader,
                      void *arg) {
  return operand->evaluate(*result, regReader, arg);
}

#ifdef __BLOCKS__

struct ByteReaderWrapper {
  EDByteBlock_t byteBlock;
};

static int readerWrapperCallback(uint8_t *byte, 
                          uint64_t address,
                          void *arg) {
  struct ByteReaderWrapper *wrapper = (struct ByteReaderWrapper *)arg;
  return wrapper->byteBlock(byte, address);
}

unsigned int EDBlockCreateInsts(EDInstRef *insts,
                                int count,
                                EDDisassemblerRef disassembler,
                                EDByteBlock_t byteBlock,
                                uint64_t address) {
  struct ByteReaderWrapper wrapper;
  wrapper.byteBlock = byteBlock;
  
  return EDCreateInsts(insts,
                       count,
                       disassembler, 
                       readerWrapperCallback, 
                       address, 
                       (void*)&wrapper);
}

int EDBlockEvaluateOperand(uint64_t *result,
                           EDOperandRef operand,
                           EDRegisterBlock_t regBlock) {
  return operand->evaluate(*result, regBlock);
}

int EDBlockVisitTokens(EDInstRef inst,
                       EDTokenVisitor_t visitor) {
  return inst->visitTokens(visitor);
}

#else

extern "C" unsigned int EDBlockCreateInsts() {
  return 0;
}

extern "C" int EDBlockEvaluateOperand() {
  return -1;
}

extern "C" int EDBlockVisitTokens() {
  return -1;
}

#endif
