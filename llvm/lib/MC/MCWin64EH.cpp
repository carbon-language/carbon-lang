//===- lib/MC/MCWin64EH.cpp - MCWin64EH implementation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCWin64EH.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

// NOTE: All relocations generated here are 4-byte image-relative.

static uint8_t CountOfUnwindCodes(std::vector<MCWin64EHInstruction> &instArray){
  uint8_t count = 0;
  for (std::vector<MCWin64EHInstruction>::const_iterator I = instArray.begin(),
       E = instArray.end(); I != E; ++I) {
    switch (I->getOperation()) {
    case Win64EH::UOP_PushNonVol:
    case Win64EH::UOP_AllocSmall:
    case Win64EH::UOP_SetFPReg:
    case Win64EH::UOP_PushMachFrame:
      count += 1;
      break;
    case Win64EH::UOP_SaveNonVol:
    case Win64EH::UOP_SaveXMM128:
      count += 2;
      break;
    case Win64EH::UOP_SaveNonVolBig:
    case Win64EH::UOP_SaveXMM128Big:
      count += 3;
      break;
    case Win64EH::UOP_AllocLarge:
      if (I->getSize() > 512*1024-8)
        count += 3;
      else
        count += 2;
      break;
    }
  }
  return count;
}

static void EmitUnwindCode(MCStreamer &streamer, MCWin64EHInstruction &inst) {
  uint8_t b1, b2;
  uint16_t w;
  b2 = (inst.getOperation() & 0x0F) << 4;
  switch (inst.getOperation()) {
  case Win64EH::UOP_PushNonVol:
    streamer.EmitIntValue(0, 1);
    b2 |= inst.getRegister() & 0x0F;
    streamer.EmitIntValue(b2, 1);
    break;
  case Win64EH::UOP_AllocLarge:
    streamer.EmitIntValue(0, 1);
    if (inst.getSize() > 512*1024-8) {
      b2 |= 1;
      streamer.EmitIntValue(b2, 1);
      w = inst.getSize() & 0xFFF8;
      streamer.EmitIntValue(w, 2);
      w = inst.getSize() >> 16;
    } else {
      streamer.EmitIntValue(b2, 1);
      w = inst.getSize() >> 3;
    }
    streamer.EmitIntValue(w, 2);
    break;
  case Win64EH::UOP_AllocSmall:
    b2 |= (inst.getSize() >> 3) & 0x0F;
    streamer.EmitIntValue(0, 1);
    streamer.EmitIntValue(b2, 1);
    break;
  case Win64EH::UOP_SetFPReg:
    b1 = inst.getOffset() & 0xF0;
    streamer.EmitIntValue(b1, 1);
    streamer.EmitIntValue(b2, 1);
    break;
  case Win64EH::UOP_SaveNonVol:
  case Win64EH::UOP_SaveXMM128:
    b2 |= inst.getRegister() & 0x0F;
    streamer.EmitIntValue(0, 1);
    streamer.EmitIntValue(b2, 1);
    w = inst.getOffset() >> 3;
    if (inst.getOperation() == Win64EH::UOP_SaveXMM128)
      w >>= 1;
    streamer.EmitIntValue(w, 2);
    break;
  case Win64EH::UOP_SaveNonVolBig:
  case Win64EH::UOP_SaveXMM128Big:
    b2 |= inst.getRegister() & 0x0F;
    streamer.EmitIntValue(0, 1);
    streamer.EmitIntValue(b2, 1);
    if (inst.getOperation() == Win64EH::UOP_SaveXMM128Big)
      w = inst.getOffset() & 0xFFF0;
    else
      w = inst.getOffset() & 0xFFF8;
    streamer.EmitIntValue(w, 2);
    w = inst.getOffset() >> 16;
    streamer.EmitIntValue(w, 2);
    break;
  case Win64EH::UOP_PushMachFrame:
    if (inst.isPushCodeFrame())
      b2 |= 1;
    streamer.EmitIntValue(0, 1);
    streamer.EmitIntValue(b2, 1);
    break;
  }
}

static void EmitRuntimeFunction(MCStreamer &streamer,
                                const MCWin64EHUnwindInfo *info) {
  MCContext &context = streamer.getContext();

  streamer.EmitValueToAlignment(4);
  streamer.EmitValue(MCSymbolRefExpr::Create(info->Begin, context), 4);
  streamer.EmitValue(MCSymbolRefExpr::Create(info->End, context), 4);
  streamer.EmitValue(MCSymbolRefExpr::Create(info->Symbol, context), 4);
}

static void EmitUnwindInfo(MCStreamer &streamer, MCWin64EHUnwindInfo *info) {
  // If this UNWIND_INFO already has a symbol, it's already been emitted.
  if (info->Symbol) return;

  MCContext &context = streamer.getContext();
  streamer.EmitValueToAlignment(4);
  // Upper 3 bits are the version number (currently 1).
  uint8_t flags = 0x20;
  info->Symbol = context.CreateTempSymbol();
  streamer.EmitLabel(info->Symbol);

  if (info->ChainedParent)
    flags |= Win64EH::UNW_ChainInfo;
  else {
    if (info->HandlesUnwind)
      flags |= Win64EH::UNW_TerminateHandler;
    if (info->HandlesExceptions)
      flags |= Win64EH::UNW_ExceptionHandler;
  }
  streamer.EmitIntValue(flags, 1);

  // Build up the prolog size expression.
  const MCExpr *prologSize = MCBinaryExpr::CreateSub(MCSymbolRefExpr::Create(
                                                      info->PrologEnd, context),
                                                     MCSymbolRefExpr::Create(
                                                          info->Begin, context),
                                                     context);
  streamer.EmitAbsValue(prologSize, 1);

  uint8_t numCodes = CountOfUnwindCodes(info->Instructions);
  streamer.EmitIntValue(numCodes, 1);

  uint8_t frame = 0;
  if (info->LastFrameInst >= 0) {
    MCWin64EHInstruction &frameInst = info->Instructions[info->LastFrameInst];
    assert(frameInst.getOperation() == Win64EH::UOP_SetFPReg);
    frame = ((frameInst.getRegister() & 0x0F) << 4) |
            ((frameInst.getOffset() >> 4) & 0x0F);
  }
  streamer.EmitIntValue(frame, 1);

  // Emit unwind instructions (in reverse order).
  uint8_t numInst = info->Instructions.size();
  for (uint8_t c = 0; c < numInst; ++c) {
    MCWin64EHInstruction inst = info->Instructions.back();
    info->Instructions.pop_back();
    EmitUnwindCode(streamer, inst);
  }

  if (flags & Win64EH::UNW_ChainInfo)
    EmitRuntimeFunction(streamer, info->ChainedParent);
  else if (flags &(Win64EH::UNW_TerminateHandler|Win64EH::UNW_ExceptionHandler))
    streamer.EmitValue(MCSymbolRefExpr::Create(info->ExceptionHandler, context),
                       4);
}

void MCWin64EHUnwindEmitter::EmitUnwindInfo(MCStreamer &streamer,
                                            MCWin64EHUnwindInfo *info) {
  // Switch sections (the static function above is meant to be called from
  // here and from Emit().
  MCContext &context = streamer.getContext();
  const TargetAsmInfo &asmInfo = context.getTargetAsmInfo();
  const MCSection *xdataSect = asmInfo.getWin64EHTableSection();
  streamer.SwitchSection(xdataSect);

  llvm::EmitUnwindInfo(streamer, info);
}

void MCWin64EHUnwindEmitter::Emit(MCStreamer &streamer) {
  MCContext &context = streamer.getContext();
  // Emit the unwind info structs first.
  const TargetAsmInfo &asmInfo = context.getTargetAsmInfo();
  const MCSection *xdataSect = asmInfo.getWin64EHTableSection();
  streamer.SwitchSection(xdataSect);
  for (unsigned i = 0; i < streamer.getNumW64UnwindInfos(); ++i)
    llvm::EmitUnwindInfo(streamer, &streamer.getW64UnwindInfo(i));
  // Now emit RUNTIME_FUNCTION entries.
  const MCSection *pdataSect = asmInfo.getWin64EHFuncTableSection();
  streamer.SwitchSection(pdataSect);
  for (unsigned i = 0; i < streamer.getNumW64UnwindInfos(); ++i)
    EmitRuntimeFunction(streamer, &streamer.getW64UnwindInfo(i));
}

} // End of namespace llvm

