//===----- JITDwarfEmitter.cpp - Write dwarf tables into memory -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a JITDwarfEmitter object that is used by the JIT to
// write dwarf tables to memory.
//
//===----------------------------------------------------------------------===//

#include "JIT.h"
#include "JITDwarfEmitter.h"
#include "llvm/Function.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

JITDwarfEmitter::JITDwarfEmitter(JIT& theJit) : Jit(theJit) {}


unsigned char* JITDwarfEmitter::EmitDwarfTable(MachineFunction& F, 
                                               JITCodeEmitter& jce,
                                               unsigned char* StartFunction,
                                               unsigned char* EndFunction) {
  const TargetMachine& TM = F.getTarget();
  TD = TM.getTargetData();
  needsIndirectEncoding = TM.getMCAsmInfo()->getNeedsIndirectEncoding();
  stackGrowthDirection = TM.getFrameInfo()->getStackGrowthDirection();
  RI = TM.getRegisterInfo();
  JCE = &jce;
  
  unsigned char* ExceptionTable = EmitExceptionTable(&F, StartFunction,
                                                     EndFunction);
      
  unsigned char* Result = 0;
  unsigned char* EHFramePtr = 0;

  const std::vector<Function *> Personalities = MMI->getPersonalities();
  EHFramePtr = EmitCommonEHFrame(Personalities[MMI->getPersonalityIndex()]);

  Result = EmitEHFrame(Personalities[MMI->getPersonalityIndex()], EHFramePtr,
                       StartFunction, EndFunction, ExceptionTable);
  
  return Result;
}


void 
JITDwarfEmitter::EmitFrameMoves(intptr_t BaseLabelPtr,
                                const std::vector<MachineMove> &Moves) const {
  unsigned PointerSize = TD->getPointerSize();
  int stackGrowth = stackGrowthDirection == TargetFrameInfo::StackGrowsUp ?
          PointerSize : -PointerSize;
  bool IsLocal = false;
  unsigned BaseLabelID = 0;

  for (unsigned i = 0, N = Moves.size(); i < N; ++i) {
    const MachineMove &Move = Moves[i];
    unsigned LabelID = Move.getLabelID();
    
    if (LabelID) {
      LabelID = MMI->MappedLabel(LabelID);
    
      // Throw out move if the label is invalid.
      if (!LabelID) continue;
    }
    
    intptr_t LabelPtr = 0;
    if (LabelID) LabelPtr = JCE->getLabelAddress(LabelID);

    const MachineLocation &Dst = Move.getDestination();
    const MachineLocation &Src = Move.getSource();
    
    // Advance row if new location.
    if (BaseLabelPtr && LabelID && (BaseLabelID != LabelID || !IsLocal)) {
      JCE->emitByte(dwarf::DW_CFA_advance_loc4);
      JCE->emitInt32(LabelPtr - BaseLabelPtr);
      
      BaseLabelID = LabelID; 
      BaseLabelPtr = LabelPtr;
      IsLocal = true;
    }
    
    // If advancing cfa.
    if (Dst.isReg() && Dst.getReg() == MachineLocation::VirtualFP) {
      if (!Src.isReg()) {
        if (Src.getReg() == MachineLocation::VirtualFP) {
          JCE->emitByte(dwarf::DW_CFA_def_cfa_offset);
        } else {
          JCE->emitByte(dwarf::DW_CFA_def_cfa);
          JCE->emitULEB128Bytes(RI->getDwarfRegNum(Src.getReg(), true));
        }
        
        int Offset = -Src.getOffset();
        
        JCE->emitULEB128Bytes(Offset);
      } else {
        llvm_unreachable("Machine move no supported yet.");
      }
    } else if (Src.isReg() &&
      Src.getReg() == MachineLocation::VirtualFP) {
      if (Dst.isReg()) {
        JCE->emitByte(dwarf::DW_CFA_def_cfa_register);
        JCE->emitULEB128Bytes(RI->getDwarfRegNum(Dst.getReg(), true));
      } else {
        llvm_unreachable("Machine move no supported yet.");
      }
    } else {
      unsigned Reg = RI->getDwarfRegNum(Src.getReg(), true);
      int Offset = Dst.getOffset() / stackGrowth;
      
      if (Offset < 0) {
        JCE->emitByte(dwarf::DW_CFA_offset_extended_sf);
        JCE->emitULEB128Bytes(Reg);
        JCE->emitSLEB128Bytes(Offset);
      } else if (Reg < 64) {
        JCE->emitByte(dwarf::DW_CFA_offset + Reg);
        JCE->emitULEB128Bytes(Offset);
      } else {
        JCE->emitByte(dwarf::DW_CFA_offset_extended);
        JCE->emitULEB128Bytes(Reg);
        JCE->emitULEB128Bytes(Offset);
      }
    }
  }
}

/// SharedTypeIds - How many leading type ids two landing pads have in common.
static unsigned SharedTypeIds(const LandingPadInfo *L,
                              const LandingPadInfo *R) {
  const std::vector<int> &LIds = L->TypeIds, &RIds = R->TypeIds;
  unsigned LSize = LIds.size(), RSize = RIds.size();
  unsigned MinSize = LSize < RSize ? LSize : RSize;
  unsigned Count = 0;

  for (; Count != MinSize; ++Count)
    if (LIds[Count] != RIds[Count])
      return Count;

  return Count;
}


/// PadLT - Order landing pads lexicographically by type id.
static bool PadLT(const LandingPadInfo *L, const LandingPadInfo *R) {
  const std::vector<int> &LIds = L->TypeIds, &RIds = R->TypeIds;
  unsigned LSize = LIds.size(), RSize = RIds.size();
  unsigned MinSize = LSize < RSize ? LSize : RSize;

  for (unsigned i = 0; i != MinSize; ++i)
    if (LIds[i] != RIds[i])
      return LIds[i] < RIds[i];

  return LSize < RSize;
}

namespace {

struct KeyInfo {
  static inline unsigned getEmptyKey() { return -1U; }
  static inline unsigned getTombstoneKey() { return -2U; }
  static unsigned getHashValue(const unsigned &Key) { return Key; }
  static bool isEqual(unsigned LHS, unsigned RHS) { return LHS == RHS; }
  static bool isPod() { return true; }
};

/// ActionEntry - Structure describing an entry in the actions table.
struct ActionEntry {
  int ValueForTypeID; // The value to write - may not be equal to the type id.
  int NextAction;
  struct ActionEntry *Previous;
};

/// PadRange - Structure holding a try-range and the associated landing pad.
struct PadRange {
  // The index of the landing pad.
  unsigned PadIndex;
  // The index of the begin and end labels in the landing pad's label lists.
  unsigned RangeIndex;
};

typedef DenseMap<unsigned, PadRange, KeyInfo> RangeMapType;

/// CallSiteEntry - Structure describing an entry in the call-site table.
struct CallSiteEntry {
  unsigned BeginLabel; // zero indicates the start of the function.
  unsigned EndLabel;   // zero indicates the end of the function.
  unsigned PadLabel;   // zero indicates that there is no landing pad.
  unsigned Action;
};

}

unsigned char* JITDwarfEmitter::EmitExceptionTable(MachineFunction* MF,
                                         unsigned char* StartFunction,
                                         unsigned char* EndFunction) const {
  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();

  const std::vector<GlobalVariable *> &TypeInfos = MMI->getTypeInfos();
  const std::vector<unsigned> &FilterIds = MMI->getFilterIds();
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();
  if (PadInfos.empty()) return 0;

  // Sort the landing pads in order of their type ids.  This is used to fold
  // duplicate actions.
  SmallVector<const LandingPadInfo *, 64> LandingPads;
  LandingPads.reserve(PadInfos.size());
  for (unsigned i = 0, N = PadInfos.size(); i != N; ++i)
    LandingPads.push_back(&PadInfos[i]);
  std::sort(LandingPads.begin(), LandingPads.end(), PadLT);

  // Negative type ids index into FilterIds, positive type ids index into
  // TypeInfos.  The value written for a positive type id is just the type
  // id itself.  For a negative type id, however, the value written is the
  // (negative) byte offset of the corresponding FilterIds entry.  The byte
  // offset is usually equal to the type id, because the FilterIds entries
  // are written using a variable width encoding which outputs one byte per
  // entry as long as the value written is not too large, but can differ.
  // This kind of complication does not occur for positive type ids because
  // type infos are output using a fixed width encoding.
  // FilterOffsets[i] holds the byte offset corresponding to FilterIds[i].
  SmallVector<int, 16> FilterOffsets;
  FilterOffsets.reserve(FilterIds.size());
  int Offset = -1;
  for(std::vector<unsigned>::const_iterator I = FilterIds.begin(),
    E = FilterIds.end(); I != E; ++I) {
    FilterOffsets.push_back(Offset);
    Offset -= MCAsmInfo::getULEB128Size(*I);
  }

  // Compute the actions table and gather the first action index for each
  // landing pad site.
  SmallVector<ActionEntry, 32> Actions;
  SmallVector<unsigned, 64> FirstActions;
  FirstActions.reserve(LandingPads.size());

  int FirstAction = 0;
  unsigned SizeActions = 0;
  for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
    const LandingPadInfo *LP = LandingPads[i];
    const std::vector<int> &TypeIds = LP->TypeIds;
    const unsigned NumShared = i ? SharedTypeIds(LP, LandingPads[i-1]) : 0;
    unsigned SizeSiteActions = 0;

    if (NumShared < TypeIds.size()) {
      unsigned SizeAction = 0;
      ActionEntry *PrevAction = 0;

      if (NumShared) {
        const unsigned SizePrevIds = LandingPads[i-1]->TypeIds.size();
        assert(Actions.size());
        PrevAction = &Actions.back();
        SizeAction = MCAsmInfo::getSLEB128Size(PrevAction->NextAction) +
          MCAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);
        for (unsigned j = NumShared; j != SizePrevIds; ++j) {
          SizeAction -= MCAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);
          SizeAction += -PrevAction->NextAction;
          PrevAction = PrevAction->Previous;
        }
      }

      // Compute the actions.
      for (unsigned I = NumShared, M = TypeIds.size(); I != M; ++I) {
        int TypeID = TypeIds[I];
        assert(-1-TypeID < (int)FilterOffsets.size() && "Unknown filter id!");
        int ValueForTypeID = TypeID < 0 ? FilterOffsets[-1 - TypeID] : TypeID;
        unsigned SizeTypeID = MCAsmInfo::getSLEB128Size(ValueForTypeID);

        int NextAction = SizeAction ? -(SizeAction + SizeTypeID) : 0;
        SizeAction = SizeTypeID + MCAsmInfo::getSLEB128Size(NextAction);
        SizeSiteActions += SizeAction;

        ActionEntry Action = {ValueForTypeID, NextAction, PrevAction};
        Actions.push_back(Action);

        PrevAction = &Actions.back();
      }

      // Record the first action of the landing pad site.
      FirstAction = SizeActions + SizeSiteActions - SizeAction + 1;
    } // else identical - re-use previous FirstAction

    FirstActions.push_back(FirstAction);

    // Compute this sites contribution to size.
    SizeActions += SizeSiteActions;
  }

  // Compute the call-site table.  Entries must be ordered by address.
  SmallVector<CallSiteEntry, 64> CallSites;

  RangeMapType PadMap;
  for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
    const LandingPadInfo *LandingPad = LandingPads[i];
    for (unsigned j=0, E = LandingPad->BeginLabels.size(); j != E; ++j) {
      unsigned BeginLabel = LandingPad->BeginLabels[j];
      assert(!PadMap.count(BeginLabel) && "Duplicate landing pad labels!");
      PadRange P = { i, j };
      PadMap[BeginLabel] = P;
    }
  }

  bool MayThrow = false;
  unsigned LastLabel = 0;
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
        I != E; ++I) {
    for (MachineBasicBlock::const_iterator MI = I->begin(), E = I->end();
          MI != E; ++MI) {
      if (!MI->isLabel()) {
        MayThrow |= MI->getDesc().isCall();
        continue;
      }

      unsigned BeginLabel = MI->getOperand(0).getImm();
      assert(BeginLabel && "Invalid label!");

      if (BeginLabel == LastLabel)
        MayThrow = false;

      RangeMapType::iterator L = PadMap.find(BeginLabel);

      if (L == PadMap.end())
        continue;

      PadRange P = L->second;
      const LandingPadInfo *LandingPad = LandingPads[P.PadIndex];

      assert(BeginLabel == LandingPad->BeginLabels[P.RangeIndex] &&
              "Inconsistent landing pad map!");

      // If some instruction between the previous try-range and this one may
      // throw, create a call-site entry with no landing pad for the region
      // between the try-ranges.
      if (MayThrow) {
        CallSiteEntry Site = {LastLabel, BeginLabel, 0, 0};
        CallSites.push_back(Site);
      }

      LastLabel = LandingPad->EndLabels[P.RangeIndex];
      CallSiteEntry Site = {BeginLabel, LastLabel,
        LandingPad->LandingPadLabel, FirstActions[P.PadIndex]};

      assert(Site.BeginLabel && Site.EndLabel && Site.PadLabel &&
              "Invalid landing pad!");

      // Try to merge with the previous call-site.
      if (CallSites.size()) {
        CallSiteEntry &Prev = CallSites.back();
        if (Site.PadLabel == Prev.PadLabel && Site.Action == Prev.Action) {
          // Extend the range of the previous entry.
          Prev.EndLabel = Site.EndLabel;
          continue;
        }
      }

      // Otherwise, create a new call-site.
      CallSites.push_back(Site);
    }
  }
  // If some instruction between the previous try-range and the end of the
  // function may throw, create a call-site entry with no landing pad for the
  // region following the try-range.
  if (MayThrow) {
    CallSiteEntry Site = {LastLabel, 0, 0, 0};
    CallSites.push_back(Site);
  }

  // Final tallies.
  unsigned SizeSites = CallSites.size() * (sizeof(int32_t) + // Site start.
                                            sizeof(int32_t) + // Site length.
                                            sizeof(int32_t)); // Landing pad.
  for (unsigned i = 0, e = CallSites.size(); i < e; ++i)
    SizeSites += MCAsmInfo::getULEB128Size(CallSites[i].Action);

  unsigned SizeTypes = TypeInfos.size() * TD->getPointerSize();

  unsigned TypeOffset = sizeof(int8_t) + // Call site format
                        // Call-site table length
                        MCAsmInfo::getULEB128Size(SizeSites) + 
                        SizeSites + SizeActions + SizeTypes;

  // Begin the exception table.
  JCE->emitAlignmentWithFill(4, 0);
  // Asm->EOL("Padding");

  unsigned char* DwarfExceptionTable = (unsigned char*)JCE->getCurrentPCValue();

  // Emit the header.
  JCE->emitByte(dwarf::DW_EH_PE_omit);
  // Asm->EOL("LPStart format (DW_EH_PE_omit)");
  JCE->emitByte(dwarf::DW_EH_PE_absptr);
  // Asm->EOL("TType format (DW_EH_PE_absptr)");
  JCE->emitULEB128Bytes(TypeOffset);
  // Asm->EOL("TType base offset");
  JCE->emitByte(dwarf::DW_EH_PE_udata4);
  // Asm->EOL("Call site format (DW_EH_PE_udata4)");
  JCE->emitULEB128Bytes(SizeSites);
  // Asm->EOL("Call-site table length");

  // Emit the landing pad site information.
  for (unsigned i = 0; i < CallSites.size(); ++i) {
    CallSiteEntry &S = CallSites[i];
    intptr_t BeginLabelPtr = 0;
    intptr_t EndLabelPtr = 0;

    if (!S.BeginLabel) {
      BeginLabelPtr = (intptr_t)StartFunction;
      JCE->emitInt32(0);
    } else {
      BeginLabelPtr = JCE->getLabelAddress(S.BeginLabel);
      JCE->emitInt32(BeginLabelPtr - (intptr_t)StartFunction);
    }

    // Asm->EOL("Region start");

    if (!S.EndLabel) {
      EndLabelPtr = (intptr_t)EndFunction;
      JCE->emitInt32((intptr_t)EndFunction - BeginLabelPtr);
    } else {
      EndLabelPtr = JCE->getLabelAddress(S.EndLabel);
      JCE->emitInt32(EndLabelPtr - BeginLabelPtr);
    }
    //Asm->EOL("Region length");

    if (!S.PadLabel) {
      JCE->emitInt32(0);
    } else {
      unsigned PadLabelPtr = JCE->getLabelAddress(S.PadLabel);
      JCE->emitInt32(PadLabelPtr - (intptr_t)StartFunction);
    }
    // Asm->EOL("Landing pad");

    JCE->emitULEB128Bytes(S.Action);
    // Asm->EOL("Action");
  }

  // Emit the actions.
  for (unsigned I = 0, N = Actions.size(); I != N; ++I) {
    ActionEntry &Action = Actions[I];

    JCE->emitSLEB128Bytes(Action.ValueForTypeID);
    //Asm->EOL("TypeInfo index");
    JCE->emitSLEB128Bytes(Action.NextAction);
    //Asm->EOL("Next action");
  }

  // Emit the type ids.
  for (unsigned M = TypeInfos.size(); M; --M) {
    GlobalVariable *GV = TypeInfos[M - 1];
    
    if (GV) {
      if (TD->getPointerSize() == sizeof(int32_t)) {
        JCE->emitInt32((intptr_t)Jit.getOrEmitGlobalVariable(GV));
      } else {
        JCE->emitInt64((intptr_t)Jit.getOrEmitGlobalVariable(GV));
      }
    } else {
      if (TD->getPointerSize() == sizeof(int32_t))
        JCE->emitInt32(0);
      else
        JCE->emitInt64(0);
    }
    // Asm->EOL("TypeInfo");
  }

  // Emit the filter typeids.
  for (unsigned j = 0, M = FilterIds.size(); j < M; ++j) {
    unsigned TypeID = FilterIds[j];
    JCE->emitULEB128Bytes(TypeID);
    //Asm->EOL("Filter TypeInfo index");
  }

  JCE->emitAlignmentWithFill(4, 0);

  return DwarfExceptionTable;
}

unsigned char*
JITDwarfEmitter::EmitCommonEHFrame(const Function* Personality) const {
  unsigned PointerSize = TD->getPointerSize();
  int stackGrowth = stackGrowthDirection == TargetFrameInfo::StackGrowsUp ?
          PointerSize : -PointerSize;
  
  unsigned char* StartCommonPtr = (unsigned char*)JCE->getCurrentPCValue();
  // EH Common Frame header
  JCE->allocateSpace(4, 0);
  unsigned char* FrameCommonBeginPtr = (unsigned char*)JCE->getCurrentPCValue();
  JCE->emitInt32((int)0);
  JCE->emitByte(dwarf::DW_CIE_VERSION);
  JCE->emitString(Personality ? "zPLR" : "zR");
  JCE->emitULEB128Bytes(1);
  JCE->emitSLEB128Bytes(stackGrowth);
  JCE->emitByte(RI->getDwarfRegNum(RI->getRARegister(), true));
  
  if (Personality) {
    // Augmentation Size: 3 small ULEBs of one byte each, and the personality
    // function which size is PointerSize.
    JCE->emitULEB128Bytes(3 + PointerSize); 
    
    // We set the encoding of the personality as direct encoding because we use
    // the function pointer. The encoding is not relative because the current
    // PC value may be bigger than the personality function pointer.
    if (PointerSize == 4) {
      JCE->emitByte(dwarf::DW_EH_PE_sdata4); 
      JCE->emitInt32(((intptr_t)Jit.getPointerToGlobal(Personality)));
    } else {
      JCE->emitByte(dwarf::DW_EH_PE_sdata8);
      JCE->emitInt64(((intptr_t)Jit.getPointerToGlobal(Personality)));
    }
    
    JCE->emitULEB128Bytes(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
    JCE->emitULEB128Bytes(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
      
  } else {
    JCE->emitULEB128Bytes(1);
    JCE->emitULEB128Bytes(dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4);
  }

  std::vector<MachineMove> Moves;
  RI->getInitialFrameState(Moves);
  EmitFrameMoves(0, Moves);

  JCE->emitAlignmentWithFill(PointerSize, dwarf::DW_CFA_nop);

  JCE->emitInt32At((uintptr_t*)StartCommonPtr,
                   (uintptr_t)((unsigned char*)JCE->getCurrentPCValue() -
                               FrameCommonBeginPtr));

  return StartCommonPtr;
}


unsigned char*
JITDwarfEmitter::EmitEHFrame(const Function* Personality,
                             unsigned char* StartCommonPtr,
                             unsigned char* StartFunction, 
                             unsigned char* EndFunction,
                             unsigned char* ExceptionTable) const {
  unsigned PointerSize = TD->getPointerSize();
  
  // EH frame header.
  unsigned char* StartEHPtr = (unsigned char*)JCE->getCurrentPCValue();
  JCE->allocateSpace(4, 0);
  unsigned char* FrameBeginPtr = (unsigned char*)JCE->getCurrentPCValue();
  // FDE CIE Offset
  JCE->emitInt32(FrameBeginPtr - StartCommonPtr);
  JCE->emitInt32(StartFunction - (unsigned char*)JCE->getCurrentPCValue());
  JCE->emitInt32(EndFunction - StartFunction);

  // If there is a personality and landing pads then point to the language
  // specific data area in the exception table.
  if (MMI->getPersonalityIndex()) {
    JCE->emitULEB128Bytes(4);
        
    if (!MMI->getLandingPads().empty()) {
      JCE->emitInt32(ExceptionTable - (unsigned char*)JCE->getCurrentPCValue());
    } else {
      JCE->emitInt32((int)0);
    }
  } else {
    JCE->emitULEB128Bytes(0);
  }
      
  // Indicate locations of function specific  callee saved registers in
  // frame.
  EmitFrameMoves((intptr_t)StartFunction, MMI->getFrameMoves());

  JCE->emitAlignmentWithFill(PointerSize, dwarf::DW_CFA_nop);

  // Indicate the size of the table
  JCE->emitInt32At((uintptr_t*)StartEHPtr,
                   (uintptr_t)((unsigned char*)JCE->getCurrentPCValue() -
                               StartEHPtr));

  // Double zeroes for the unwind runtime
  if (PointerSize == 8) {
    JCE->emitInt64(0);
    JCE->emitInt64(0);
  } else {
    JCE->emitInt32(0);
    JCE->emitInt32(0);
  }
  
  return StartEHPtr;
}

unsigned JITDwarfEmitter::GetDwarfTableSizeInBytes(MachineFunction& F,
                                         JITCodeEmitter& jce,
                                         unsigned char* StartFunction,
                                         unsigned char* EndFunction) {
  const TargetMachine& TM = F.getTarget();
  TD = TM.getTargetData();
  needsIndirectEncoding = TM.getMCAsmInfo()->getNeedsIndirectEncoding();
  stackGrowthDirection = TM.getFrameInfo()->getStackGrowthDirection();
  RI = TM.getRegisterInfo();
  JCE = &jce;
  unsigned FinalSize = 0;
  
  FinalSize += GetExceptionTableSizeInBytes(&F);
      
  const std::vector<Function *> Personalities = MMI->getPersonalities();
  FinalSize += 
    GetCommonEHFrameSizeInBytes(Personalities[MMI->getPersonalityIndex()]);

  FinalSize += GetEHFrameSizeInBytes(Personalities[MMI->getPersonalityIndex()],
                                     StartFunction);
  
  return FinalSize;
}

/// RoundUpToAlign - Add the specified alignment to FinalSize and returns
/// the new value.
static unsigned RoundUpToAlign(unsigned FinalSize, unsigned Alignment) {
  if (Alignment == 0) Alignment = 1;
  // Since we do not know where the buffer will be allocated, be pessimistic.
  return FinalSize + Alignment;
}
  
unsigned
JITDwarfEmitter::GetEHFrameSizeInBytes(const Function* Personality,
                                       unsigned char* StartFunction) const { 
  unsigned PointerSize = TD->getPointerSize();
  unsigned FinalSize = 0;
  // EH frame header.
  FinalSize += PointerSize;
  // FDE CIE Offset
  FinalSize += 3 * PointerSize;
  // If there is a personality and landing pads then point to the language
  // specific data area in the exception table.
  if (MMI->getPersonalityIndex()) {
    FinalSize += MCAsmInfo::getULEB128Size(4); 
    FinalSize += PointerSize;
  } else {
    FinalSize += MCAsmInfo::getULEB128Size(0);
  }
      
  // Indicate locations of function specific  callee saved registers in
  // frame.
  FinalSize += GetFrameMovesSizeInBytes((intptr_t)StartFunction,
                                        MMI->getFrameMoves());
      
  FinalSize = RoundUpToAlign(FinalSize, 4);
  
  // Double zeroes for the unwind runtime
  FinalSize += 2 * PointerSize;

  return FinalSize;
}

unsigned JITDwarfEmitter::GetCommonEHFrameSizeInBytes(const Function* Personality) 
  const {

  unsigned PointerSize = TD->getPointerSize();
  int stackGrowth = stackGrowthDirection == TargetFrameInfo::StackGrowsUp ?
          PointerSize : -PointerSize;
  unsigned FinalSize = 0; 
  // EH Common Frame header
  FinalSize += PointerSize;
  FinalSize += 4;
  FinalSize += 1;
  FinalSize += Personality ? 5 : 3; // "zPLR" or "zR"
  FinalSize += MCAsmInfo::getULEB128Size(1);
  FinalSize += MCAsmInfo::getSLEB128Size(stackGrowth);
  FinalSize += 1;
  
  if (Personality) {
    FinalSize += MCAsmInfo::getULEB128Size(7);
    
    // Encoding
    FinalSize+= 1;
    //Personality
    FinalSize += PointerSize;
    
    FinalSize += MCAsmInfo::getULEB128Size(dwarf::DW_EH_PE_pcrel);
    FinalSize += MCAsmInfo::getULEB128Size(dwarf::DW_EH_PE_pcrel);
      
  } else {
    FinalSize += MCAsmInfo::getULEB128Size(1);
    FinalSize += MCAsmInfo::getULEB128Size(dwarf::DW_EH_PE_pcrel);
  }

  std::vector<MachineMove> Moves;
  RI->getInitialFrameState(Moves);
  FinalSize += GetFrameMovesSizeInBytes(0, Moves);
  FinalSize = RoundUpToAlign(FinalSize, 4);
  return FinalSize;
}

unsigned
JITDwarfEmitter::GetFrameMovesSizeInBytes(intptr_t BaseLabelPtr,
                                  const std::vector<MachineMove> &Moves) const {
  unsigned PointerSize = TD->getPointerSize();
  int stackGrowth = stackGrowthDirection == TargetFrameInfo::StackGrowsUp ?
          PointerSize : -PointerSize;
  bool IsLocal = BaseLabelPtr;
  unsigned FinalSize = 0; 

  for (unsigned i = 0, N = Moves.size(); i < N; ++i) {
    const MachineMove &Move = Moves[i];
    unsigned LabelID = Move.getLabelID();
    
    if (LabelID) {
      LabelID = MMI->MappedLabel(LabelID);
    
      // Throw out move if the label is invalid.
      if (!LabelID) continue;
    }
    
    intptr_t LabelPtr = 0;
    if (LabelID) LabelPtr = JCE->getLabelAddress(LabelID);

    const MachineLocation &Dst = Move.getDestination();
    const MachineLocation &Src = Move.getSource();
    
    // Advance row if new location.
    if (BaseLabelPtr && LabelID && (BaseLabelPtr != LabelPtr || !IsLocal)) {
      FinalSize++;
      FinalSize += PointerSize;
      BaseLabelPtr = LabelPtr;
      IsLocal = true;
    }
    
    // If advancing cfa.
    if (Dst.isReg() && Dst.getReg() == MachineLocation::VirtualFP) {
      if (!Src.isReg()) {
        if (Src.getReg() == MachineLocation::VirtualFP) {
          ++FinalSize;
        } else {
          ++FinalSize;
          unsigned RegNum = RI->getDwarfRegNum(Src.getReg(), true);
          FinalSize += MCAsmInfo::getULEB128Size(RegNum);
        }
        
        int Offset = -Src.getOffset();
        
        FinalSize += MCAsmInfo::getULEB128Size(Offset);
      } else {
        llvm_unreachable("Machine move no supported yet.");
      }
    } else if (Src.isReg() &&
      Src.getReg() == MachineLocation::VirtualFP) {
      if (Dst.isReg()) {
        ++FinalSize;
        unsigned RegNum = RI->getDwarfRegNum(Dst.getReg(), true);
        FinalSize += MCAsmInfo::getULEB128Size(RegNum);
      } else {
        llvm_unreachable("Machine move no supported yet.");
      }
    } else {
      unsigned Reg = RI->getDwarfRegNum(Src.getReg(), true);
      int Offset = Dst.getOffset() / stackGrowth;
      
      if (Offset < 0) {
        ++FinalSize;
        FinalSize += MCAsmInfo::getULEB128Size(Reg);
        FinalSize += MCAsmInfo::getSLEB128Size(Offset);
      } else if (Reg < 64) {
        ++FinalSize;
        FinalSize += MCAsmInfo::getULEB128Size(Offset);
      } else {
        ++FinalSize;
        FinalSize += MCAsmInfo::getULEB128Size(Reg);
        FinalSize += MCAsmInfo::getULEB128Size(Offset);
      }
    }
  }
  return FinalSize;
}

unsigned 
JITDwarfEmitter::GetExceptionTableSizeInBytes(MachineFunction* MF) const {
  unsigned FinalSize = 0;

  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();

  const std::vector<GlobalVariable *> &TypeInfos = MMI->getTypeInfos();
  const std::vector<unsigned> &FilterIds = MMI->getFilterIds();
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();
  if (PadInfos.empty()) return 0;

  // Sort the landing pads in order of their type ids.  This is used to fold
  // duplicate actions.
  SmallVector<const LandingPadInfo *, 64> LandingPads;
  LandingPads.reserve(PadInfos.size());
  for (unsigned i = 0, N = PadInfos.size(); i != N; ++i)
    LandingPads.push_back(&PadInfos[i]);
  std::sort(LandingPads.begin(), LandingPads.end(), PadLT);

  // Negative type ids index into FilterIds, positive type ids index into
  // TypeInfos.  The value written for a positive type id is just the type
  // id itself.  For a negative type id, however, the value written is the
  // (negative) byte offset of the corresponding FilterIds entry.  The byte
  // offset is usually equal to the type id, because the FilterIds entries
  // are written using a variable width encoding which outputs one byte per
  // entry as long as the value written is not too large, but can differ.
  // This kind of complication does not occur for positive type ids because
  // type infos are output using a fixed width encoding.
  // FilterOffsets[i] holds the byte offset corresponding to FilterIds[i].
  SmallVector<int, 16> FilterOffsets;
  FilterOffsets.reserve(FilterIds.size());
  int Offset = -1;
  for(std::vector<unsigned>::const_iterator I = FilterIds.begin(),
    E = FilterIds.end(); I != E; ++I) {
    FilterOffsets.push_back(Offset);
    Offset -= MCAsmInfo::getULEB128Size(*I);
  }

  // Compute the actions table and gather the first action index for each
  // landing pad site.
  SmallVector<ActionEntry, 32> Actions;
  SmallVector<unsigned, 64> FirstActions;
  FirstActions.reserve(LandingPads.size());

  int FirstAction = 0;
  unsigned SizeActions = 0;
  for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
    const LandingPadInfo *LP = LandingPads[i];
    const std::vector<int> &TypeIds = LP->TypeIds;
    const unsigned NumShared = i ? SharedTypeIds(LP, LandingPads[i-1]) : 0;
    unsigned SizeSiteActions = 0;

    if (NumShared < TypeIds.size()) {
      unsigned SizeAction = 0;
      ActionEntry *PrevAction = 0;

      if (NumShared) {
        const unsigned SizePrevIds = LandingPads[i-1]->TypeIds.size();
        assert(Actions.size());
        PrevAction = &Actions.back();
        SizeAction = MCAsmInfo::getSLEB128Size(PrevAction->NextAction) +
          MCAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);
        for (unsigned j = NumShared; j != SizePrevIds; ++j) {
          SizeAction -= MCAsmInfo::getSLEB128Size(PrevAction->ValueForTypeID);
          SizeAction += -PrevAction->NextAction;
          PrevAction = PrevAction->Previous;
        }
      }

      // Compute the actions.
      for (unsigned I = NumShared, M = TypeIds.size(); I != M; ++I) {
        int TypeID = TypeIds[I];
        assert(-1-TypeID < (int)FilterOffsets.size() && "Unknown filter id!");
        int ValueForTypeID = TypeID < 0 ? FilterOffsets[-1 - TypeID] : TypeID;
        unsigned SizeTypeID = MCAsmInfo::getSLEB128Size(ValueForTypeID);

        int NextAction = SizeAction ? -(SizeAction + SizeTypeID) : 0;
        SizeAction = SizeTypeID + MCAsmInfo::getSLEB128Size(NextAction);
        SizeSiteActions += SizeAction;

        ActionEntry Action = {ValueForTypeID, NextAction, PrevAction};
        Actions.push_back(Action);

        PrevAction = &Actions.back();
      }

      // Record the first action of the landing pad site.
      FirstAction = SizeActions + SizeSiteActions - SizeAction + 1;
    } // else identical - re-use previous FirstAction

    FirstActions.push_back(FirstAction);

    // Compute this sites contribution to size.
    SizeActions += SizeSiteActions;
  }

  // Compute the call-site table.  Entries must be ordered by address.
  SmallVector<CallSiteEntry, 64> CallSites;

  RangeMapType PadMap;
  for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
    const LandingPadInfo *LandingPad = LandingPads[i];
    for (unsigned j=0, E = LandingPad->BeginLabels.size(); j != E; ++j) {
      unsigned BeginLabel = LandingPad->BeginLabels[j];
      assert(!PadMap.count(BeginLabel) && "Duplicate landing pad labels!");
      PadRange P = { i, j };
      PadMap[BeginLabel] = P;
    }
  }

  bool MayThrow = false;
  unsigned LastLabel = 0;
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
        I != E; ++I) {
    for (MachineBasicBlock::const_iterator MI = I->begin(), E = I->end();
          MI != E; ++MI) {
      if (!MI->isLabel()) {
        MayThrow |= MI->getDesc().isCall();
        continue;
      }

      unsigned BeginLabel = MI->getOperand(0).getImm();
      assert(BeginLabel && "Invalid label!");

      if (BeginLabel == LastLabel)
        MayThrow = false;

      RangeMapType::iterator L = PadMap.find(BeginLabel);

      if (L == PadMap.end())
        continue;

      PadRange P = L->second;
      const LandingPadInfo *LandingPad = LandingPads[P.PadIndex];

      assert(BeginLabel == LandingPad->BeginLabels[P.RangeIndex] &&
              "Inconsistent landing pad map!");

      // If some instruction between the previous try-range and this one may
      // throw, create a call-site entry with no landing pad for the region
      // between the try-ranges.
      if (MayThrow) {
        CallSiteEntry Site = {LastLabel, BeginLabel, 0, 0};
        CallSites.push_back(Site);
      }

      LastLabel = LandingPad->EndLabels[P.RangeIndex];
      CallSiteEntry Site = {BeginLabel, LastLabel,
        LandingPad->LandingPadLabel, FirstActions[P.PadIndex]};

      assert(Site.BeginLabel && Site.EndLabel && Site.PadLabel &&
              "Invalid landing pad!");

      // Try to merge with the previous call-site.
      if (CallSites.size()) {
        CallSiteEntry &Prev = CallSites.back();
        if (Site.PadLabel == Prev.PadLabel && Site.Action == Prev.Action) {
          // Extend the range of the previous entry.
          Prev.EndLabel = Site.EndLabel;
          continue;
        }
      }

      // Otherwise, create a new call-site.
      CallSites.push_back(Site);
    }
  }
  // If some instruction between the previous try-range and the end of the
  // function may throw, create a call-site entry with no landing pad for the
  // region following the try-range.
  if (MayThrow) {
    CallSiteEntry Site = {LastLabel, 0, 0, 0};
    CallSites.push_back(Site);
  }

  // Final tallies.
  unsigned SizeSites = CallSites.size() * (sizeof(int32_t) + // Site start.
                                            sizeof(int32_t) + // Site length.
                                            sizeof(int32_t)); // Landing pad.
  for (unsigned i = 0, e = CallSites.size(); i < e; ++i)
    SizeSites += MCAsmInfo::getULEB128Size(CallSites[i].Action);

  unsigned SizeTypes = TypeInfos.size() * TD->getPointerSize();

  unsigned TypeOffset = sizeof(int8_t) + // Call site format
                        // Call-site table length
                        MCAsmInfo::getULEB128Size(SizeSites) + 
                        SizeSites + SizeActions + SizeTypes;

  unsigned TotalSize = sizeof(int8_t) + // LPStart format
                       sizeof(int8_t) + // TType format
                       MCAsmInfo::getULEB128Size(TypeOffset) + // TType base offset
                       TypeOffset;

  unsigned SizeAlign = (4 - TotalSize) & 3;

  // Begin the exception table.
  FinalSize = RoundUpToAlign(FinalSize, 4);
  for (unsigned i = 0; i != SizeAlign; ++i) {
    ++FinalSize;
  }
  
  unsigned PointerSize = TD->getPointerSize();

  // Emit the header.
  ++FinalSize;
  // Asm->EOL("LPStart format (DW_EH_PE_omit)");
  ++FinalSize;
  // Asm->EOL("TType format (DW_EH_PE_absptr)");
  ++FinalSize;
  // Asm->EOL("TType base offset");
  ++FinalSize;
  // Asm->EOL("Call site format (DW_EH_PE_udata4)");
  ++FinalSize;
  // Asm->EOL("Call-site table length");

  // Emit the landing pad site information.
  for (unsigned i = 0; i < CallSites.size(); ++i) {
    CallSiteEntry &S = CallSites[i];

    // Asm->EOL("Region start");
    FinalSize += PointerSize;
    
    //Asm->EOL("Region length");
    FinalSize += PointerSize;

    // Asm->EOL("Landing pad");
    FinalSize += PointerSize;

    FinalSize += MCAsmInfo::getULEB128Size(S.Action);
    // Asm->EOL("Action");
  }

  // Emit the actions.
  for (unsigned I = 0, N = Actions.size(); I != N; ++I) {
    ActionEntry &Action = Actions[I];

    //Asm->EOL("TypeInfo index");
    FinalSize += MCAsmInfo::getSLEB128Size(Action.ValueForTypeID);
    //Asm->EOL("Next action");
    FinalSize += MCAsmInfo::getSLEB128Size(Action.NextAction);
  }

  // Emit the type ids.
  for (unsigned M = TypeInfos.size(); M; --M) {
    // Asm->EOL("TypeInfo");
    FinalSize += PointerSize;
  }

  // Emit the filter typeids.
  for (unsigned j = 0, M = FilterIds.size(); j < M; ++j) {
    unsigned TypeID = FilterIds[j];
    FinalSize += MCAsmInfo::getULEB128Size(TypeID);
    //Asm->EOL("Filter TypeInfo index");
  }
  
  FinalSize = RoundUpToAlign(FinalSize, 4);

  return FinalSize;
}
