//===-- SparcV8.h - Top-level interface for SparcV8 representation -*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// SparcV8 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_SPARCV8_H
#define TARGET_SPARCV8_H

#include <iosfwd>
#include <cassert>

namespace llvm {

  class FunctionPass;
  class TargetMachine;

  FunctionPass *createSparcV8ISelDag(TargetMachine &TM);

  FunctionPass *createSparcV8CodePrinterPass(std::ostream &OS,
                                             TargetMachine &TM);
  FunctionPass *createSparcV8DelaySlotFillerPass(TargetMachine &TM);
  FunctionPass *createSparcV8FPMoverPass(TargetMachine &TM);
} // end namespace llvm;

// Defines symbolic names for SparcV8 registers.  This defines a mapping from
// register name to register number.
//
#include "SparcV8GenRegisterNames.inc"

// Defines symbolic names for the SparcV8 instructions.
//
#include "SparcV8GenInstrNames.inc"


namespace llvm {
  // Enums corresponding to SparcV8 condition codes, both icc's and fcc's.  These
  // values must be kept in sync with the ones in the .td file.
  namespace V8CC {
    enum CondCodes {
      //ICC_A   =  8   ,  // Always
      //ICC_N   =  0   ,  // Never
      ICC_NE  =  9   ,  // Not Equal
      ICC_E   =  1   ,  // Equal
      ICC_G   = 10   ,  // Greater
      ICC_LE  =  2   ,  // Less or Equal
      ICC_GE  = 11   ,  // Greater or Equal
      ICC_L   =  3   ,  // Less
      ICC_GU  = 12   ,  // Greater Unsigned
      ICC_LEU =  4   ,  // Less or Equal Unsigned
      ICC_CC  = 13   ,  // Carry Clear/Great or Equal Unsigned
      ICC_CS  =  5   ,  // Carry Set/Less Unsigned
      ICC_POS = 14   ,  // Positive
      ICC_NEG =  6   ,  // Negative
      ICC_VC  = 15   ,  // Overflow Clear
      ICC_VS  =  7   ,  // Overflow Set
      
      //FCC_A   =  8+16,  // Always
      //FCC_N   =  0+16,  // Never
      FCC_U   =  7+16,  // Unordered
      FCC_G   =  6+16,  // Greater
      FCC_UG  =  5+16,  // Unordered or Greater
      FCC_L   =  4+16,  // Less
      FCC_UL  =  3+16,  // Unordered or Less
      FCC_LG  =  2+16,  // Less or Greater
      FCC_NE  =  1+16,  // Not Equal
      FCC_E   =  9+16,  // Equal
      FCC_UE  = 10+16,  // Unordered or Equal
      FCC_GE  = 11+16,  // Greater or Equal
      FCC_UGE = 12+16,  // Unordered or Greater or Equal
      FCC_LE  = 13+16,  // Less or Equal
      FCC_ULE = 14+16,  // Unordered or Less or Equal
      FCC_O   = 15+16,  // Ordered
    };
  }
  
  static unsigned SPARCCondCodeToBranchInstr(V8CC::CondCodes CC) {
    switch (CC) {
    default: assert(0 && "Unknown condition code");
    case V8CC::ICC_NE:  return V8::BNE;
    case V8CC::ICC_E:   return V8::BE;
    case V8CC::ICC_G:   return V8::BG;
    case V8CC::ICC_LE:  return V8::BLE;
    case V8CC::ICC_GE:  return V8::BGE;
    case V8CC::ICC_L:   return V8::BL;
    case V8CC::ICC_GU:  return V8::BGU;
    case V8CC::ICC_LEU: return V8::BLEU;
    case V8CC::ICC_CC:  return V8::BCC;
    case V8CC::ICC_CS:  return V8::BCS;
    case V8CC::ICC_POS: return V8::BPOS;
    case V8CC::ICC_NEG: return V8::BNEG;
    case V8CC::ICC_VC:  return V8::BVC;
    case V8CC::ICC_VS:  return V8::BVS;
    case V8CC::FCC_U:   return V8::FBU;
    case V8CC::FCC_G:   return V8::FBG;
    case V8CC::FCC_UG:  return V8::FBUG;
    case V8CC::FCC_L:   return V8::FBL;
    case V8CC::FCC_UL:  return V8::FBUL;
    case V8CC::FCC_LG:  return V8::FBLG;
    case V8CC::FCC_NE:  return V8::FBNE;
    case V8CC::FCC_E:   return V8::FBE;
    case V8CC::FCC_UE:  return V8::FBUE;
    case V8CC::FCC_GE:  return V8::FBGE;
    case V8CC::FCC_UGE: return V8::FBUGE;
    case V8CC::FCC_LE:  return V8::FBLE;
    case V8CC::FCC_ULE: return V8::FBULE;
    case V8CC::FCC_O:   return V8::FBO;
    }       
  }
  
  static const char *SPARCCondCodeToString(V8CC::CondCodes CC) {
    switch (CC) {
    default: assert(0 && "Unknown condition code");
    case V8CC::ICC_NE:  return "ne";
    case V8CC::ICC_E:   return "e";
    case V8CC::ICC_G:   return "g";
    case V8CC::ICC_LE:  return "le";
    case V8CC::ICC_GE:  return "ge";
    case V8CC::ICC_L:   return "l";
    case V8CC::ICC_GU:  return "gu";
    case V8CC::ICC_LEU: return "leu";
    case V8CC::ICC_CC:  return "cc";
    case V8CC::ICC_CS:  return "cs";
    case V8CC::ICC_POS: return "pos";
    case V8CC::ICC_NEG: return "neg";
    case V8CC::ICC_VC:  return "vc";
    case V8CC::ICC_VS:  return "vs";
    case V8CC::FCC_U:   return "u";
    case V8CC::FCC_G:   return "g";
    case V8CC::FCC_UG:  return "ug";
    case V8CC::FCC_L:   return "l";
    case V8CC::FCC_UL:  return "ul";
    case V8CC::FCC_LG:  return "lg";
    case V8CC::FCC_NE:  return "ne";
    case V8CC::FCC_E:   return "e";
    case V8CC::FCC_UE:  return "ue";
    case V8CC::FCC_GE:  return "ge";
    case V8CC::FCC_UGE: return "uge";
    case V8CC::FCC_LE:  return "le";
    case V8CC::FCC_ULE: return "ule";
    case V8CC::FCC_O:   return "o";
    }       
  }
}  // end namespace llvm
#endif
