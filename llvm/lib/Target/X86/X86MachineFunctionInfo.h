//====- X86MachineFuctionInfo.h - X86 machine function info -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares X86-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef X86MACHINEFUNCTIONINFO_H
#define X86MACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

enum NameDecorationStyle {
  None,
  StdCall,
  FastCall
};
  
/// X86MachineFunctionInfo - This class is derived from MachineFunction private
/// X86 target-specific information for each MachineFunction.
class X86MachineFunctionInfo : public MachineFunctionInfo {
  /// ForceFramePointer - True if the function is required to use of frame
  /// pointer for reasons other than it containing dynamic allocation or 
  /// that FP eliminatation is turned off. For example, Cygwin main function
  /// contains stack pointer re-alignment code which requires FP.
  bool ForceFramePointer;

  /// CalleeSavedFrameSize - Size of the callee-saved register portion of the
  /// stack frame in bytes.
  unsigned CalleeSavedFrameSize;

  /// BytesToPopOnReturn - amount of bytes function pops on return.
  /// Used on windows platform for stdcall & fastcall name decoration
  unsigned BytesToPopOnReturn;

  /// If the function requires additional name decoration, DecorationStyle holds
  /// the right way to do so.
  NameDecorationStyle DecorationStyle;
  
public:
  X86MachineFunctionInfo() : ForceFramePointer(false),
                             CalleeSavedFrameSize(0),
                             BytesToPopOnReturn(0),
                             DecorationStyle(None) {}
  
  X86MachineFunctionInfo(MachineFunction &MF) : ForceFramePointer(false),
                                                CalleeSavedFrameSize(0),
                                                BytesToPopOnReturn(0),
                                                DecorationStyle(None) {}
  
  bool getForceFramePointer() const { return ForceFramePointer;} 
  void setForceFramePointer(bool forceFP) { ForceFramePointer = forceFP; }

  unsigned getCalleeSavedFrameSize() const { return CalleeSavedFrameSize; }
  void setCalleeSavedFrameSize(unsigned bytes) { CalleeSavedFrameSize = bytes; }

  unsigned getBytesToPopOnReturn() const { return BytesToPopOnReturn; }
  void setBytesToPopOnReturn (unsigned bytes) { BytesToPopOnReturn = bytes;}

  NameDecorationStyle getDecorationStyle() const { return DecorationStyle; }
  void setDecorationStyle(NameDecorationStyle style) { DecorationStyle = style;}
  
};
} // End llvm namespace

#endif
