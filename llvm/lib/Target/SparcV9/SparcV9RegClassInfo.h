//===-- SparcV9RegClassInfo.h - Register class def'ns for SparcV9 ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the register classes used by the SparcV9 target description.
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_REG_CLASS_INFO_H
#define SPARC_REG_CLASS_INFO_H

#include "llvm/Target/TargetRegInfo.h"

namespace llvm {

//-----------------------------------------------------------------------------
// Integer Register Class
//-----------------------------------------------------------------------------

struct SparcV9IntRegClass : public TargetRegClassInfo {
  SparcV9IntRegClass(unsigned ID) 
    : TargetRegClassInfo(ID, NumOfAvailRegs, NumOfAllRegs) {  }

  void colorIGNode(IGNode *Node,
                   const std::vector<bool> &IsColorUsedArr) const;

  inline bool isRegVolatile(int Reg) const {
    return (Reg < (int)StartOfNonVolatileRegs); 
  }

  inline bool modifiedByCall(int Reg) const {
    return Reg==(int)ModifiedByCall;
  }

  enum {   // colors possible for a LR (in preferred order)
     // --- following colors are volatile across function calls
     // %g0 can't be used for coloring - always 0
     o0, o1, o2, o3, o4, o5, o7,  // %o0-%o5, 

     // %o6 is sp, 
     // all %0's can get modified by a call

     // --- following colors are NON-volatile across function calls
     l0, l1, l2, l3, l4, l5, l6, l7,    //  %l0-%l7
     i0, i1, i2, i3, i4, i5,         // %i0-%i5: i's need not be preserved 
      
     // %i6 is the fp - so not allocated
     // %i7 is the ret address by convention - can be used for others

     // max # of colors reg coloring  can allocate (NumOfAvailRegs)

     // --- following colors are not available for allocation within this phase
     // --- but can appear for pre-colored ranges 

     i6, i7, g0,  g1, g2, g3, g4, g5, g6, g7, o6,

     NumOfAllRegs,  // Must be first AFTER registers...
     
     //*** NOTE: If we decide to use some %g regs, they are volatile
     // (see sparc64ABI)
     // Move the %g regs from the end of the enumeration to just above the
     // enumeration of %o0 (change StartOfAllRegs below)
     // change isRegVloatile method below
     // Also change IntRegNames above.

     // max # of colors reg coloring  can allocate
     NumOfAvailRegs = i6,

     StartOfNonVolatileRegs = l0,
     StartOfAllRegs = o0,
     
     ModifiedByCall = o7,
  };

  const char * const getRegName(unsigned reg) const;
};




//-----------------------------------------------------------------------------
// Float Register Class
//-----------------------------------------------------------------------------

class SparcV9FloatRegClass : public TargetRegClassInfo {
  int findFloatColor(const LiveRange *LR, unsigned Start,
		     unsigned End,
                     const std::vector<bool> &IsColorUsedArr) const;
public:
  SparcV9FloatRegClass(unsigned ID) 
    : TargetRegClassInfo(ID, NumOfAvailRegs, NumOfAllRegs) {}

  // This method marks the registers used for a given register number.
  // This marks a single register for Float regs, but the R,R+1 pair
  // for double-precision registers.
  // 
  virtual void markColorsUsed(unsigned RegInClass,
                              int UserRegType,
                              int RegTypeWanted,
                              std::vector<bool> &IsColorUsedArr) const;
  
  // This method finds unused registers of the specified register type,
  // using the given "used" flag array IsColorUsedArr.  It checks a single
  // entry in the array directly for float regs, and checks the pair [R,R+1]
  // for double-precision registers
  // It returns -1 if no unused color is found.
  // 
  virtual int findUnusedColor(int RegTypeWanted,
                              const std::vector<bool> &IsColorUsedArr) const;

  void colorIGNode(IGNode *Node,
                   const std::vector<bool> &IsColorUsedArr) const;

  // according to  SparcV9 64 ABI, all %fp regs are volatile
  inline bool isRegVolatile(int Reg) const { return true; }

  enum {
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, 
    f10, f11, f12, f13, f14, f15, f16, f17, f18, f19,
    f20, f21, f22, f23, f24, f25, f26, f27, f28, f29,
    f30, f31, f32, f33, f34, f35, f36, f37, f38, f39,
    f40, f41, f42, f43, f44, f45, f46, f47, f48, f49,
    f50, f51, f52, f53, f54, f55, f56, f57, f58, f59,
    f60, f61, f62, f63,

    // there are 64 regs alltogether but only 32 regs can be allocated at
    // a time.
    //
    NumOfAvailRegs = 32,
    NumOfAllRegs = 64,

    StartOfNonVolatileRegs = f32,
    StartOfAllRegs = f0,
  };

  const char * const getRegName(unsigned reg) const;
};




//-----------------------------------------------------------------------------
// Int CC Register Class
// Only one integer cc register is available. However, this register is
// referred to as %xcc or %icc when instructions like subcc are executed but 
// referred to as %ccr (i.e., %xcc . %icc") when this register is moved
// into an integer register using RD or WR instrcutions. So, three ids are
// allocated for the three names.
//-----------------------------------------------------------------------------

struct SparcV9IntCCRegClass : public TargetRegClassInfo {
  SparcV9IntCCRegClass(unsigned ID) 
    : TargetRegClassInfo(ID, 1, 3) {  }
  
  void colorIGNode(IGNode *Node,
                   const std::vector<bool> &IsColorUsedArr) const;

  // according to  SparcV9 64 ABI,  %ccr is volatile
  //
  inline bool isRegVolatile(int Reg) const { return true; }

  enum {
    xcc, icc, ccr   // only one is available - see the note above
  };

  const char * const getRegName(unsigned reg) const;
};


//-----------------------------------------------------------------------------
// Float CC Register Class
// Only 4 Float CC registers are available for allocation.
//-----------------------------------------------------------------------------

struct SparcV9FloatCCRegClass : public TargetRegClassInfo {
  SparcV9FloatCCRegClass(unsigned ID) 
    : TargetRegClassInfo(ID, 4, 5) {  }

  void colorIGNode(IGNode *Node,
                   const std::vector<bool> &IsColorUsedArr) const;
  
  // according to  SparcV9 64 ABI, all %fp CC regs are volatile
  //
  inline bool isRegVolatile(int Reg) const { return true; }

  enum {
    fcc0, fcc1, fcc2, fcc3, fsr         // fsr is not used in allocation
  };                                    // but has a name in getRegName()

  const char * const getRegName(unsigned reg) const;
};

//-----------------------------------------------------------------------------
// SparcV9 special register class.  These registers are not used for allocation
// but are used as arguments of some instructions.
//-----------------------------------------------------------------------------

struct SparcV9SpecialRegClass : public TargetRegClassInfo {
  SparcV9SpecialRegClass(unsigned ID) 
    : TargetRegClassInfo(ID, 0, 1) {  }

  void colorIGNode(IGNode *Node,
                   const std::vector<bool> &IsColorUsedArr) const {
    assert(0 && "SparcV9SpecialRegClass should never be used for allocation");
  }
  
  // all currently included special regs are volatile
  inline bool isRegVolatile(int Reg) const { return true; }

  enum {
    fsr                                 // floating point state register
  };

  const char * const getRegName(unsigned reg) const;
};

} // End llvm namespace

#endif
