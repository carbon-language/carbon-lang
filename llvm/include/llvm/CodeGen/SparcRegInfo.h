/* Title:   SparcRegClassInfo.h    -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Aug 20, 01
   Purpose: Contains the description of integer register class of Sparc
*/


#ifndef SPARC_INT_REG_CLASS_H
#define SPARC_INT_REG_CLASS_H

#include "llvm/Target/RegInfo.h"
#include "llvm/CodeGen/IGNode.h"

//-----------------------------------------------------------------------------
// Integer Register Class
//-----------------------------------------------------------------------------


// Int register names in same order as enum in class SparcIntRegOrder

static string const IntRegNames[] = 
  {       "g1", "g2", "g3", "g4", "g5", "g6", "g7",
    "o0", "o1", "o2", "o3", "o4", "o5",       "o7",
    "l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7",
    "i0", "i1", "i2", "i3", "i4", "i5", 
    "g0", "i6", "i7",  "o6" }; 



class SparcIntRegOrder{ 

 public:

  enum RegsInPrefOrder   // colors possible for a LR (in preferred order)
   { 
     // --- following colors are volatile across function calls
     // %g0 can't be used for coloring - always 0
                     
     g1, g2, g3, g4, g5, g6, g7,  //%g1-%g7  
     o0, o1, o2, o3, o4, o5, o7,  // %o0-%o5, 

     // %o6 is sp, 
     // all %0's can get modified by a call

     // --- following colors are NON-volatile across function calls
      
     l0, l1, l2, l3, l4, l5, l6, l7,    //  %l0-%l7
     i0, i1, i2, i3, i4, i5,            // %i0-%i5: i's need not be preserved 
      
     // %i6 is the fp - so not allocated
     // %i7 is the ret address - can be used if saved

     // max # of colors reg coloring  can allocate (NumOfAvailRegs)

     // --- following colors are not available for allocation within this phase
     // --- but can appear for pre-colored ranges 

     g0, i6, i7,  o6

 

   };

  // max # of colors reg coloring  can allocate
  static unsigned int const NumOfAvailRegs = g0;

  static unsigned int const StartOfNonVolatileRegs = l0;
  static unsigned int const StartOfAllRegs = g1;
  static unsigned int const NumOfAllRegs = o6 + 1; 


  static const string  getRegName(const unsigned reg) {
    assert( reg < NumOfAllRegs );
    return IntRegNames[reg];
  }

};



class SparcIntRegClass : public MachineRegClassInfo
{
 public:

  SparcIntRegClass(unsigned ID) 
    : MachineRegClassInfo(ID, 
			  SparcIntRegOrder::NumOfAvailRegs,
			  SparcIntRegOrder::NumOfAllRegs)
    {  }

  void colorIGNode(IGNode * Node, bool IsColorUsedArr[] ) const;

};

//-----------------------------------------------------------------------------
// Float Register Class
//-----------------------------------------------------------------------------

static string const FloatRegNames[] = 
  {    
    "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",  "f8",  "f9", 
    "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19",
    "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29",
    "f30", "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38", "f39",
    "f40", "f41", "f42", "f43", "f44", "f45", "f46", "f47", "f48", "f49",
    "f50", "f51", "f52", "f53", "f54", "f55", "f56", "f57", "f58", "f59",
    "f60", "f61", "f62", "f63"
  };


class SparcFloatRegOrder{ 

 public:

  enum RegsInPrefOrder {

    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, 
    f10, f11, f12, f13, f14, f15, f16, f17, f18, f19,
    f20, f21, f22, f23, f24, f25, f26, f27, f28, f29,
    f30, f31, f32, f33, f34, f35, f36, f37, f38, f39,
    f40, f41, f42, f43, f44, f45, f46, f47, f48, f49,
    f50, f51, f52, f53, f54, f55, f56, f57, f58, f59,
    f60, f61, f62, f63

  };

  // there are 64 regs alltogether but only 32 regs can be allocated at
  // a time.

  static unsigned int const NumOfAvailRegs = 32;
  static unsigned int const NumOfAllRegs = 64;

  static unsigned int const StartOfNonVolatileRegs = f6;
  static unsigned int const StartOfAllRegs = f0;


  static const string  getRegName(const unsigned reg) {
    assert( reg < NumOfAllRegs );
    return FloatRegNames[reg];
  }



};



class SparcFloatRegClass : public MachineRegClassInfo
{
 private:

  int findFloatColor(const IGNode *const Node, unsigned Start,
		     unsigned End, bool IsColorUsedArr[] ) const;

 public:

  SparcFloatRegClass(unsigned ID) 
    : MachineRegClassInfo(ID, 
			  SparcFloatRegOrder::NumOfAvailRegs,
			  SparcFloatRegOrder::NumOfAllRegs)
    {  }

  void colorIGNode(IGNode * Node, bool IsColorUsedArr[] ) const;

};




//-----------------------------------------------------------------------------
// Int CC Register Class
// Only one integer cc register is available
//-----------------------------------------------------------------------------


class SparcIntCCRegClass : public MachineRegClassInfo
{
public:

  SparcIntCCRegClass(unsigned ID) 
    : MachineRegClassInfo(ID,1, 1) {  }

  inline void colorIGNode(IGNode * Node, bool IsColorUsedArr[] ) const {
    Node->setColor(0);    // only one int cc reg is available
  }

};



//-----------------------------------------------------------------------------
// Float CC Register Class
// Only 4 Float CC registers are available
//-----------------------------------------------------------------------------


static string const FloatCCRegNames[] = 
  {    
    "fcc0",  "fcc1",  "fcc2",  "fcc3"
  };


class SparcFloatCCRegOrder{ 

 public:

  enum RegsInPrefOrder {

    fcc0, fcc1, fcc2, fcc3
  };

  static const string  getRegName(const unsigned reg) {
    assert( reg < 4 );
    return FloatCCRegNames[reg];
  }

};



class SparcFloatCCRegClass : public MachineRegClassInfo
{
public:

  SparcFloatCCRegClass(unsigned ID) 
    : MachineRegClassInfo(ID, 4, 4) {  }

  void colorIGNode(IGNode * Node, bool IsColorUsedArr[] ) const {
    int c;
    for(c=0; c < 4  && IsColorUsedArr[c] ; ++c) ; // find color
    assert( (c < 4)  && "Can allocate only 4 float cc registers");
    Node->setColor(c);   
  }

};




#endif
