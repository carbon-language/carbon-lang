/* Title:   MachineRegInfo.h
   Author:  Ruchira Sasanka
   Date:    Aug 20, 01
   Purpose: Contains the description of machine register classes.

   Notes:

   A machine will have several register classes. For each register class, the
   machine has to provide a class which is derived class of the virtual class
   MachineRegClass. This virtual class itself is machine independent but 
   the derived classes are all machine dependent.
*/


#ifndef MACHINE_REG_INFO_H
#define MACHINE_REG_INFO_H

#include "llvm/CodeGen/MachineInstr.h"



// This is the virtual class which must be subclassed by all machine specific
// register classes.

  unsigned RegClassID;        // integer ID of a reg class
  unsigned NumOfAvailRegs;    // # of avail for coloring (without SP, g0 etc)
  unsigned NumOfAllRegs;      // # of all registers (including SP, g0 etc


class MachineRegClass 
{

 private:
  const unsigned RegClassID;

 public:

  virtual unsigned getRegClassID() const = 0;

  // Number of registes available for coloring (e.g., without SP, g0 etc)
  virtual unsigned getNumOfAvailRegs() const = 0;

  // Number of all registers (e.g., including SP, g0 etc)
  virtual unsigned getNumOfAllRegs() const = 0;

  // This method should find a color which is not used by neighbors
  // (i.e., a false position in IsColorUsedArr) and 
  virtual void colorIGNode(IGNode * Node, bool IsColorUsedArr[] ) const = 0;


  MachineRegClass(const unsigned ID) : RegClassID(ID) { }


};


// include .h files that describes machine reg classes here

#include "RegAlloc/Sparc/SparcIntRegClass.h"

typedef vector<const MachineRegClass *> MachineRegClassArrayType;



class MachineRegInfo
{
 private:

  // A vector of all machine register classes
  MachineRegClassArrayType MachineRegClassArr;


 public:

  MachineRegInfo() : MachineRegClassArr() { 

    MachineRegClassArr.push_back( new SparcIntRegClass(0) );
    // RegClassArr.pushback( new SparcFloatRegClass(1) );
    // RegClassArr.pushback( new SparcFloatCCRegClass(2) );

    if(DEBUG_RA)
      cerr << "Created machine register classes." << endl;

  }


  inline unsigned int getNumOfRegClasses() const { 
    return MachineRegClassArr.size(); 
  }  

  unsigned getRegClassIDOfValue (const Value *const Val) const ;

  const MachineRegClass *const getMachineRegClass(unsigned i) const { 
    return MachineRegClassArr[i]; 
  }

  inline bool isCallInst(const MachineInstr *const MI) const {
    MachineOpCode Op = MI->getOpCode();
    return false; // ########################################
    // return (Op == CALL || Op == JMPL);
  }

};


// This function should detrmine the register class of a value. This can be 
// done on type information in the value class. The register class returned 
// must be same as  the array index of RegClassArr.

unsigned MachineRegInfo::getRegClassIDOfValue (const Value *const Val) const
{
  
  return 0;
  
}




#endif

