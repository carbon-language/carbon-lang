/* Title:   LiveRange.h
   Author:  Ruchira Sasanka
   Date:    July 25, 01
   Purpose: To keep info about a live range. 
   Asuumptions:

   Since the Value pointed by a use is the same as of its def, it is sufficient
   to keep only defs in a LiveRange.
*/

#ifndef LIVE_RANGE_H
#define LIVE_RANGE_H

#include "llvm/Analysis/LiveVar/ValueSet.h"
#include "llvm/Type.h"




class RegClass;
class IGNode;


class LiveRange : public ValueSet
{
 private:

  RegClass *MyRegClass;       // register classs (e.g., int, FP) for this LR

  // a list of call instructions that interferes with this live range
  vector<const Instruction *> CallInterferenceList;  

  IGNode *UserIGNode;         // IGNode which uses this LR
  int Color;                  // color assigned to this live range
  bool mustSpill;             // whether this LR must be spilt

  // whether this LR must be saved accross calls
  bool mustSaveAcrossCalls;        

  // bool mustLoadFromStack;     // must load from stack at start of method

 public:


  ~LiveRange() {}             // empty destructor 

  void setRegClass(RegClass *const RC) 
    { MyRegClass = RC; }

  inline RegClass *const getRegClass() const 
    { assert(MyRegClass); return MyRegClass; } 

  inline bool hasColor() const 
    { return Color != -1; }

  inline unsigned int getColor() const 
    { assert( Color != -1); return (unsigned) Color ; }

  inline void setColor(unsigned int Col) 
    { Color = (int) Col ; }

  
  inline void addCallInterference(const Instruction *const  Inst) 
    { CallInterferenceList.push_back( Inst ); }

  inline const Instruction *const getCallInterference(const unsigned i) const {
    assert( i < CallInterferenceList.size() ); 
    return CallInterferenceList[i];  
  }

  inline unsigned int getNumOfCallInterferences() const 
    { return CallInterferenceList.size(); } 

  
  inline void markForSpill() { mustSpill = true; }

  inline void markForSaveAcrossCalls() { mustSaveAcrossCalls = true; }

  // inline void markForLoadFromStack() { mustLoadFromStack = true; 


  inline void setUserIGNode( IGNode *const IGN) 
    { assert( !UserIGNode); UserIGNode = IGN; }

  inline IGNode * getUserIGNode() const 
    { return UserIGNode; }    // NULL if the user is not allocated

  inline Type::PrimitiveID getTypeID() const {
    const Value *val = *begin();
    assert(val && "Can't find type - Live range is empty" );
    return (val->getType())->getPrimitiveID();
  }


  inline LiveRange() : ValueSet() , CallInterferenceList() 
    {
      Color = -1;                 // not yet colored 
      mustSpill = mustSaveAcrossCalls = false;
      MyRegClass = NULL;
      UserIGNode = NULL;
    }

};





#endif

