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


//----------------------------------------------------------------------------
// Class LiveRange
//
// Implements a live range using a ValueSet. A LiveRange is a simple set
// of Values. 
//----------------------------------------------------------------------------

class LiveRange : public ValueSet
{
 private:

  RegClass *MyRegClass;       // register classs (e.g., int, FP) for this LR


  bool doesSpanAcrossCalls;
  //
  // Does this live range span across calls? 
  // This information is used by graph
  // coloring algo to avoid allocating volatile colors to live ranges
  // that span across calls (since they have to be saved/restored)
  

  IGNode *UserIGNode;         // IGNode which uses this LR

  int Color;                  // color assigned to this live range

  bool mustSpill;             // whether this LR must be spilt


  bool mustSaveAcrossCalls;        
  //
  // whether this LR must be saved accross calls ***TODO REMOVE this
  
  int SuggestedColor;        // The suggested color for this LR
  //
  // if this LR has a suggested color, can it be really alloated?
  // A suggested color cannot be allocated when the suggested color is
  // volatile and when there are call interferences.

  bool CanUseSuggestedCol;
  // 
  // It is possible that a suggested color for this live range is not
  // available before graph coloring (e.g., it can be allocated to another
  // live range which interferes with this)

  int SpilledStackOffsetFromFP;
  //
  // if this LR is spilled, its stack offset from *FP*. The spilled offsets
  // must always be relative to the FP.

  bool HasSpillOffset;
  //
  // Whether this live range has a spill offset

  unsigned SpillCost;
  //
  // The spill cost of this live range. Calculated using loop depth of
  // each reference to each Value in the live range

 public:

  // constructor
  //
  LiveRange() : ValueSet() {
    Color = SuggestedColor = -1;        // not yet colored 
    mustSpill = mustSaveAcrossCalls = false;
    MyRegClass = NULL;
    UserIGNode = NULL;
    doesSpanAcrossCalls = false;
    CanUseSuggestedCol = true;
    HasSpillOffset  = false;
    SpillCost = 0;
  }

  // empty destructor since there are nothing to be deleted
  //
  ~LiveRange() {}          


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

  
  inline void setCallInterference() { 
    doesSpanAcrossCalls = 1;
  }


  inline bool isCallInterference() const { 
    return (doesSpanAcrossCalls == 1); 
  } 

  inline void markForSpill() { mustSpill = true; }

  inline bool isMarkedForSpill() { return  mustSpill; }

  inline void setSpillOffFromFP(int StackOffset) {
    assert( mustSpill && "This LR is not spilled");
    SpilledStackOffsetFromFP = StackOffset;
    HasSpillOffset = true;
  }

  inline void modifySpillOffFromFP(int StackOffset) {
    assert( mustSpill && "This LR is not spilled");
    SpilledStackOffsetFromFP = StackOffset;
    HasSpillOffset = true;
  }



  inline bool hasSpillOffset() const {
    return  HasSpillOffset;
  }


  inline int getSpillOffFromFP() const {
    assert( HasSpillOffset && "This LR is not spilled");
    return SpilledStackOffsetFromFP;
  }


  inline void markForSaveAcrossCalls() { mustSaveAcrossCalls = true; }

  
  inline void setUserIGNode( IGNode *const IGN) 
    { assert( !UserIGNode); UserIGNode = IGN; }

  inline IGNode * getUserIGNode() const 
    { return UserIGNode; }    // NULL if the user is not allocated

  inline const Type* getType() const {
    const Value *val = *begin();
    assert(val && "Can't find type - Live range is empty" );
    return val->getType();
  }
  
  inline Type::PrimitiveID getTypeID() const {
    return this->getType()->getPrimitiveID();
  }

  inline void setSuggestedColor(int Col) {
    //assert( (SuggestedColor == -1) && "Changing an already suggested color");

    if(SuggestedColor == -1 )
      SuggestedColor = Col;
    else if (DEBUG_RA) 
      cerr << "Already has a suggested color " << Col << endl;
  }

  inline unsigned getSuggestedColor() const {
    assert( SuggestedColor != -1);      // only a valid color is obtained
    return (unsigned) SuggestedColor;
  }

  inline bool hasSuggestedColor() const {
    return ( SuggestedColor > -1);
  }

  inline bool isSuggestedColorUsable() const {
    assert( hasSuggestedColor() && "No suggested color");
    return CanUseSuggestedCol;
  }

  inline void setSuggestedColorUsable(const bool val) {
    assert( hasSuggestedColor() && "No suggested color");
    CanUseSuggestedCol = val;
  }

  inline void addSpillCost(unsigned cost) {
    SpillCost += cost;
  }

  inline unsigned getSpillCost() const {
    return SpillCost;
  }

};





#endif

