//===-- LiveRange.h - Store info about a live range --------------*- C++ -*--=//
//
// Implements a live range using a ValueSet. A LiveRange is a simple set
// of Values. 
//
// Since the Value pointed by a use is the same as of its def, it is sufficient
// to keep only defs in a LiveRange.
//
//===----------------------------------------------------------------------===//

#ifndef LIVE_RANGE_H
#define LIVE_RANGE_H

#include "llvm/CodeGen/ValueSet.h"
#include "llvm/Value.h"

class RegClass;
class IGNode;
class Type;

class LiveRange : public ValueSet {
  RegClass *MyRegClass;       // register classs (e.g., int, FP) for this LR

  // doesSpanAcrossCalls - Does this live range span across calls? 
  // This information is used by graph
  // coloring algo to avoid allocating volatile colors to live ranges
  // that span across calls (since they have to be saved/restored)
  //
  bool doesSpanAcrossCalls;

  IGNode *UserIGNode;         // IGNode which uses this LR
  int Color;                  // color assigned to this live range
  bool mustSpill;             // whether this LR must be spilt

  // mustSaveAcrossCalls - whether this LR must be saved accross calls
  // ***TODO REMOVE this
  //
  bool mustSaveAcrossCalls;        
  
  // SuggestedColor - if this LR has a suggested color, can it be
  // really alloated?  A suggested color cannot be allocated when the
  // suggested color is volatile and when there are call
  // interferences.
  //
  int SuggestedColor;        // The suggested color for this LR

  // CanUseSuggestedCol - It is possible that a suggested color for
  // this live range is not available before graph coloring (e.g., it
  // can be allocated to another live range which interferes with
  // this)
  // 
  bool CanUseSuggestedCol;

  // SpilledStackOffsetFromFP - If this LR is spilled, its stack
  // offset from *FP*. The spilled offsets must always be relative to
  // the FP.
  //
  int SpilledStackOffsetFromFP;

  // HasSpillOffset 0 Whether this live range has a spill offset
  //
  bool HasSpillOffset;

  // The spill cost of this live range. Calculated using loop depth of
  // each reference to each Value in the live range
  //
  unsigned SpillCost;

public:
  LiveRange() {
    Color = SuggestedColor = -1;        // not yet colored 
    mustSpill = mustSaveAcrossCalls = false;
    MyRegClass = 0;
    UserIGNode = 0;
    doesSpanAcrossCalls = false;
    CanUseSuggestedCol = true;
    HasSpillOffset = false;
    SpillCost = 0;
  }

  void setRegClass(RegClass *RC) { MyRegClass = RC; }

  RegClass *getRegClass() const { assert(MyRegClass); return MyRegClass; }
  unsigned getRegClassID() const;

  bool hasColor() const { return Color != -1; }
  
  unsigned getColor() const { assert(Color != -1); return (unsigned)Color; }

  void setColor(unsigned Col) { Color = (int)Col; }

  inline void setCallInterference() { 
    doesSpanAcrossCalls = 1;
  }
  inline void clearCallInterference() { 
    doesSpanAcrossCalls = 0;
  }

  inline bool isCallInterference() const { 
    return doesSpanAcrossCalls == 1; 
  } 

  inline void markForSpill() { mustSpill = true; }

  inline bool isMarkedForSpill() { return mustSpill; }

  inline void setSpillOffFromFP(int StackOffset) {
    assert(mustSpill && "This LR is not spilled");
    SpilledStackOffsetFromFP = StackOffset;
    HasSpillOffset = true;
  }

  inline void modifySpillOffFromFP(int StackOffset) {
    assert(mustSpill && "This LR is not spilled");
    SpilledStackOffsetFromFP = StackOffset;
    HasSpillOffset = true;
  }

  inline bool hasSpillOffset() const {
    return HasSpillOffset;
  }

  inline int getSpillOffFromFP() const {
    assert(HasSpillOffset && "This LR is not spilled");
    return SpilledStackOffsetFromFP;
  }

  inline void markForSaveAcrossCalls() { mustSaveAcrossCalls = true; }
  
  inline void setUserIGNode(IGNode *IGN) {
    assert(!UserIGNode); UserIGNode = IGN;
  }

  // getUserIGNode - NULL if the user is not allocated
  inline IGNode *getUserIGNode() const { return UserIGNode; }

  inline const Type *getType() const {
    return (*begin())->getType();  // set's don't have a front
  }
  
  inline void setSuggestedColor(int Col) {
    if (SuggestedColor == -1)
      SuggestedColor = Col;
  }

  inline unsigned getSuggestedColor() const {
    assert(SuggestedColor != -1);      // only a valid color is obtained
    return (unsigned)SuggestedColor;
  }

  inline bool hasSuggestedColor() const {
    return SuggestedColor != -1;
  }

  inline bool isSuggestedColorUsable() const {
    assert(hasSuggestedColor() && "No suggested color");
    return CanUseSuggestedCol;
  }

  inline void setSuggestedColorUsable(bool val) {
    assert(hasSuggestedColor() && "No suggested color");
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
