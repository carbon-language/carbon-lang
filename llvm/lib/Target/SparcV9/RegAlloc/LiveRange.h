//===-- LiveRange.h - Store info about a live range -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements a live range using a SetVector of Value *s.  We keep only
// defs in a V9LiveRange.
//
//===----------------------------------------------------------------------===//

#ifndef LIVERANGE_H
#define LIVERANGE_H

#include "llvm/Value.h"
#include "llvm/ADT/SetVector.h"
#include <iostream>

namespace llvm {

class RegClass;
class IGNode;

class V9LiveRange {
public:
  typedef SetVector<const Value *> ValueContainerType;
  typedef ValueContainerType::iterator iterator;
  typedef ValueContainerType::const_iterator const_iterator;

private:
  ValueContainerType MyValues; // Values in this V9LiveRange
  RegClass *MyRegClass;        // register class (e.g., int, FP) for this LR

  /// doesSpanAcrossCalls - Does this live range span across calls?
  /// This information is used by graph coloring algo to avoid allocating
  /// volatile colors to live ranges that span across calls (since they have to
  /// be saved/restored)
  ///
  bool doesSpanAcrossCalls;

  IGNode *UserIGNode;         // IGNode which uses this LR
  int Color;                  // color assigned to this live range
  bool mustSpill;             // whether this LR must be spilt

  /// SuggestedColor - if this LR has a suggested color, can it
  /// really be allocated?  A suggested color cannot be allocated when the
  /// suggested color is volatile and when there are call
  /// interferences.
  ///
  int SuggestedColor;        // The suggested color for this LR

  /// CanUseSuggestedCol - It is possible that a suggested color for
  /// this live range is not available before graph coloring (e.g., it
  /// can be allocated to another live range which interferes with
  /// this)
  ///
  bool CanUseSuggestedCol;

  /// SpilledStackOffsetFromFP - If this LR is spilled, its stack
  /// offset from *FP*. The spilled offsets must always be relative to
  /// the FP.
  ///
  int SpilledStackOffsetFromFP;

  /// HasSpillOffset - True iff this live range has a spill offset.
  ///
  bool HasSpillOffset;

  /// SpillCost - The spill cost of this live range. Calculated using loop depth
  /// of each reference to each Value in the live range.
  ///
  unsigned SpillCost;

public:
  iterator        begin()       { return MyValues.begin();    }
  const_iterator  begin() const { return MyValues.begin();    }
  iterator          end()       { return MyValues.end();      }
  const_iterator    end() const { return MyValues.end();      }
  bool insert(const Value *&X)  { return MyValues.insert (X); }
  void insert(iterator b, iterator e) { MyValues.insert (b, e); }

  V9LiveRange() {
    Color = SuggestedColor = -1;        // not yet colored
    mustSpill = false;
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

  inline bool isMarkedForSpill() const { return mustSpill; }

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

static inline std::ostream &operator << (std::ostream &os,
                                         const V9LiveRange &lr) {
  os << "LiveRange@" << (void *)(&lr);
  return os;
};

} // End llvm namespace

#endif
