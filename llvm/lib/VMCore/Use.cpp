//===-- Use.cpp - Implement the Use class ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the algorithm for finding the User of a Use.
//
//===----------------------------------------------------------------------===//

#include "llvm/User.h"

namespace llvm {

//===----------------------------------------------------------------------===//
//                         Use swap Implementation
//===----------------------------------------------------------------------===//

void Use::swap(Use &RHS) {
  ptrdiff_t dist((char*)&RHS - (char*)this);

  if (dist) {
    Use *valid1(stripTag<tagMaskN>(Next));
    Use *valid2(stripTag<tagMaskN>(RHS.Next));
    if (valid1 && valid2) {
      bool real1(fullStopTagN != extractTag<NextPtrTag, tagMaskN>(Next));
      bool real2(fullStopTagN != extractTag<NextPtrTag, tagMaskN>(RHS.Next));
      (char*&)*stripTag<tagMask>(Prev) += dist;
      (char*&)*stripTag<tagMask>(RHS.Prev) -= dist;
      if (real1)
	(char*&)valid1->Next += dist;
      if (real2)
        (char*&)valid2->Next -= dist;

    }

    // swap the members
    std::swap(Next, RHS.Next);
    Use** Prev1 = transferTag<tagMask>(Prev, stripTag<tagMask>(RHS.Prev));
    RHS.Prev = transferTag<tagMask>(RHS.Prev, stripTag<tagMask>(Prev));
    Prev = Prev1;
  }
  /*  Value *V1(Val1);
  Value *V2(RHS.Val1);
  if (V1 != V2) {
    if (V1) {
      removeFromList();
    }

    if (V2) {
      RHS.removeFromList();
      Val1 = V2;
      V2->addUse(*this);
    } else {
      Val1 = 0;
    }

    if (V1) {
      RHS.Val1 = V1;
      V1->addUse(RHS);
    } else {
      RHS.Val1 = 0;
    }
  }
  */
}

//===----------------------------------------------------------------------===//
//                         Use getImpliedUser Implementation
//===----------------------------------------------------------------------===//

const Use *Use::getImpliedUser() const {
  const Use *Current = this;

  while (true) {
    unsigned Tag = extractTag<PrevPtrTag, tagMask>((Current++)->Prev);
    switch (Tag) {
      case zeroDigitTag:
      case oneDigitTag:
        continue;

      case stopTag: {
        ++Current;
        ptrdiff_t Offset = 1;
        while (true) {
          unsigned Tag = extractTag<PrevPtrTag, tagMask>(Current->Prev);
          switch (Tag) {
            case zeroDigitTag:
            case oneDigitTag:
              ++Current;
              Offset = (Offset << 1) + Tag;
              continue;
            default:
              return Current + Offset;
          }
        }
      }

      case fullStopTag:
        return Current;
    }
  }
}

//===----------------------------------------------------------------------===//
//                         Use initTags Implementation
//===----------------------------------------------------------------------===//

Use *Use::initTags(Use * const Start, Use *Stop, ptrdiff_t Done) {
  ptrdiff_t Count = Done;
  while (Start != Stop) {
    --Stop;
    Stop->Val1 = 0;
    Stop->Next = nilUse(0);
    if (!Count) {
      Stop->Prev = reinterpret_cast<Use**>(Done == 0 ? fullStopTag : stopTag);
      ++Done;
      Count = Done;
    } else {
      Stop->Prev = reinterpret_cast<Use**>(Count & 1);
      Count >>= 1;
      ++Done;
    }
  }

  return Start;
}

//===----------------------------------------------------------------------===//
//                         Use zap Implementation
//===----------------------------------------------------------------------===//

void Use::zap(Use *Start, const Use *Stop, bool del) {
  if (del) {
    while (Start != Stop) {
      (--Stop)->~Use();
    }
    ::operator delete(Start);
    return;
  }

  while (Start != Stop) {
    (Start++)->set(0);
  }
}

//===----------------------------------------------------------------------===//
//                         AugmentedUse layout struct
//===----------------------------------------------------------------------===//

struct AugmentedUse : Use {
  User *ref;
  AugmentedUse(); // not implemented
};


//===----------------------------------------------------------------------===//
//                         Use getUser Implementation
//===----------------------------------------------------------------------===//

User *Use::getUser() const {
  const Use *End = getImpliedUser();
  User *She = static_cast<const AugmentedUse*>(End - 1)->ref;
  She = extractTag<Tag, tagOne>(She)
      ? llvm::stripTag<tagOne>(She)
      : reinterpret_cast<User*>(const_cast<Use*>(End));

  return She;
}

//===----------------------------------------------------------------------===//
//                         User allocHungoffUses Implementation
//===----------------------------------------------------------------------===//

Use *User::allocHungoffUses(unsigned N) const {
  Use *Begin = static_cast<Use*>(::operator new(sizeof(Use) * N
                                                + sizeof(AugmentedUse)
                                                - sizeof(Use)));
  Use *End = Begin + N;
  static_cast<AugmentedUse&>(End[-1]).ref = addTag(this, tagOne);
  return Use::initTags(Begin, End);
}

} // End llvm namespace
