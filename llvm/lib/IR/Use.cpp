//===-- Use.cpp - Implement the Use class ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include <new>

namespace llvm {

void Use::swap(Use &RHS) {
  Value *V1(Val);
  Value *V2(RHS.Val);
  if (V1 != V2) {
    if (V1) {
      removeFromList();
    }

    if (V2) {
      RHS.removeFromList();
      Val = V2;
      V2->addUse(*this);
    } else {
      Val = 0;
    }

    if (V1) {
      RHS.Val = V1;
      V1->addUse(RHS);
    } else {
      RHS.Val = 0;
    }
  }
}

User *Use::getUser() const {
  const Use *End = getImpliedUser();
  const UserRef *ref = reinterpret_cast<const UserRef*>(End);
  return ref->getInt()
    ? ref->getPointer()
    : reinterpret_cast<User*>(const_cast<Use*>(End));
}

// Sets up the waymarking algoritm's tags for a series of Uses. See the
// algorithm details here:
//
//   http://www.llvm.org/docs/ProgrammersManual.html#UserLayout
//
Use *Use::initTags(Use * const Start, Use *Stop) {
  ptrdiff_t Done = 0;
  while (Done < 20) {
    if (Start == Stop--)
      return Start;
    static const PrevPtrTag tags[20] = { fullStopTag, oneDigitTag, stopTag,
                                         oneDigitTag, oneDigitTag, stopTag,
                                         zeroDigitTag, oneDigitTag, oneDigitTag,
                                         stopTag, zeroDigitTag, oneDigitTag,
                                         zeroDigitTag, oneDigitTag, stopTag,
                                         oneDigitTag, oneDigitTag, oneDigitTag,
                                         oneDigitTag, stopTag
                                       };
    new(Stop) Use(tags[Done++]);
  }

  ptrdiff_t Count = Done;
  while (Start != Stop) {
    --Stop;
    if (!Count) {
      new(Stop) Use(stopTag);
      ++Done;
      Count = Done;
    } else {
      new(Stop) Use(PrevPtrTag(Count & 1));
      Count >>= 1;
      ++Done;
    }
  }

  return Start;
}

void Use::zap(Use *Start, const Use *Stop, bool del) {
  while (Start != Stop)
    (--Stop)->~Use();
  if (del)
    ::operator delete(Start);
}

const Use *Use::getImpliedUser() const {
  const Use *Current = this;

  while (true) {
    unsigned Tag = (Current++)->Prev.getInt();
    switch (Tag) {
      case zeroDigitTag:
      case oneDigitTag:
        continue;

      case stopTag: {
        ++Current;
        ptrdiff_t Offset = 1;
        while (true) {
          unsigned Tag = Current->Prev.getInt();
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

} // End llvm namespace
