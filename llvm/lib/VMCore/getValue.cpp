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
// this can later be removed:
#include <iostream>

namespace llvm {

class Use::UseWaymark {

  friend class Use;

enum { spareBits = 2, requiredSteps = sizeof(Value*) * 8 - spareBits };

/// repaintByCopying -- given a pattern and a
/// junk tagspace, copy the former's tags into
/// the latter
///
static inline void repaintByCopying(Use *Tagspace, Use *Junk) {
    for (int I = requiredSteps; I; --I) {
        Use *Next = stripTag<Use::tagMaskN>(Junk->Next);
        Junk->Next = transferTag<Use::tagMaskN>(Tagspace->Next, Next);
        Tagspace = stripTag<Use::tagMaskN>(Tagspace->Next);
        Junk = Next;
    }

    assert((extractTag<Use::NextPtrTag, Use::tagMaskN>(Junk->Next) == Use::stopTagN)
           && "Why repaint by copying if the next is not Stop?");
}


/// repaintByCalculating -- given a pattern and a
/// junk tagspace, compute tags into the latter
///
static inline void repaintByCalculating(unsigned long Tags, Use *Junk) {
    Tags >>= spareBits;

    for (int I = requiredSteps - 1; I >= 0; --I) {
        Use::NextPtrTag Tag(Tags & (1 << I) ? Use::oneDigitTagN : Use::zeroDigitTagN);
        Use *Next = stripTag<Use::tagMaskN>(Junk->Next);
        Junk->Next = transferTag<Use::tagMaskN>(reinterpret_cast<Use*>(Tag), Next);
        Junk = Next;
    }

    assert((extractTag<Use::NextPtrTag, Use::tagMaskN>(Junk->Next) == Use::fullStopTagN)
           && "Why repaint by calculating if the next is not FullStop?");
}

/// punchAwayDigits -- ensure that repainted area
/// begins with a stop
///
static inline void punchAwayDigits(Use **Uprev) {
  Uprev = stripTag<Use::tagMask>(Uprev);
  // if (PrevU)
  //   assert(&PrevU->Next == Uprev && "U->Prev differs from PrevU?");
  *Uprev = stripTag<Use::tagMaskN>(*Uprev);
}


/// gatherAndPotentiallyRepaint is invoked for either
///  - sweeping over (a small amount of) initial junk
///  - or sweeping over a great amount of junk which
///    provides enough space to reproduce the bit pattern
///    of the Value* at its end, in which case it gets
///    overpainted.
///  In any case this routine is invoked with U being
///  pointed at from a Use with a stop tag.
///
static inline Value *gatherAndPotentiallyRepaint(Use *U) {
  int Cushion = requiredSteps;

  Use *Next(U->Next);
  // __builtin_prefetch(Next);
  Use::NextPtrTag Tag(extractTag<Use::NextPtrTag, Use::tagMaskN>(Next));
  Next = stripTag<Use::tagMaskN>(Next);

  // try to pick up exactly requiredSteps digits
  // from immediately behind the (precondition) stop
  unsigned long Acc(0);
  while (1) {
      if (Cushion <= 0) {
          assert((Tag == Use::fullStopTagN || Tag == Use::stopTagN)
                 && "More digits?");
          return reinterpret_cast<Value*>(Acc << spareBits);
      }

      switch (Tag) {
          case Use::fullStopTagN:
              return reinterpret_cast<Value*>(Next);
          case Use::stopTagN: {
	      goto efficiency;
          }
          default:
              Acc = (Acc << 1) | (Tag & 1);
              Next = Next->Next;
              // __builtin_prefetch(Next);
              --Cushion;
              Tag = extractTag<Use::NextPtrTag, Use::tagMaskN>(Next);
              Next = stripTag<Use::tagMaskN>(Next);
              continue;
      }
      break;
  }

  while (Cushion > 0) {
    switch (Tag) {
    case Use::fullStopTagN:
        return reinterpret_cast<Value*>(Next);
    case Use::stopTagN: {
        efficiency:
        // try to pick up exactly requiredSteps digits
        int digits = requiredSteps;
        Acc = 0;

        while (1) {
            if (!digits)
                return reinterpret_cast<Value*>(Acc << spareBits);

            Next = Next->Next;
            // __builtin_prefetch(Next);
            --Cushion;
            Tag = extractTag<Use::NextPtrTag, Use::tagMaskN>(Next);
            Next = stripTag<Use::tagMaskN>(Next);
            switch (Tag) {
                case Use::fullStopTagN:
                    if (Cushion <= 0) {
                        punchAwayDigits(U->Prev);
                        repaintByCalculating(reinterpret_cast<unsigned long>(Next), U);
                    }
                    return reinterpret_cast<Value*>(Next);
                case Use::stopTagN: {
                    if (Cushion <= 0) {
                        U = stripTag<Use::tagMaskN>(U->Next);
                    }
                    goto efficiency;
                }
                default:
                    --digits;
                    Acc = (Acc << 1) | (Tag & 1);
                    if (Cushion <= 0) {
                        U = stripTag<Use::tagMaskN>(U->Next);
                    }
                    continue;
            }
            break;
        }
    }
    // fall through
    default:
        Next = Next->Next;
        // __builtin_prefetch(Next);
        --Cushion;
        Tag = extractTag<Use::NextPtrTag, Use::tagMaskN>(Next);
        Next = stripTag<Use::tagMaskN>(Next);
    } // switch
  } // while

  // Now we know that we have a nice cushion between U and Next. Do the same
  // thing as above, but don't decrement Cushion any more, instead push U
  // forward. After the value is found, repaint beginning at U.

  while (1) {
    switch (Tag) {
    case Use::fullStopTagN: {
        punchAwayDigits(U->Prev);
        repaintByCalculating(reinterpret_cast<unsigned long>(Next), U);
        return reinterpret_cast<Value*>(Next);
    }
    case Use::stopTagN: {
        // try to pick up exactly requiredSteps digits
        int digits = requiredSteps;
        Acc = 0;
        Use *Tagspace(Next);

        while (1) {
            if (!digits) {
                punchAwayDigits(U->Prev);
                repaintByCopying(Tagspace, U);
                return reinterpret_cast<Value*>(Acc << spareBits);
            }

            Next = Next->Next;
            // __builtin_prefetch(Next);
            U = stripTag<Use::tagMaskN>(U->Next);
            Tag = extractTag<Use::NextPtrTag, Use::tagMaskN>(Next);
            Next = stripTag<Use::tagMaskN>(Next);
            switch (Tag) {
                case Use::fullStopTagN: {
                    punchAwayDigits(U->Prev);
                    repaintByCalculating(reinterpret_cast<unsigned long>(Next), U);
                    return reinterpret_cast<Value*>(Next);
                }
                case Use::stopTagN: {
                    break;
                }
                default:
                    --digits;
                    Acc = (Acc << 1) | (Tag & 1);
                    continue;
            }
            break;
        }
    }
    // fall through
    default:
        Next = Next->Next;
        // __builtin_prefetch(Next);
        U = stripTag<Use::tagMaskN>(U->Next);
        Tag = extractTag<Use::NextPtrTag, Use::tagMaskN>(Next);
        Next = stripTag<Use::tagMaskN>(Next);
    } // switch
  } // while
}


/// skipPotentiallyGathering is invoked for either
///  - picking up exactly ToGo digits
///  - or finding a stop which marks the beginning
///    of a repaintable area
///
static inline Value *skipPotentiallyGathering(Use *U,
                                              unsigned long Acc,
                                              int ToGo) {
  while (1) {
    if (!ToGo)
      return reinterpret_cast<Value*>(Acc << spareBits);

    Use *Next(U->Next);
    // __builtin_prefetch(Next);
    Use::NextPtrTag Tag(extractTag<Use::NextPtrTag, Use::tagMaskN>(Next));
    Next = stripTag<Use::tagMaskN>(Next);
    switch (Tag) {
    case Use::fullStopTagN:
      return reinterpret_cast<Value*>(Next);
    case Use::stopTagN:
      return gatherAndPotentiallyRepaint(Next);
    default:
      Acc = (Acc << 1) | (Tag & 1);
      --ToGo;
      U = Next;
    }
  }
}

}; // class UseWaymark

Value *Use::getValue() const {
  // __builtin_prefetch(Next);
  NextPtrTag Tag(extractTag<NextPtrTag, tagMaskN>(Next));
  switch (Tag) {
  case fullStopTagN:
      return reinterpret_cast<Value*>(stripTag<tagMaskN>(Next));
  case stopTagN:
    return UseWaymark::gatherAndPotentiallyRepaint(Next);
  default:
    return UseWaymark::skipPotentiallyGathering(stripTag<tagMaskN>(Next),
                                    Tag & 1,
                                    Use::UseWaymark::requiredSteps - 1);
  }
}

static char TagChar(int Tag) {
  return "s10S"[Tag];
}

void Use::showWaymarks() const {
  const Use *me(this);
  if (NextPtrTag Tag = extractTag<Use::NextPtrTag, Use::tagMaskN>(me)) {
    me = stripTag<tagMaskN>(me);
    std::cerr << '(' << TagChar(Tag) << ')';
  }

  me = me->Next;
  NextPtrTag TagHere = extractTag<Use::NextPtrTag, Use::tagMaskN>(me);
  std::cerr << TagChar(TagHere);

  me = stripTag<tagMaskN>(me);
  if (TagHere == fullStopTagN) {
    std::cerr << " ---> " << me << std::endl;
    std::cerr << "1234567890123456789012345678901234567890123456789012345678901234567890" << std::endl;
    std::cerr << "         1         2         3         4         5         6         7" << std::endl;
  }
  else
    me->showWaymarks();
}

} // namespace llvm

#if 0

// ################################################################
// ############################# TESTS ############################
// ################################################################

using namespace llvm;

namespace T1
{
    Use u00;
}

namespace T2
{
    Use u00((Value*)0xCAFEBABCUL);
    Use u01(u00);
}

namespace T3
{
    template <unsigned NEST>
    struct UseChain;

    template <>
    struct UseChain<0> {
        Use first;
        UseChain(Value *V) : first(V) {}
        UseChain(Use &U) : first(U) {}
    };

    template <unsigned NEST>
    struct UseChain {
        Use first;
        UseChain<NEST - 1> rest;
        UseChain(Value *V)
            : first(rest.first)
            , rest(V) {}
        UseChain(Use &U)
            : first(rest.first)
            , rest(U) {}
    };

    UseChain<30> uc31((Value*)0xCAFEBABCUL);
    Use& u31(uc31.first);
    UseChain<31> uc32((Value*)0xCAFEBABCUL);
    Use& u32(uc32.first);
}

namespace T4
{
    template <unsigned NEST, unsigned long PAT>
    struct UseChain;

    template <unsigned long PAT>
    struct UseChain<0, PAT> {
        Use first;
        static const unsigned long Pat = PAT >> 2;
        UseChain(Value *V) : first(V) {}
        UseChain(Use &U) : first(U) {}
    };

    template <unsigned NEST, unsigned long PAT>
    struct UseChain {
        Use first;
        UseChain<NEST - 1, PAT> rest;
        static const unsigned long Digit = UseChain<NEST - 1, PAT>::Pat & 1;
        static const Use::NextPtrTag Tag = Digit ? Use::oneDigitTagN : Use::zeroDigitTagN;
        static const unsigned long Pat = UseChain<NEST - 1, PAT>::Pat >> 1;
        UseChain(Value *V = reinterpret_cast<Value*>(PAT))
            : first(*addTag(&rest.first, Tag))
            , rest(V) {}
        UseChain(Use &U)
            : first(*addTag(&rest.first, Tag))
            , rest(U) {}
    };

    UseChain<30, 0xCAFEBABCUL> uc31;
    Use& u31(uc31.first);
    Use us32(u31);

    UseChain<29, 0xCAFEBABCUL> uc30;
    Use& u30(uc30.first);
    Use us31(u30);

    T3::UseChain<3> s3a((Value*)0xCAFEBABCUL);
    UseChain<30, 0xCAFEBABCUL> uc31s3S(s3a.first);
    Use& m35(uc31s3S.first);

    T3::UseChain<3> s3b((Value*)0xCAFEBABCUL);
    UseChain<29, 0xCAFEBABCUL> uc30s3S(s3b.first);
    Use& m34(uc30s3S.first);
    
    T3::UseChain<3> s3c((Value*)0xCAFEBABCUL);
    UseChain<25, 0xCAFEBABCUL> uc25s3S(s3c.first);
    Use& m30(uc25s3S.first);

    Use ms36(m35);
    Use ms35(m34);
    Use ms31(m30);
    Use ms32(ms31);

    UseChain<24, 0xCAFEBABCUL> uc25;
    Use& u25(uc25.first);
    T3::UseChain<10> m11s24dS(u25);
    Use& m36(m11s24dS.first);


    T3::UseChain<10> s10S((Value*)0xCAFEBABCUL);
    UseChain<20, 0xCAFEBABCUL> d20ss10S(s10S.first);
    T3::UseChain<20> s20sd20ss10S(d20ss10S.first);
    Use& m53(s20sd20ss10S.first);
}


int main(){
    if (NULL != T1::u00.getValue())
        return 1;
    if ((Value*)0xCAFEBABCUL != T2::u00.getValue())
        return 2;
    if ((Value*)0xCAFEBABCUL != T2::u01.getValue())
        return 2;
    if ((Value*)0xCAFEBABCUL != T3::u31.getValue())
        return 3;
    if ((Value*)0xCAFEBABCUL != T3::u32.getValue())
        return 3;
    if ((Value*)0xCAFEBABCUL != T3::u32.getValue()) // check the mutated value
        return 3;
    if ((Value*)0xCAFEBABCUL != T4::u31.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::u30.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::us32.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::us31.getValue())
        return 4;

    // mixed tests
    if ((Value*)0xCAFEBABCUL != T4::m35.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::m34.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::m30.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::ms36.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::ms35.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::ms35.getValue()) // check the mutated value
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::ms32.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::ms32.getValue())
        return 4;
    if ((Value*)0xCAFEBABCUL != T4::ms32.getValue()) // check the mutated value
        return 4;

    T4::m36.showWaymarks();
    if ((Value*)0xCAFEBABCUL != T4::m36.getValue())
        return 4;
    T4::m36.showWaymarks();
    if ((Value*)0xCAFEBABCUL != T4::m36.getValue()) // check the mutated value
        return 4;
    T4::m36.showWaymarks();

    T4::m53.showWaymarks();
    if ((Value*)0xCAFEBABCUL != T4::m53.getValue())
        return 4;
    T4::m53.showWaymarks();
    if ((Value*)0xCAFEBABCUL != T4::m53.getValue()) // check the mutated value
        return 4;
    T4::m53.showWaymarks();
}

#endif
