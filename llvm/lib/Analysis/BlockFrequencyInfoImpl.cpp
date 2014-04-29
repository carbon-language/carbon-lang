//===- BlockFrequencyImplInfo.cpp - Block Frequency Info Implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>

using namespace llvm;
using namespace llvm::bfi_detail;

#define DEBUG_TYPE "block-freq"

//===----------------------------------------------------------------------===//
//
// UnsignedFloat implementation.
//
//===----------------------------------------------------------------------===//
#ifndef _MSC_VER
const int32_t UnsignedFloatBase::MaxExponent;
const int32_t UnsignedFloatBase::MinExponent;
#endif

static void appendDigit(std::string &Str, unsigned D) {
  assert(D < 10);
  Str += '0' + D % 10;
}

static void appendNumber(std::string &Str, uint64_t N) {
  while (N) {
    appendDigit(Str, N % 10);
    N /= 10;
  }
}

static bool doesRoundUp(char Digit) {
  switch (Digit) {
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    return true;
  default:
    return false;
  }
}

static std::string toStringAPFloat(uint64_t D, int E, unsigned Precision) {
  assert(E >= UnsignedFloatBase::MinExponent);
  assert(E <= UnsignedFloatBase::MaxExponent);

  // Find a new E, but don't let it increase past MaxExponent.
  int LeadingZeros = UnsignedFloatBase::countLeadingZeros64(D);
  int NewE = std::min(UnsignedFloatBase::MaxExponent, E + 63 - LeadingZeros);
  int Shift = 63 - (NewE - E);
  assert(Shift <= LeadingZeros);
  assert(Shift == LeadingZeros || NewE == UnsignedFloatBase::MaxExponent);
  D <<= Shift;
  E = NewE;

  // Check for a denormal.
  unsigned AdjustedE = E + 16383;
  if (!(D >> 63)) {
    assert(E == UnsignedFloatBase::MaxExponent);
    AdjustedE = 0;
  }

  // Build the float and print it.
  uint64_t RawBits[2] = {D, AdjustedE};
  APFloat Float(APFloat::x87DoubleExtended, APInt(80, RawBits));
  SmallVector<char, 24> Chars;
  Float.toString(Chars, Precision, 0);
  return std::string(Chars.begin(), Chars.end());
}

static std::string stripTrailingZeros(const std::string &Float) {
  size_t NonZero = Float.find_last_not_of('0');
  assert(NonZero != std::string::npos && "no . in floating point string");

  if (Float[NonZero] == '.')
    ++NonZero;

  return Float.substr(0, NonZero + 1);
}

std::string UnsignedFloatBase::toString(uint64_t D, int16_t E, int Width,
                                        unsigned Precision) {
  if (!D)
    return "0.0";

  // Canonicalize exponent and digits.
  uint64_t Above0 = 0;
  uint64_t Below0 = 0;
  uint64_t Extra = 0;
  int ExtraShift = 0;
  if (E == 0) {
    Above0 = D;
  } else if (E > 0) {
    if (int Shift = std::min(int16_t(countLeadingZeros64(D)), E)) {
      D <<= Shift;
      E -= Shift;

      if (!E)
        Above0 = D;
    }
  } else if (E > -64) {
    Above0 = D >> -E;
    Below0 = D << (64 + E);
  } else if (E > -120) {
    Below0 = D >> (-E - 64);
    Extra = D << (128 + E);
    ExtraShift = -64 - E;
  }

  // Fall back on APFloat for very small and very large numbers.
  if (!Above0 && !Below0)
    return toStringAPFloat(D, E, Precision);

  // Append the digits before the decimal.
  std::string Str;
  size_t DigitsOut = 0;
  if (Above0) {
    appendNumber(Str, Above0);
    DigitsOut = Str.size();
  } else
    appendDigit(Str, 0);
  std::reverse(Str.begin(), Str.end());

  // Return early if there's nothing after the decimal.
  if (!Below0)
    return Str + ".0";

  // Append the decimal and beyond.
  Str += '.';
  uint64_t Error = UINT64_C(1) << (64 - Width);

  // We need to shift Below0 to the right to make space for calculating
  // digits.  Save the precision we're losing in Extra.
  Extra = (Below0 & 0xf) << 56 | (Extra >> 8);
  Below0 >>= 4;
  size_t SinceDot = 0;
  size_t AfterDot = Str.size();
  do {
    if (ExtraShift) {
      --ExtraShift;
      Error *= 5;
    } else
      Error *= 10;

    Below0 *= 10;
    Extra *= 10;
    Below0 += (Extra >> 60);
    Extra = Extra & (UINT64_MAX >> 4);
    appendDigit(Str, Below0 >> 60);
    Below0 = Below0 & (UINT64_MAX >> 4);
    if (DigitsOut || Str.back() != '0')
      ++DigitsOut;
    ++SinceDot;
  } while (Error && (Below0 << 4 | Extra >> 60) >= Error / 2 &&
           (!Precision || DigitsOut <= Precision || SinceDot < 2));

  // Return early for maximum precision.
  if (!Precision || DigitsOut <= Precision)
    return stripTrailingZeros(Str);

  // Find where to truncate.
  size_t Truncate =
      std::max(Str.size() - (DigitsOut - Precision), AfterDot + 1);

  // Check if there's anything to truncate.
  if (Truncate >= Str.size())
    return stripTrailingZeros(Str);

  bool Carry = doesRoundUp(Str[Truncate]);
  if (!Carry)
    return stripTrailingZeros(Str.substr(0, Truncate));

  // Round with the first truncated digit.
  for (std::string::reverse_iterator I(Str.begin() + Truncate), E = Str.rend();
       I != E; ++I) {
    if (*I == '.')
      continue;
    if (*I == '9') {
      *I = '0';
      continue;
    }

    ++*I;
    Carry = false;
    break;
  }

  // Add "1" in front if we still need to carry.
  return stripTrailingZeros(std::string(Carry, '1') + Str.substr(0, Truncate));
}

raw_ostream &UnsignedFloatBase::print(raw_ostream &OS, uint64_t D, int16_t E,
                                      int Width, unsigned Precision) {
  return OS << toString(D, E, Width, Precision);
}

void UnsignedFloatBase::dump(uint64_t D, int16_t E, int Width) {
  print(dbgs(), D, E, Width, 0) << "[" << Width << ":" << D << "*2^" << E
                                << "]";
}

static std::pair<uint64_t, int16_t>
getRoundedFloat(uint64_t N, bool ShouldRound, int64_t Shift) {
  if (ShouldRound)
    if (!++N)
      // Rounding caused an overflow.
      return std::make_pair(UINT64_C(1), Shift + 64);
  return std::make_pair(N, Shift);
}

std::pair<uint64_t, int16_t> UnsignedFloatBase::divide64(uint64_t Dividend,
                                                         uint64_t Divisor) {
  // Input should be sanitized.
  assert(Divisor);
  assert(Dividend);

  // Minimize size of divisor.
  int16_t Shift = 0;
  if (int Zeros = countTrailingZeros(Divisor)) {
    Shift -= Zeros;
    Divisor >>= Zeros;
  }

  // Check for powers of two.
  if (Divisor == 1)
    return std::make_pair(Dividend, Shift);

  // Maximize size of dividend.
  if (int Zeros = countLeadingZeros64(Dividend)) {
    Shift -= Zeros;
    Dividend <<= Zeros;
  }

  // Start with the result of a divide.
  uint64_t Quotient = Dividend / Divisor;
  Dividend %= Divisor;

  // Continue building the quotient with long division.
  //
  // TODO: continue with largers digits.
  while (!(Quotient >> 63) && Dividend) {
    // Shift Dividend, and check for overflow.
    bool IsOverflow = Dividend >> 63;
    Dividend <<= 1;
    --Shift;

    // Divide.
    bool DoesDivide = IsOverflow || Divisor <= Dividend;
    Quotient = (Quotient << 1) | uint64_t(DoesDivide);
    Dividend -= DoesDivide ? Divisor : 0;
  }

  // Round.
  if (Dividend >= getHalf(Divisor))
    if (!++Quotient)
      // Rounding caused an overflow in Quotient.
      return std::make_pair(UINT64_C(1), Shift + 64);

  return getRoundedFloat(Quotient, Dividend >= getHalf(Divisor), Shift);
}

std::pair<uint64_t, int16_t> UnsignedFloatBase::multiply64(uint64_t L,
                                                           uint64_t R) {
  // Separate into two 32-bit digits (U.L).
  uint64_t UL = L >> 32, LL = L & UINT32_MAX, UR = R >> 32, LR = R & UINT32_MAX;

  // Compute cross products.
  uint64_t P1 = UL * UR, P2 = UL * LR, P3 = LL * UR, P4 = LL * LR;

  // Sum into two 64-bit digits.
  uint64_t Upper = P1, Lower = P4;
  auto addWithCarry = [&](uint64_t N) {
    uint64_t NewLower = Lower + (N << 32);
    Upper += (N >> 32) + (NewLower < Lower);
    Lower = NewLower;
  };
  addWithCarry(P2);
  addWithCarry(P3);

  // Check whether the upper digit is empty.
  if (!Upper)
    return std::make_pair(Lower, 0);

  // Shift as little as possible to maximize precision.
  unsigned LeadingZeros = countLeadingZeros64(Upper);
  int16_t Shift = 64 - LeadingZeros;
  if (LeadingZeros)
    Upper = Upper << LeadingZeros | Lower >> Shift;
  bool ShouldRound = Shift && (Lower & UINT64_C(1) << (Shift - 1));
  return getRoundedFloat(Upper, ShouldRound, Shift);
}

//===----------------------------------------------------------------------===//
//
// BlockMass implementation.
//
//===----------------------------------------------------------------------===//
UnsignedFloat<uint64_t> BlockMass::toFloat() const {
  if (isFull())
    return UnsignedFloat<uint64_t>(1, 0);
  return UnsignedFloat<uint64_t>(getMass() + 1, -64);
}

void BlockMass::dump() const { print(dbgs()); }

static char getHexDigit(int N) {
  assert(N < 16);
  if (N < 10)
    return '0' + N;
  return 'a' + N - 10;
}
raw_ostream &BlockMass::print(raw_ostream &OS) const {
  for (int Digits = 0; Digits < 16; ++Digits)
    OS << getHexDigit(Mass >> (60 - Digits * 4) & 0xf);
  return OS;
}

//===----------------------------------------------------------------------===//
//
// BlockFrequencyInfoImpl implementation.
//
//===----------------------------------------------------------------------===//
namespace {

typedef BlockFrequencyInfoImplBase::BlockNode BlockNode;
typedef BlockFrequencyInfoImplBase::Distribution Distribution;
typedef BlockFrequencyInfoImplBase::Distribution::WeightList WeightList;
typedef BlockFrequencyInfoImplBase::Float Float;
typedef BlockFrequencyInfoImplBase::LoopData LoopData;
typedef BlockFrequencyInfoImplBase::Weight Weight;
typedef BlockFrequencyInfoImplBase::FrequencyData FrequencyData;

/// \brief Dithering mass distributer.
///
/// This class splits up a single mass into portions by weight, dithering to
/// spread out error.  No mass is lost.  The dithering precision depends on the
/// precision of the product of \a BlockMass and \a BranchProbability.
///
/// The distribution algorithm follows.
///
///  1. Initialize by saving the sum of the weights in \a RemWeight and the
///     mass to distribute in \a RemMass.
///
///  2. For each portion:
///
///      1. Construct a branch probability, P, as the portion's weight divided
///         by the current value of \a RemWeight.
///      2. Calculate the portion's mass as \a RemMass times P.
///      3. Update \a RemWeight and \a RemMass at each portion by subtracting
///         the current portion's weight and mass.
struct DitheringDistributer {
  uint32_t RemWeight;
  BlockMass RemMass;

  DitheringDistributer(Distribution &Dist, const BlockMass &Mass);

  BlockMass takeMass(uint32_t Weight);
};
}

DitheringDistributer::DitheringDistributer(Distribution &Dist,
                                           const BlockMass &Mass) {
  Dist.normalize();
  RemWeight = Dist.Total;
  RemMass = Mass;
}

BlockMass DitheringDistributer::takeMass(uint32_t Weight) {
  assert(Weight && "invalid weight");
  assert(Weight <= RemWeight);
  BlockMass Mass = RemMass * BranchProbability(Weight, RemWeight);

  // Decrement totals (dither).
  RemWeight -= Weight;
  RemMass -= Mass;
  return Mass;
}

void Distribution::add(const BlockNode &Node, uint64_t Amount,
                       Weight::DistType Type) {
  assert(Amount && "invalid weight of 0");
  uint64_t NewTotal = Total + Amount;

  // Check for overflow.  It should be impossible to overflow twice.
  bool IsOverflow = NewTotal < Total;
  assert(!(DidOverflow && IsOverflow) && "unexpected repeated overflow");
  DidOverflow |= IsOverflow;

  // Update the total.
  Total = NewTotal;

  // Save the weight.
  Weight W;
  W.TargetNode = Node;
  W.Amount = Amount;
  W.Type = Type;
  Weights.push_back(W);
}

static void combineWeight(Weight &W, const Weight &OtherW) {
  assert(OtherW.TargetNode.isValid());
  if (!W.Amount) {
    W = OtherW;
    return;
  }
  assert(W.Type == OtherW.Type);
  assert(W.TargetNode == OtherW.TargetNode);
  assert(W.Amount < W.Amount + OtherW.Amount && "Unexpected overflow");
  W.Amount += OtherW.Amount;
}
static void combineWeightsBySorting(WeightList &Weights) {
  // Sort so edges to the same node are adjacent.
  std::sort(Weights.begin(), Weights.end(),
            [](const Weight &L,
               const Weight &R) { return L.TargetNode < R.TargetNode; });

  // Combine adjacent edges.
  WeightList::iterator O = Weights.begin();
  for (WeightList::const_iterator I = O, L = O, E = Weights.end(); I != E;
       ++O, (I = L)) {
    *O = *I;

    // Find the adjacent weights to the same node.
    for (++L; L != E && I->TargetNode == L->TargetNode; ++L)
      combineWeight(*O, *L);
  }

  // Erase extra entries.
  Weights.erase(O, Weights.end());
  return;
}
static void combineWeightsByHashing(WeightList &Weights) {
  // Collect weights into a DenseMap.
  typedef DenseMap<BlockNode::IndexType, Weight> HashTable;
  HashTable Combined(NextPowerOf2(2 * Weights.size()));
  for (const Weight &W : Weights)
    combineWeight(Combined[W.TargetNode.Index], W);

  // Check whether anything changed.
  if (Weights.size() == Combined.size())
    return;

  // Fill in the new weights.
  Weights.clear();
  Weights.reserve(Combined.size());
  for (const auto &I : Combined)
    Weights.push_back(I.second);
}
static void combineWeights(WeightList &Weights) {
  // Use a hash table for many successors to keep this linear.
  if (Weights.size() > 128) {
    combineWeightsByHashing(Weights);
    return;
  }

  combineWeightsBySorting(Weights);
}
static uint64_t shiftRightAndRound(uint64_t N, int Shift) {
  assert(Shift >= 0);
  assert(Shift < 64);
  if (!Shift)
    return N;
  return (N >> Shift) + (UINT64_C(1) & N >> (Shift - 1));
}
void Distribution::normalize() {
  // Early exit for termination nodes.
  if (Weights.empty())
    return;

  // Only bother if there are multiple successors.
  if (Weights.size() > 1)
    combineWeights(Weights);

  // Early exit when combined into a single successor.
  if (Weights.size() == 1) {
    Total = 1;
    Weights.front().Amount = 1;
    return;
  }

  // Determine how much to shift right so that the total fits into 32-bits.
  //
  // If we shift at all, shift by 1 extra.  Otherwise, the lower limit of 1
  // for each weight can cause a 32-bit overflow.
  int Shift = 0;
  if (DidOverflow)
    Shift = 33;
  else if (Total > UINT32_MAX)
    Shift = 33 - countLeadingZeros(Total);

  // Early exit if nothing needs to be scaled.
  if (!Shift)
    return;

  // Recompute the total through accumulation (rather than shifting it) so that
  // it's accurate after shifting.
  Total = 0;

  // Sum the weights to each node and shift right if necessary.
  for (Weight &W : Weights) {
    // Scale down below UINT32_MAX.  Since Shift is larger than necessary, we
    // can round here without concern about overflow.
    assert(W.TargetNode.isValid());
    W.Amount = std::max(UINT64_C(1), shiftRightAndRound(W.Amount, Shift));
    assert(W.Amount <= UINT32_MAX);

    // Update the total.
    Total += W.Amount;
  }
  assert(Total <= UINT32_MAX);
}

void BlockFrequencyInfoImplBase::clear() {
  // Swap with a default-constructed std::vector, since std::vector<>::clear()
  // does not actually clear heap storage.
  std::vector<FrequencyData>().swap(Freqs);
  std::vector<WorkingData>().swap(Working);
  Loops.clear();
}

/// \brief Clear all memory not needed downstream.
///
/// Releases all memory not used downstream.  In particular, saves Freqs.
static void cleanup(BlockFrequencyInfoImplBase &BFI) {
  std::vector<FrequencyData> SavedFreqs(std::move(BFI.Freqs));
  BFI.clear();
  BFI.Freqs = std::move(SavedFreqs);
}

bool BlockFrequencyInfoImplBase::addToDist(Distribution &Dist,
                                           const LoopData *OuterLoop,
                                           const BlockNode &Pred,
                                           const BlockNode &Succ,
                                           uint64_t Weight) {
  if (!Weight)
    Weight = 1;

  auto isLoopHeader = [&OuterLoop](const BlockNode &Node) {
    return OuterLoop && OuterLoop->isHeader(Node);
  };

  BlockNode Resolved = Working[Succ.Index].getResolvedNode();

#ifndef NDEBUG
  auto debugSuccessor = [&](const char *Type) {
    dbgs() << "  =>"
           << " [" << Type << "] weight = " << Weight;
    if (!isLoopHeader(Resolved))
      dbgs() << ", succ = " << getBlockName(Succ);
    if (Resolved != Succ)
      dbgs() << ", resolved = " << getBlockName(Resolved);
    dbgs() << "\n";
  };
  (void)debugSuccessor;
#endif

  if (isLoopHeader(Resolved)) {
    DEBUG(debugSuccessor("backedge"));
    Dist.addBackedge(OuterLoop->getHeader(), Weight);
    return true;
  }

  if (Working[Resolved.Index].getContainingLoop() != OuterLoop) {
    DEBUG(debugSuccessor("  exit  "));
    Dist.addExit(Resolved, Weight);
    return true;
  }

  if (Resolved < Pred) {
    if (!isLoopHeader(Pred)) {
      // If OuterLoop is an irreducible loop, we can't actually handle this.
      assert((!OuterLoop || !OuterLoop->isIrreducible()) &&
             "unhandled irreducible control flow");

      // Irreducible backedge.  Abort.
      DEBUG(debugSuccessor("abort!!!"));
      return false;
    }

    // If "Pred" is a loop header, then this isn't really a backedge; rather,
    // OuterLoop must be irreducible.  These false backedges can come only from
    // secondary loop headers.
    assert(OuterLoop && OuterLoop->isIrreducible() && !isLoopHeader(Resolved) &&
           "unhandled irreducible control flow");
  }

  DEBUG(debugSuccessor(" local  "));
  Dist.addLocal(Resolved, Weight);
  return true;
}

bool BlockFrequencyInfoImplBase::addLoopSuccessorsToDist(
    const LoopData *OuterLoop, LoopData &Loop, Distribution &Dist) {
  // Copy the exit map into Dist.
  for (const auto &I : Loop.Exits)
    if (!addToDist(Dist, OuterLoop, Loop.getHeader(), I.first,
                   I.second.getMass()))
      // Irreducible backedge.
      return false;

  return true;
}

/// \brief Get the maximum allowed loop scale.
///
/// Gives the maximum number of estimated iterations allowed for a loop.  Very
/// large numbers cause problems downstream (even within 64-bits).
static Float getMaxLoopScale() { return Float(1, 12); }

/// \brief Compute the loop scale for a loop.
void BlockFrequencyInfoImplBase::computeLoopScale(LoopData &Loop) {
  // Compute loop scale.
  DEBUG(dbgs() << "compute-loop-scale: " << getLoopName(Loop) << "\n");

  // LoopScale == 1 / ExitMass
  // ExitMass == HeadMass - BackedgeMass
  BlockMass ExitMass = BlockMass::getFull() - Loop.BackedgeMass;

  // Block scale stores the inverse of the scale.
  Loop.Scale = ExitMass.toFloat().inverse();

  DEBUG(dbgs() << " - exit-mass = " << ExitMass << " (" << BlockMass::getFull()
               << " - " << Loop.BackedgeMass << ")\n"
               << " - scale = " << Loop.Scale << "\n");

  if (Loop.Scale > getMaxLoopScale()) {
    Loop.Scale = getMaxLoopScale();
    DEBUG(dbgs() << " - reduced-to-max-scale: " << getMaxLoopScale() << "\n");
  }
}

/// \brief Package up a loop.
void BlockFrequencyInfoImplBase::packageLoop(LoopData &Loop) {
  DEBUG(dbgs() << "packaging-loop: " << getLoopName(Loop) << "\n");

  // Clear the subloop exits to prevent quadratic memory usage.
  for (const BlockNode &M : Loop.Nodes) {
    if (auto *Loop = Working[M.Index].getPackagedLoop())
      Loop->Exits.clear();
    DEBUG(dbgs() << " - node: " << getBlockName(M.Index) << "\n");
  }
  Loop.IsPackaged = true;
}

void BlockFrequencyInfoImplBase::distributeMass(const BlockNode &Source,
                                                LoopData *OuterLoop,
                                                Distribution &Dist) {
  BlockMass Mass = Working[Source.Index].getMass();
  DEBUG(dbgs() << "  => mass:  " << Mass << "\n");

  // Distribute mass to successors as laid out in Dist.
  DitheringDistributer D(Dist, Mass);

#ifndef NDEBUG
  auto debugAssign = [&](const BlockNode &T, const BlockMass &M,
                         const char *Desc) {
    dbgs() << "  => assign " << M << " (" << D.RemMass << ")";
    if (Desc)
      dbgs() << " [" << Desc << "]";
    if (T.isValid())
      dbgs() << " to " << getBlockName(T);
    dbgs() << "\n";
  };
  (void)debugAssign;
#endif

  for (const Weight &W : Dist.Weights) {
    // Check for a local edge (non-backedge and non-exit).
    BlockMass Taken = D.takeMass(W.Amount);
    if (W.Type == Weight::Local) {
      Working[W.TargetNode.Index].getMass() += Taken;
      DEBUG(debugAssign(W.TargetNode, Taken, nullptr));
      continue;
    }

    // Backedges and exits only make sense if we're processing a loop.
    assert(OuterLoop && "backedge or exit outside of loop");

    // Check for a backedge.
    if (W.Type == Weight::Backedge) {
      OuterLoop->BackedgeMass += Taken;
      DEBUG(debugAssign(BlockNode(), Taken, "back"));
      continue;
    }

    // This must be an exit.
    assert(W.Type == Weight::Exit);
    OuterLoop->Exits.push_back(std::make_pair(W.TargetNode, Taken));
    DEBUG(debugAssign(W.TargetNode, Taken, "exit"));
  }
}

static void convertFloatingToInteger(BlockFrequencyInfoImplBase &BFI,
                                     const Float &Min, const Float &Max) {
  // Scale the Factor to a size that creates integers.  Ideally, integers would
  // be scaled so that Max == UINT64_MAX so that they can be best
  // differentiated.  However, the register allocator currently deals poorly
  // with large numbers.  Instead, push Min up a little from 1 to give some
  // room to differentiate small, unequal numbers.
  //
  // TODO: fix issues downstream so that ScalingFactor can be Float(1,64)/Max.
  Float ScalingFactor = Min.inverse();
  if ((Max / Min).lg() < 60)
    ScalingFactor <<= 3;

  // Translate the floats to integers.
  DEBUG(dbgs() << "float-to-int: min = " << Min << ", max = " << Max
               << ", factor = " << ScalingFactor << "\n");
  for (size_t Index = 0; Index < BFI.Freqs.size(); ++Index) {
    Float Scaled = BFI.Freqs[Index].Floating * ScalingFactor;
    BFI.Freqs[Index].Integer = std::max(UINT64_C(1), Scaled.toInt<uint64_t>());
    DEBUG(dbgs() << " - " << BFI.getBlockName(Index) << ": float = "
                 << BFI.Freqs[Index].Floating << ", scaled = " << Scaled
                 << ", int = " << BFI.Freqs[Index].Integer << "\n");
  }
}

/// \brief Unwrap a loop package.
///
/// Visits all the members of a loop, adjusting their BlockData according to
/// the loop's pseudo-node.
static void unwrapLoop(BlockFrequencyInfoImplBase &BFI, LoopData &Loop) {
  DEBUG(dbgs() << "unwrap-loop-package: " << BFI.getLoopName(Loop)
               << ": mass = " << Loop.Mass << ", scale = " << Loop.Scale
               << "\n");
  Loop.Scale *= Loop.Mass.toFloat();
  Loop.IsPackaged = false;
  DEBUG(dbgs() << "  => combined-scale = " << Loop.Scale << "\n");

  // Propagate the head scale through the loop.  Since members are visited in
  // RPO, the head scale will be updated by the loop scale first, and then the
  // final head scale will be used for updated the rest of the members.
  for (const BlockNode &N : Loop.Nodes) {
    const auto &Working = BFI.Working[N.Index];
    Float &F = Working.isAPackage() ? Working.getPackagedLoop()->Scale
                                    : BFI.Freqs[N.Index].Floating;
    Float New = Loop.Scale * F;
    DEBUG(dbgs() << " - " << BFI.getBlockName(N) << ": " << F << " => " << New
                 << "\n");
    F = New;
  }
}

void BlockFrequencyInfoImplBase::unwrapLoops() {
  // Set initial frequencies from loop-local masses.
  for (size_t Index = 0; Index < Working.size(); ++Index)
    Freqs[Index].Floating = Working[Index].Mass.toFloat();

  for (LoopData &Loop : Loops)
    unwrapLoop(*this, Loop);
}

void BlockFrequencyInfoImplBase::finalizeMetrics() {
  // Unwrap loop packages in reverse post-order, tracking min and max
  // frequencies.
  auto Min = Float::getLargest();
  auto Max = Float::getZero();
  for (size_t Index = 0; Index < Working.size(); ++Index) {
    // Update min/max scale.
    Min = std::min(Min, Freqs[Index].Floating);
    Max = std::max(Max, Freqs[Index].Floating);
  }

  // Convert to integers.
  convertFloatingToInteger(*this, Min, Max);

  // Clean up data structures.
  cleanup(*this);

  // Print out the final stats.
  DEBUG(dump());
}

BlockFrequency
BlockFrequencyInfoImplBase::getBlockFreq(const BlockNode &Node) const {
  if (!Node.isValid())
    return 0;
  return Freqs[Node.Index].Integer;
}
Float
BlockFrequencyInfoImplBase::getFloatingBlockFreq(const BlockNode &Node) const {
  if (!Node.isValid())
    return Float::getZero();
  return Freqs[Node.Index].Floating;
}

std::string
BlockFrequencyInfoImplBase::getBlockName(const BlockNode &Node) const {
  return std::string();
}
std::string
BlockFrequencyInfoImplBase::getLoopName(const LoopData &Loop) const {
  return getBlockName(Loop.getHeader()) + (Loop.isIrreducible() ? "**" : "*");
}

raw_ostream &
BlockFrequencyInfoImplBase::printBlockFreq(raw_ostream &OS,
                                           const BlockNode &Node) const {
  return OS << getFloatingBlockFreq(Node);
}

raw_ostream &
BlockFrequencyInfoImplBase::printBlockFreq(raw_ostream &OS,
                                           const BlockFrequency &Freq) const {
  Float Block(Freq.getFrequency(), 0);
  Float Entry(getEntryFreq(), 0);

  return OS << Block / Entry;
}

void IrreducibleGraph::addNodesInLoop(const BFIBase::LoopData &OuterLoop) {
  Start = OuterLoop.getHeader();
  Nodes.reserve(OuterLoop.Nodes.size());
  for (auto N : OuterLoop.Nodes)
    addNode(N);
  indexNodes();
}
void IrreducibleGraph::addNodesInFunction() {
  Start = 0;
  for (uint32_t Index = 0; Index < BFI.Working.size(); ++Index)
    if (!BFI.Working[Index].isPackaged())
      addNode(Index);
  indexNodes();
}
void IrreducibleGraph::indexNodes() {
  for (auto &I : Nodes)
    Lookup[I.Node.Index] = &I;
}
void IrreducibleGraph::addEdge(IrrNode &Irr, const BlockNode &Succ,
                               const BFIBase::LoopData *OuterLoop) {
  if (OuterLoop && OuterLoop->isHeader(Succ))
    return;
  auto L = Lookup.find(Succ.Index);
  if (L == Lookup.end())
    return;
  IrrNode &SuccIrr = *L->second;
  Irr.Edges.push_back(&SuccIrr);
  SuccIrr.Edges.push_front(&Irr);
  ++SuccIrr.NumIn;
}

namespace llvm {
template <> struct GraphTraits<IrreducibleGraph> {
  typedef bfi_detail::IrreducibleGraph GraphT;

  typedef const GraphT::IrrNode NodeType;
  typedef GraphT::IrrNode::iterator ChildIteratorType;

  static const NodeType *getEntryNode(const GraphT &G) {
    return G.StartIrr;
  }
  static ChildIteratorType child_begin(NodeType *N) { return N->succ_begin(); }
  static ChildIteratorType child_end(NodeType *N) { return N->succ_end(); }
};
}

/// \brief Find extra irreducible headers.
///
/// Find entry blocks and other blocks with backedges, which exist when \c G
/// contains irreducible sub-SCCs.
static void findIrreducibleHeaders(
    const BlockFrequencyInfoImplBase &BFI,
    const IrreducibleGraph &G,
    const std::vector<const IrreducibleGraph::IrrNode *> &SCC,
    LoopData::NodeList &Headers, LoopData::NodeList &Others) {
  // Map from nodes in the SCC to whether it's an entry block.
  SmallDenseMap<const IrreducibleGraph::IrrNode *, bool, 8> InSCC;

  // InSCC also acts the set of nodes in the graph.  Seed it.
  for (const auto *I : SCC)
    InSCC[I] = false;

  for (auto I = InSCC.begin(), E = InSCC.end(); I != E; ++I) {
    auto &Irr = *I->first;
    for (const auto *P : make_range(Irr.pred_begin(), Irr.pred_end())) {
      if (InSCC.count(P))
        continue;

      // This is an entry block.
      I->second = true;
      Headers.push_back(Irr.Node);
      DEBUG(dbgs() << "  => entry = " << BFI.getBlockName(Irr.Node) << "\n");
      break;
    }
  }
  assert(Headers.size() >= 2 && "Should be irreducible");
  if (Headers.size() == InSCC.size()) {
    // Every block is a header.
    std::sort(Headers.begin(), Headers.end());
    return;
  }

  // Look for extra headers from irreducible sub-SCCs.
  for (const auto &I : InSCC) {
    // Entry blocks are already headers.
    if (I.second)
      continue;

    auto &Irr = *I.first;
    for (const auto *P : make_range(Irr.pred_begin(), Irr.pred_end())) {
      // Skip forward edges.
      if (P->Node < Irr.Node)
        continue;

      // Skip predecessors from entry blocks.  These can have inverted
      // ordering.
      if (InSCC.lookup(P))
        continue;

      // Store the extra header.
      Headers.push_back(Irr.Node);
      DEBUG(dbgs() << "  => extra = " << BFI.getBlockName(Irr.Node) << "\n");
      break;
    }
    if (Headers.back() == Irr.Node)
      // Added this as a header.
      continue;

    // This is not a header.
    Others.push_back(Irr.Node);
    DEBUG(dbgs() << "  => other = " << BFI.getBlockName(Irr.Node) << "\n");
  }
  std::sort(Headers.begin(), Headers.end());
  std::sort(Others.begin(), Others.end());
}

static void createIrreducibleLoop(
    BlockFrequencyInfoImplBase &BFI, const IrreducibleGraph &G,
    LoopData *OuterLoop, std::list<LoopData>::iterator Insert,
    const std::vector<const IrreducibleGraph::IrrNode *> &SCC) {
  // Translate the SCC into RPO.
  DEBUG(dbgs() << " - found-scc\n");

  LoopData::NodeList Headers;
  LoopData::NodeList Others;
  findIrreducibleHeaders(BFI, G, SCC, Headers, Others);

  auto Loop = BFI.Loops.emplace(Insert, OuterLoop, Headers.begin(),
                                Headers.end(), Others.begin(), Others.end());

  // Update loop hierarchy.
  for (const auto &N : Loop->Nodes)
    if (BFI.Working[N.Index].isLoopHeader())
      BFI.Working[N.Index].Loop->Parent = &*Loop;
    else
      BFI.Working[N.Index].Loop = &*Loop;
}

iterator_range<std::list<LoopData>::iterator>
BlockFrequencyInfoImplBase::analyzeIrreducible(
    const IrreducibleGraph &G, LoopData *OuterLoop,
    std::list<LoopData>::iterator Insert) {
  assert((OuterLoop == nullptr) == (Insert == Loops.begin()));
  auto Prev = OuterLoop ? std::prev(Insert) : Loops.end();

  for (auto I = scc_begin(G); !I.isAtEnd(); ++I) {
    if (I->size() < 2)
      continue;

    // Translate the SCC into RPO.
    createIrreducibleLoop(*this, G, OuterLoop, Insert, *I);
  }

  if (OuterLoop)
    return make_range(std::next(Prev), Insert);
  return make_range(Loops.begin(), Insert);
}

void
BlockFrequencyInfoImplBase::updateLoopWithIrreducible(LoopData &OuterLoop) {
  OuterLoop.Exits.clear();
  OuterLoop.BackedgeMass = BlockMass::getEmpty();
  auto O = OuterLoop.Nodes.begin() + 1;
  for (auto I = O, E = OuterLoop.Nodes.end(); I != E; ++I)
    if (!Working[I->Index].isPackaged())
      *O++ = *I;
  OuterLoop.Nodes.erase(O, OuterLoop.Nodes.end());
}
