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
BlockMass &BlockMass::operator*=(const BranchProbability &P) {
  uint32_t N = P.getNumerator(), D = P.getDenominator();
  assert(D && "divide by 0");
  assert(N <= D && "fraction greater than 1");

  // Fast path for multiplying by 1.0.
  if (!Mass || N == D)
    return *this;

  // Get as much precision as we can.
  int Shift = countLeadingZeros(Mass);
  uint64_t ShiftedQuotient = (Mass << Shift) / D;
  uint64_t Product = ShiftedQuotient * N >> Shift;

  // Now check for what's lost.
  uint64_t Left = ShiftedQuotient * (D - N) >> Shift;
  uint64_t Lost = Mass - Product - Left;

  // TODO: prove this assertion.
  assert(Lost <= UINT32_MAX);

  // Take the product plus a portion of the spoils.
  Mass = Product + Lost * N / D;
  return *this;
}

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
///
/// Mass is distributed in two ways: full distribution and forward
/// distribution.  The latter ignores backedges, and uses the parallel fields
/// \a RemForwardWeight and \a RemForwardMass.
struct DitheringDistributer {
  uint32_t RemWeight;
  uint32_t RemForwardWeight;

  BlockMass RemMass;
  BlockMass RemForwardMass;

  DitheringDistributer(Distribution &Dist, const BlockMass &Mass);

  BlockMass takeLocalMass(uint32_t Weight) {
    (void)takeMass(Weight);
    return takeForwardMass(Weight);
  }
  BlockMass takeExitMass(uint32_t Weight) {
    (void)takeForwardMass(Weight);
    return takeMass(Weight);
  }
  BlockMass takeBackedgeMass(uint32_t Weight) { return takeMass(Weight); }

private:
  BlockMass takeForwardMass(uint32_t Weight);
  BlockMass takeMass(uint32_t Weight);
};
}

DitheringDistributer::DitheringDistributer(Distribution &Dist,
                                           const BlockMass &Mass) {
  Dist.normalize();
  RemWeight = Dist.Total;
  RemForwardWeight = Dist.ForwardTotal;
  RemMass = Mass;
  RemForwardMass = Dist.ForwardTotal ? Mass : BlockMass();
}

BlockMass DitheringDistributer::takeForwardMass(uint32_t Weight) {
  // Compute the amount of mass to take.
  assert(Weight && "invalid weight");
  assert(Weight <= RemForwardWeight);
  BlockMass Mass = RemForwardMass * BranchProbability(Weight, RemForwardWeight);

  // Decrement totals (dither).
  RemForwardWeight -= Weight;
  RemForwardMass -= Mass;
  return Mass;
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

  if (Type == Weight::Backedge)
    return;

  // Update forward total.  Don't worry about overflow here, since then Total
  // will exceed 32-bits and they'll both be recomputed in normalize().
  ForwardTotal += Amount;
}

static void combineWeight(Weight &W, const Weight &OtherW) {
  assert(OtherW.TargetNode.isValid());
  if (!W.Amount) {
    W = OtherW;
    return;
  }
  assert(W.Type == OtherW.Type);
  assert(W.TargetNode == OtherW.TargetNode);
  assert(W.Amount < W.Amount + OtherW.Amount);
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
    ForwardTotal = Weights.front().Type != Weight::Backedge;
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
  // it's accurate after shifting.  ForwardTotal is dirty here anyway.
  Total = 0;
  ForwardTotal = 0;

  // Sum the weights to each node and shift right if necessary.
  for (Weight &W : Weights) {
    // Scale down below UINT32_MAX.  Since Shift is larger than necessary, we
    // can round here without concern about overflow.
    assert(W.TargetNode.isValid());
    W.Amount = std::max(UINT64_C(1), shiftRightAndRound(W.Amount, Shift));
    assert(W.Amount <= UINT32_MAX);

    // Update the total.
    Total += W.Amount;
    if (W.Type == Weight::Backedge)
      continue;

    // Update the forward total.
    ForwardTotal += W.Amount;
  }
  assert(Total <= UINT32_MAX);
}

void BlockFrequencyInfoImplBase::clear() {
  // Swap with a default-constructed std::vector, since std::vector<>::clear()
  // does not actually clear heap storage.
  std::vector<FrequencyData>().swap(Freqs);
  std::vector<WorkingData>().swap(Working);
  std::vector<LoopData>().swap(PackagedLoops);
}

/// \brief Clear all memory not needed downstream.
///
/// Releases all memory not used downstream.  In particular, saves Freqs.
static void cleanup(BlockFrequencyInfoImplBase &BFI) {
  std::vector<FrequencyData> SavedFreqs(std::move(BFI.Freqs));
  BFI.clear();
  BFI.Freqs = std::move(SavedFreqs);
}

/// \brief Get a possibly packaged node.
///
/// Get the node currently representing Node, which could be a containing
/// loop.
///
/// This function should only be called when distributing mass.  As long as
/// there are no irreducilbe edges to Node, then it will have complexity O(1)
/// in this context.
///
/// In general, the complexity is O(L), where L is the number of loop headers
/// Node has been packaged into.  Since this method is called in the context
/// of distributing mass, L will be the number of loop headers an early exit
/// edge jumps out of.
static BlockNode getPackagedNode(const BlockFrequencyInfoImplBase &BFI,
                                 const BlockNode &Node) {
  assert(Node.isValid());
  if (!BFI.Working[Node.Index].IsPackaged)
    return Node;
  if (!BFI.Working[Node.Index].ContainingLoop.isValid())
    return Node;
  return getPackagedNode(BFI, BFI.Working[Node.Index].ContainingLoop);
}

/// \brief Get the appropriate mass for a possible pseudo-node loop package.
///
/// Get appropriate mass for Node.  If Node is a loop-header (whose loop has
/// been packaged), returns the mass of its pseudo-node.  If it's a node inside
/// a packaged loop, it returns the loop's pseudo-node.
static BlockMass &getPackageMass(BlockFrequencyInfoImplBase &BFI,
                                 const BlockNode &Node) {
  assert(Node.isValid());
  assert(!BFI.Working[Node.Index].IsPackaged);
  if (!BFI.Working[Node.Index].IsAPackage)
    return BFI.Working[Node.Index].Mass;

  return BFI.getLoopPackage(Node).Mass;
}

void BlockFrequencyInfoImplBase::addToDist(Distribution &Dist,
                                           const BlockNode &LoopHead,
                                           const BlockNode &Pred,
                                           const BlockNode &Succ,
                                           uint64_t Weight) {
  if (!Weight)
    Weight = 1;

#ifndef NDEBUG
  auto debugSuccessor = [&](const char *Type, const BlockNode &Resolved) {
    dbgs() << "  =>"
           << " [" << Type << "] weight = " << Weight;
    if (Succ != LoopHead)
      dbgs() << ", succ = " << getBlockName(Succ);
    if (Resolved != Succ)
      dbgs() << ", resolved = " << getBlockName(Resolved);
    dbgs() << "\n";
  };
  (void)debugSuccessor;
#endif

  if (Succ == LoopHead) {
    DEBUG(debugSuccessor("backedge", Succ));
    Dist.addBackedge(LoopHead, Weight);
    return;
  }
  BlockNode Resolved = getPackagedNode(*this, Succ);
  assert(Resolved != LoopHead);

  if (Working[Resolved.Index].ContainingLoop != LoopHead) {
    DEBUG(debugSuccessor("  exit  ", Resolved));
    Dist.addExit(Resolved, Weight);
    return;
  }

  if (!LoopHead.isValid() && Resolved < Pred) {
    // Irreducible backedge.  Skip this edge in the distribution.
    DEBUG(debugSuccessor("skipped ", Resolved));
    return;
  }

  DEBUG(debugSuccessor(" local  ", Resolved));
  Dist.addLocal(Resolved, Weight);
}

void BlockFrequencyInfoImplBase::addLoopSuccessorsToDist(
    const BlockNode &LoopHead, const BlockNode &LocalLoopHead,
    Distribution &Dist) {
  LoopData &LoopPackage = getLoopPackage(LocalLoopHead);
  const LoopData::ExitMap &Exits = LoopPackage.Exits;

  // Copy the exit map into Dist.
  for (const auto &I : Exits)
    addToDist(Dist, LoopHead, LocalLoopHead, I.first, I.second.getMass());

  // We don't need this map any more.  Clear it to prevent quadratic memory
  // usage in deeply nested loops with irreducible control flow.
  LoopPackage.Exits.clear();
}

/// \brief Get the maximum allowed loop scale.
///
/// Gives the maximum number of estimated iterations allowed for a loop.  Very
/// large numbers cause problems downstream (even within 64-bits).
static Float getMaxLoopScale() { return Float(1, 12); }

/// \brief Compute the loop scale for a loop.
void BlockFrequencyInfoImplBase::computeLoopScale(const BlockNode &LoopHead) {
  // Compute loop scale.
  DEBUG(dbgs() << "compute-loop-scale: " << getBlockName(LoopHead) << "\n");

  // LoopScale == 1 / ExitMass
  // ExitMass == HeadMass - BackedgeMass
  LoopData &LoopPackage = getLoopPackage(LoopHead);
  BlockMass ExitMass = BlockMass::getFull() - LoopPackage.BackedgeMass;

  // Block scale stores the inverse of the scale.
  LoopPackage.Scale = ExitMass.toFloat().inverse();

  DEBUG(dbgs() << " - exit-mass = " << ExitMass << " (" << BlockMass::getFull()
               << " - " << LoopPackage.BackedgeMass << ")\n"
               << " - scale = " << LoopPackage.Scale << "\n");

  if (LoopPackage.Scale > getMaxLoopScale()) {
    LoopPackage.Scale = getMaxLoopScale();
    DEBUG(dbgs() << " - reduced-to-max-scale: " << getMaxLoopScale() << "\n");
  }
}

/// \brief Package up a loop.
void BlockFrequencyInfoImplBase::packageLoop(const BlockNode &LoopHead) {
  DEBUG(dbgs() << "packaging-loop: " << getBlockName(LoopHead) << "\n");
  Working[LoopHead.Index].IsAPackage = true;
  for (const BlockNode &M : getLoopPackage(LoopHead).Members) {
    DEBUG(dbgs() << " - node: " << getBlockName(M.Index) << "\n");
    Working[M.Index].IsPackaged = true;
  }
}

void BlockFrequencyInfoImplBase::distributeMass(const BlockNode &Source,
                                                const BlockNode &LoopHead,
                                                Distribution &Dist) {
  BlockMass Mass = getPackageMass(*this, Source);
  DEBUG(dbgs() << "  => mass:  " << Mass
               << " (    general     |    forward     )\n");

  // Distribute mass to successors as laid out in Dist.
  DitheringDistributer D(Dist, Mass);

#ifndef NDEBUG
  auto debugAssign = [&](const BlockNode &T, const BlockMass &M,
                         const char *Desc) {
    dbgs() << "  => assign " << M << " (" << D.RemMass << "|"
           << D.RemForwardMass << ")";
    if (Desc)
      dbgs() << " [" << Desc << "]";
    if (T.isValid())
      dbgs() << " to " << getBlockName(T);
    dbgs() << "\n";
  };
  (void)debugAssign;
#endif

  LoopData *LoopPackage = 0;
  if (LoopHead.isValid())
    LoopPackage = &getLoopPackage(LoopHead);
  for (const Weight &W : Dist.Weights) {
    // Check for a local edge (forward and non-exit).
    if (W.Type == Weight::Local) {
      BlockMass Local = D.takeLocalMass(W.Amount);
      getPackageMass(*this, W.TargetNode) += Local;
      DEBUG(debugAssign(W.TargetNode, Local, nullptr));
      continue;
    }

    // Backedges and exits only make sense if we're processing a loop.
    assert(LoopPackage && "backedge or exit outside of loop");

    // Check for a backedge.
    if (W.Type == Weight::Backedge) {
      BlockMass Back = D.takeBackedgeMass(W.Amount);
      LoopPackage->BackedgeMass += Back;
      DEBUG(debugAssign(BlockNode(), Back, "back"));
      continue;
    }

    // This must be an exit.
    assert(W.Type == Weight::Exit);
    BlockMass Exit = D.takeExitMass(W.Amount);
    LoopPackage->Exits.push_back(std::make_pair(W.TargetNode, Exit));
    DEBUG(debugAssign(W.TargetNode, Exit, "exit"));
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

static void scaleBlockData(BlockFrequencyInfoImplBase &BFI,
                           const BlockNode &Node,
                           const LoopData &Loop) {
  Float F = Loop.Mass.toFloat() * Loop.Scale;

  Float &Current = BFI.Freqs[Node.Index].Floating;
  Float Updated = Current * F;

  DEBUG(dbgs() << " - " << BFI.getBlockName(Node) << ": " << Current << " => "
               << Updated << "\n");

  Current = Updated;
}

/// \brief Unwrap a loop package.
///
/// Visits all the members of a loop, adjusting their BlockData according to
/// the loop's pseudo-node.
static void unwrapLoopPackage(BlockFrequencyInfoImplBase &BFI,
                              const BlockNode &Head) {
  assert(Head.isValid());

  LoopData &LoopPackage = BFI.getLoopPackage(Head);
  DEBUG(dbgs() << "unwrap-loop-package: " << BFI.getBlockName(Head)
               << ": mass = " << LoopPackage.Mass
               << ", scale = " << LoopPackage.Scale << "\n");
  scaleBlockData(BFI, Head, LoopPackage);

  // Propagate the head scale through the loop.  Since members are visited in
  // RPO, the head scale will be updated by the loop scale first, and then the
  // final head scale will be used for updated the rest of the members.
  for (const BlockNode &M : LoopPackage.Members) {
    const FrequencyData &HeadData = BFI.Freqs[Head.Index];
    FrequencyData &Freqs = BFI.Freqs[M.Index];
    Float NewFreq = Freqs.Floating * HeadData.Floating;
    DEBUG(dbgs() << " - " << BFI.getBlockName(M) << ": " << Freqs.Floating
                 << " => " << NewFreq << "\n");
    Freqs.Floating = NewFreq;
  }
}

void BlockFrequencyInfoImplBase::finalizeMetrics() {
  // Set initial frequencies from loop-local masses.
  for (size_t Index = 0; Index < Working.size(); ++Index)
    Freqs[Index].Floating = Working[Index].Mass.toFloat();

  // Unwrap loop packages in reverse post-order, tracking min and max
  // frequencies.
  auto Min = Float::getLargest();
  auto Max = Float::getZero();
  for (size_t Index = 0; Index < Working.size(); ++Index) {
    if (Working[Index].isLoopHeader())
      unwrapLoopPackage(*this, BlockNode(Index));

    // Update max scale.
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
