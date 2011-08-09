//===------------ FixedLenDecoderEmitter.cpp - Decoder Generator ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// It contains the tablegen backend that emits the decoder functions for
// targets with fixed length instruction set.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "decoder-emitter"

#include "FixedLenDecoderEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <map>
#include <string>

using namespace llvm;

// The set (BIT_TRUE, BIT_FALSE, BIT_UNSET) represents a ternary logic system
// for a bit value.
//
// BIT_UNFILTERED is used as the init value for a filter position.  It is used
// only for filter processings.
typedef enum {
  BIT_TRUE,      // '1'
  BIT_FALSE,     // '0'
  BIT_UNSET,     // '?'
  BIT_UNFILTERED // unfiltered
} bit_value_t;

static bool ValueSet(bit_value_t V) {
  return (V == BIT_TRUE || V == BIT_FALSE);
}
static bool ValueNotSet(bit_value_t V) {
  return (V == BIT_UNSET);
}
static int Value(bit_value_t V) {
  return ValueNotSet(V) ? -1 : (V == BIT_FALSE ? 0 : 1);
}
static bit_value_t bitFromBits(BitsInit &bits, unsigned index) {
  if (BitInit *bit = dynamic_cast<BitInit*>(bits.getBit(index)))
    return bit->getValue() ? BIT_TRUE : BIT_FALSE;

  // The bit is uninitialized.
  return BIT_UNSET;
}
// Prints the bit value for each position.
static void dumpBits(raw_ostream &o, BitsInit &bits) {
  unsigned index;

  for (index = bits.getNumBits(); index > 0; index--) {
    switch (bitFromBits(bits, index - 1)) {
    case BIT_TRUE:
      o << "1";
      break;
    case BIT_FALSE:
      o << "0";
      break;
    case BIT_UNSET:
      o << "_";
      break;
    default:
      assert(0 && "unexpected return value from bitFromBits");
    }
  }
}

static BitsInit &getBitsField(const Record &def, const char *str) {
  BitsInit *bits = def.getValueAsBitsInit(str);
  return *bits;
}

// Forward declaration.
class FilterChooser;

// Representation of the instruction to work on.
typedef std::vector<bit_value_t> insn_t;

/// Filter - Filter works with FilterChooser to produce the decoding tree for
/// the ISA.
///
/// It is useful to think of a Filter as governing the switch stmts of the
/// decoding tree in a certain level.  Each case stmt delegates to an inferior
/// FilterChooser to decide what further decoding logic to employ, or in another
/// words, what other remaining bits to look at.  The FilterChooser eventually
/// chooses a best Filter to do its job.
///
/// This recursive scheme ends when the number of Opcodes assigned to the
/// FilterChooser becomes 1 or if there is a conflict.  A conflict happens when
/// the Filter/FilterChooser combo does not know how to distinguish among the
/// Opcodes assigned.
///
/// An example of a conflict is
///
/// Conflict:
///                     111101000.00........00010000....
///                     111101000.00........0001........
///                     1111010...00........0001........
///                     1111010...00....................
///                     1111010.........................
///                     1111............................
///                     ................................
///     VST4q8a         111101000_00________00010000____
///     VST4q8b         111101000_00________00010000____
///
/// The Debug output shows the path that the decoding tree follows to reach the
/// the conclusion that there is a conflict.  VST4q8a is a vst4 to double-spaced
/// even registers, while VST4q8b is a vst4 to double-spaced odd regsisters.
///
/// The encoding info in the .td files does not specify this meta information,
/// which could have been used by the decoder to resolve the conflict.  The
/// decoder could try to decode the even/odd register numbering and assign to
/// VST4q8a or VST4q8b, but for the time being, the decoder chooses the "a"
/// version and return the Opcode since the two have the same Asm format string.
class Filter {
protected:
  FilterChooser *Owner; // points to the FilterChooser who owns this filter
  unsigned StartBit; // the starting bit position
  unsigned NumBits; // number of bits to filter
  bool Mixed; // a mixed region contains both set and unset bits

  // Map of well-known segment value to the set of uid's with that value.
  std::map<uint64_t, std::vector<unsigned> > FilteredInstructions;

  // Set of uid's with non-constant segment values.
  std::vector<unsigned> VariableInstructions;

  // Map of well-known segment value to its delegate.
  std::map<unsigned, FilterChooser*> FilterChooserMap;

  // Number of instructions which fall under FilteredInstructions category.
  unsigned NumFiltered;

  // Keeps track of the last opcode in the filtered bucket.
  unsigned LastOpcFiltered;

  // Number of instructions which fall under VariableInstructions category.
  unsigned NumVariable;

public:
  unsigned getNumFiltered() { return NumFiltered; }
  unsigned getNumVariable() { return NumVariable; }
  unsigned getSingletonOpc() {
    assert(NumFiltered == 1);
    return LastOpcFiltered;
  }
  // Return the filter chooser for the group of instructions without constant
  // segment values.
  FilterChooser &getVariableFC() {
    assert(NumFiltered == 1);
    assert(FilterChooserMap.size() == 1);
    return *(FilterChooserMap.find((unsigned)-1)->second);
  }

  Filter(const Filter &f);
  Filter(FilterChooser &owner, unsigned startBit, unsigned numBits, bool mixed);

  ~Filter();

  // Divides the decoding task into sub tasks and delegates them to the
  // inferior FilterChooser's.
  //
  // A special case arises when there's only one entry in the filtered
  // instructions.  In order to unambiguously decode the singleton, we need to
  // match the remaining undecoded encoding bits against the singleton.
  void recurse();

  // Emit code to decode instructions given a segment or segments of bits.
  void emit(raw_ostream &o, unsigned &Indentation);

  // Returns the number of fanout produced by the filter.  More fanout implies
  // the filter distinguishes more categories of instructions.
  unsigned usefulness() const;
}; // End of class Filter

// These are states of our finite state machines used in FilterChooser's
// filterProcessor() which produces the filter candidates to use.
typedef enum {
  ATTR_NONE,
  ATTR_FILTERED,
  ATTR_ALL_SET,
  ATTR_ALL_UNSET,
  ATTR_MIXED
} bitAttr_t;

/// FilterChooser - FilterChooser chooses the best filter among a set of Filters
/// in order to perform the decoding of instructions at the current level.
///
/// Decoding proceeds from the top down.  Based on the well-known encoding bits
/// of instructions available, FilterChooser builds up the possible Filters that
/// can further the task of decoding by distinguishing among the remaining
/// candidate instructions.
///
/// Once a filter has been chosen, it is called upon to divide the decoding task
/// into sub-tasks and delegates them to its inferior FilterChoosers for further
/// processings.
///
/// It is useful to think of a Filter as governing the switch stmts of the
/// decoding tree.  And each case is delegated to an inferior FilterChooser to
/// decide what further remaining bits to look at.
class FilterChooser {
protected:
  friend class Filter;

  // Vector of codegen instructions to choose our filter.
  const std::vector<const CodeGenInstruction*> &AllInstructions;

  // Vector of uid's for this filter chooser to work on.
  const std::vector<unsigned> Opcodes;

  // Lookup table for the operand decoding of instructions.
  std::map<unsigned, std::vector<OperandInfo> > &Operands;

  // Vector of candidate filters.
  std::vector<Filter> Filters;

  // Array of bit values passed down from our parent.
  // Set to all BIT_UNFILTERED's for Parent == NULL.
  std::vector<bit_value_t> FilterBitValues;

  // Links to the FilterChooser above us in the decoding tree.
  FilterChooser *Parent;

  // Index of the best filter from Filters.
  int BestIndex;

  // Width of instructions
  unsigned BitWidth;

public:
  FilterChooser(const FilterChooser &FC) :
    AllInstructions(FC.AllInstructions), Opcodes(FC.Opcodes),
      Operands(FC.Operands), Filters(FC.Filters),
      FilterBitValues(FC.FilterBitValues), Parent(FC.Parent),
      BestIndex(FC.BestIndex), BitWidth(FC.BitWidth) { }

  FilterChooser(const std::vector<const CodeGenInstruction*> &Insts,
                const std::vector<unsigned> &IDs,
    std::map<unsigned, std::vector<OperandInfo> > &Ops,
                unsigned BW) :
      AllInstructions(Insts), Opcodes(IDs), Operands(Ops), Filters(),
      Parent(NULL), BestIndex(-1), BitWidth(BW) {
    for (unsigned i = 0; i < BitWidth; ++i)
      FilterBitValues.push_back(BIT_UNFILTERED);

    doFilter();
  }

  FilterChooser(const std::vector<const CodeGenInstruction*> &Insts,
                const std::vector<unsigned> &IDs,
        std::map<unsigned, std::vector<OperandInfo> > &Ops,
                std::vector<bit_value_t> &ParentFilterBitValues,
                FilterChooser &parent) :
      AllInstructions(Insts), Opcodes(IDs), Operands(Ops),
      Filters(), FilterBitValues(ParentFilterBitValues),
      Parent(&parent), BestIndex(-1), BitWidth(parent.BitWidth) {
    doFilter();
  }

  // The top level filter chooser has NULL as its parent.
  bool isTopLevel() { return Parent == NULL; }

  // Emit the top level typedef and decodeInstruction() function.
  void emitTop(raw_ostream &o, unsigned Indentation, std::string Namespace);

protected:
  // Populates the insn given the uid.
  void insnWithID(insn_t &Insn, unsigned Opcode) const {
    BitsInit &Bits = getBitsField(*AllInstructions[Opcode]->TheDef, "Inst");

    for (unsigned i = 0; i < BitWidth; ++i)
      Insn.push_back(bitFromBits(Bits, i));
  }

  // Returns the record name.
  const std::string &nameWithID(unsigned Opcode) const {
    return AllInstructions[Opcode]->TheDef->getName();
  }

  // Populates the field of the insn given the start position and the number of
  // consecutive bits to scan for.
  //
  // Returns false if there exists any uninitialized bit value in the range.
  // Returns true, otherwise.
  bool fieldFromInsn(uint64_t &Field, insn_t &Insn, unsigned StartBit,
      unsigned NumBits) const;

  /// dumpFilterArray - dumpFilterArray prints out debugging info for the given
  /// filter array as a series of chars.
  void dumpFilterArray(raw_ostream &o, std::vector<bit_value_t> & filter);

  /// dumpStack - dumpStack traverses the filter chooser chain and calls
  /// dumpFilterArray on each filter chooser up to the top level one.
  void dumpStack(raw_ostream &o, const char *prefix);

  Filter &bestFilter() {
    assert(BestIndex != -1 && "BestIndex not set");
    return Filters[BestIndex];
  }

  // Called from Filter::recurse() when singleton exists.  For debug purpose.
  void SingletonExists(unsigned Opc);

  bool PositionFiltered(unsigned i) {
    return ValueSet(FilterBitValues[i]);
  }

  // Calculates the island(s) needed to decode the instruction.
  // This returns a lit of undecoded bits of an instructions, for example,
  // Inst{20} = 1 && Inst{3-0} == 0b1111 represents two islands of yet-to-be
  // decoded bits in order to verify that the instruction matches the Opcode.
  unsigned getIslands(std::vector<unsigned> &StartBits,
      std::vector<unsigned> &EndBits, std::vector<uint64_t> &FieldVals,
      insn_t &Insn);

  // Emits code to decode the singleton.  Return true if we have matched all the
  // well-known bits.
  bool emitSingletonDecoder(raw_ostream &o, unsigned &Indentation,unsigned Opc);

  // Emits code to decode the singleton, and then to decode the rest.
  void emitSingletonDecoder(raw_ostream &o, unsigned &Indentation,Filter &Best);

  void emitBinaryParser(raw_ostream &o , unsigned &Indentation,
                        OperandInfo &OpInfo);

  // Assign a single filter and run with it.
  void runSingleFilter(FilterChooser &owner, unsigned startBit, unsigned numBit,
      bool mixed);

  // reportRegion is a helper function for filterProcessor to mark a region as
  // eligible for use as a filter region.
  void reportRegion(bitAttr_t RA, unsigned StartBit, unsigned BitIndex,
      bool AllowMixed);

  // FilterProcessor scans the well-known encoding bits of the instructions and
  // builds up a list of candidate filters.  It chooses the best filter and
  // recursively descends down the decoding tree.
  bool filterProcessor(bool AllowMixed, bool Greedy = true);

  // Decides on the best configuration of filter(s) to use in order to decode
  // the instructions.  A conflict of instructions may occur, in which case we
  // dump the conflict set to the standard error.
  void doFilter();

  // Emits code to decode our share of instructions.  Returns true if the
  // emitted code causes a return, which occurs if we know how to decode
  // the instruction at this level or the instruction is not decodeable.
  bool emit(raw_ostream &o, unsigned &Indentation);
};

///////////////////////////
//                       //
// Filter Implmenetation //
//                       //
///////////////////////////

Filter::Filter(const Filter &f) :
  Owner(f.Owner), StartBit(f.StartBit), NumBits(f.NumBits), Mixed(f.Mixed),
  FilteredInstructions(f.FilteredInstructions),
  VariableInstructions(f.VariableInstructions),
  FilterChooserMap(f.FilterChooserMap), NumFiltered(f.NumFiltered),
  LastOpcFiltered(f.LastOpcFiltered), NumVariable(f.NumVariable) {
}

Filter::Filter(FilterChooser &owner, unsigned startBit, unsigned numBits,
    bool mixed) : Owner(&owner), StartBit(startBit), NumBits(numBits),
                  Mixed(mixed) {
  assert(StartBit + NumBits - 1 < Owner->BitWidth);

  NumFiltered = 0;
  LastOpcFiltered = 0;
  NumVariable = 0;

  for (unsigned i = 0, e = Owner->Opcodes.size(); i != e; ++i) {
    insn_t Insn;

    // Populates the insn given the uid.
    Owner->insnWithID(Insn, Owner->Opcodes[i]);

    uint64_t Field;
    // Scans the segment for possibly well-specified encoding bits.
    bool ok = Owner->fieldFromInsn(Field, Insn, StartBit, NumBits);

    if (ok) {
      // The encoding bits are well-known.  Lets add the uid of the
      // instruction into the bucket keyed off the constant field value.
      LastOpcFiltered = Owner->Opcodes[i];
      FilteredInstructions[Field].push_back(LastOpcFiltered);
      ++NumFiltered;
    } else {
      // Some of the encoding bit(s) are unspecfied.  This contributes to
      // one additional member of "Variable" instructions.
      VariableInstructions.push_back(Owner->Opcodes[i]);
      ++NumVariable;
    }
  }

  assert((FilteredInstructions.size() + VariableInstructions.size() > 0)
         && "Filter returns no instruction categories");
}

Filter::~Filter() {
  std::map<unsigned, FilterChooser*>::iterator filterIterator;
  for (filterIterator = FilterChooserMap.begin();
       filterIterator != FilterChooserMap.end();
       filterIterator++) {
    delete filterIterator->second;
  }
}

// Divides the decoding task into sub tasks and delegates them to the
// inferior FilterChooser's.
//
// A special case arises when there's only one entry in the filtered
// instructions.  In order to unambiguously decode the singleton, we need to
// match the remaining undecoded encoding bits against the singleton.
void Filter::recurse() {
  std::map<uint64_t, std::vector<unsigned> >::const_iterator mapIterator;

  // Starts by inheriting our parent filter chooser's filter bit values.
  std::vector<bit_value_t> BitValueArray(Owner->FilterBitValues);

  unsigned bitIndex;

  if (VariableInstructions.size()) {
    // Conservatively marks each segment position as BIT_UNSET.
    for (bitIndex = 0; bitIndex < NumBits; bitIndex++)
      BitValueArray[StartBit + bitIndex] = BIT_UNSET;

    // Delegates to an inferior filter chooser for further processing on this
    // group of instructions whose segment values are variable.
    FilterChooserMap.insert(std::pair<unsigned, FilterChooser*>(
                              (unsigned)-1,
                              new FilterChooser(Owner->AllInstructions,
                                                VariableInstructions,
                                                Owner->Operands,
                                                BitValueArray,
                                                *Owner)
                              ));
  }

  // No need to recurse for a singleton filtered instruction.
  // See also Filter::emit().
  if (getNumFiltered() == 1) {
    //Owner->SingletonExists(LastOpcFiltered);
    assert(FilterChooserMap.size() == 1);
    return;
  }

  // Otherwise, create sub choosers.
  for (mapIterator = FilteredInstructions.begin();
       mapIterator != FilteredInstructions.end();
       mapIterator++) {

    // Marks all the segment positions with either BIT_TRUE or BIT_FALSE.
    for (bitIndex = 0; bitIndex < NumBits; bitIndex++) {
      if (mapIterator->first & (1ULL << bitIndex))
        BitValueArray[StartBit + bitIndex] = BIT_TRUE;
      else
        BitValueArray[StartBit + bitIndex] = BIT_FALSE;
    }

    // Delegates to an inferior filter chooser for further processing on this
    // category of instructions.
    FilterChooserMap.insert(std::pair<unsigned, FilterChooser*>(
                              mapIterator->first,
                              new FilterChooser(Owner->AllInstructions,
                                                mapIterator->second,
                                                Owner->Operands,
                                                BitValueArray,
                                                *Owner)
                              ));
  }
}

// Emit code to decode instructions given a segment or segments of bits.
void Filter::emit(raw_ostream &o, unsigned &Indentation) {
  o.indent(Indentation) << "// Check Inst{";

  if (NumBits > 1)
    o << (StartBit + NumBits - 1) << '-';

  o << StartBit << "} ...\n";

  o.indent(Indentation) << "switch (fieldFromInstruction" << Owner->BitWidth
                        << "(insn, " << StartBit << ", "
                        << NumBits << ")) {\n";

  std::map<unsigned, FilterChooser*>::iterator filterIterator;

  bool DefaultCase = false;
  for (filterIterator = FilterChooserMap.begin();
       filterIterator != FilterChooserMap.end();
       filterIterator++) {

    // Field value -1 implies a non-empty set of variable instructions.
    // See also recurse().
    if (filterIterator->first == (unsigned)-1) {
      DefaultCase = true;

      o.indent(Indentation) << "default:\n";
      o.indent(Indentation) << "  break; // fallthrough\n";

      // Closing curly brace for the switch statement.
      // This is unconventional because we want the default processing to be
      // performed for the fallthrough cases as well, i.e., when the "cases"
      // did not prove a decoded instruction.
      o.indent(Indentation) << "}\n";

    } else
      o.indent(Indentation) << "case " << filterIterator->first << ":\n";

    // We arrive at a category of instructions with the same segment value.
    // Now delegate to the sub filter chooser for further decodings.
    // The case may fallthrough, which happens if the remaining well-known
    // encoding bits do not match exactly.
    if (!DefaultCase) { ++Indentation; ++Indentation; }

    bool finished = filterIterator->second->emit(o, Indentation);
    // For top level default case, there's no need for a break statement.
    if (Owner->isTopLevel() && DefaultCase)
      break;
    if (!finished)
      o.indent(Indentation) << "break;\n";

    if (!DefaultCase) { --Indentation; --Indentation; }
  }

  // If there is no default case, we still need to supply a closing brace.
  if (!DefaultCase) {
    // Closing curly brace for the switch statement.
    o.indent(Indentation) << "}\n";
  }
}

// Returns the number of fanout produced by the filter.  More fanout implies
// the filter distinguishes more categories of instructions.
unsigned Filter::usefulness() const {
  if (VariableInstructions.size())
    return FilteredInstructions.size();
  else
    return FilteredInstructions.size() + 1;
}

//////////////////////////////////
//                              //
// Filterchooser Implementation //
//                              //
//////////////////////////////////

// Emit the top level typedef and decodeInstruction() function.
void FilterChooser::emitTop(raw_ostream &o, unsigned Indentation,
                            std::string Namespace) {
  o.indent(Indentation) <<
    "static bool decode" << Namespace << "Instruction" << BitWidth
    << "(MCInst &MI, uint" << BitWidth << "_t insn, uint64_t Address, "
    << "const void *Decoder) {\n";
  o.indent(Indentation) << "  unsigned tmp = 0;\n  (void)tmp;\n";

  ++Indentation; ++Indentation;
  // Emits code to decode the instructions.
  emit(o, Indentation);

  o << '\n';
  o.indent(Indentation) << "return false;\n";
  --Indentation; --Indentation;

  o.indent(Indentation) << "}\n";

  o << '\n';
}

// Populates the field of the insn given the start position and the number of
// consecutive bits to scan for.
//
// Returns false if and on the first uninitialized bit value encountered.
// Returns true, otherwise.
bool FilterChooser::fieldFromInsn(uint64_t &Field, insn_t &Insn,
    unsigned StartBit, unsigned NumBits) const {
  Field = 0;

  for (unsigned i = 0; i < NumBits; ++i) {
    if (Insn[StartBit + i] == BIT_UNSET)
      return false;

    if (Insn[StartBit + i] == BIT_TRUE)
      Field = Field | (1ULL << i);
  }

  return true;
}

/// dumpFilterArray - dumpFilterArray prints out debugging info for the given
/// filter array as a series of chars.
void FilterChooser::dumpFilterArray(raw_ostream &o,
                                    std::vector<bit_value_t> &filter) {
  unsigned bitIndex;

  for (bitIndex = BitWidth; bitIndex > 0; bitIndex--) {
    switch (filter[bitIndex - 1]) {
    case BIT_UNFILTERED:
      o << ".";
      break;
    case BIT_UNSET:
      o << "_";
      break;
    case BIT_TRUE:
      o << "1";
      break;
    case BIT_FALSE:
      o << "0";
      break;
    }
  }
}

/// dumpStack - dumpStack traverses the filter chooser chain and calls
/// dumpFilterArray on each filter chooser up to the top level one.
void FilterChooser::dumpStack(raw_ostream &o, const char *prefix) {
  FilterChooser *current = this;

  while (current) {
    o << prefix;
    dumpFilterArray(o, current->FilterBitValues);
    o << '\n';
    current = current->Parent;
  }
}

// Called from Filter::recurse() when singleton exists.  For debug purpose.
void FilterChooser::SingletonExists(unsigned Opc) {
  insn_t Insn0;
  insnWithID(Insn0, Opc);

  errs() << "Singleton exists: " << nameWithID(Opc)
         << " with its decoding dominating ";
  for (unsigned i = 0; i < Opcodes.size(); ++i) {
    if (Opcodes[i] == Opc) continue;
    errs() << nameWithID(Opcodes[i]) << ' ';
  }
  errs() << '\n';

  dumpStack(errs(), "\t\t");
  for (unsigned i = 0; i < Opcodes.size(); i++) {
    const std::string &Name = nameWithID(Opcodes[i]);

    errs() << '\t' << Name << " ";
    dumpBits(errs(),
             getBitsField(*AllInstructions[Opcodes[i]]->TheDef, "Inst"));
    errs() << '\n';
  }
}

// Calculates the island(s) needed to decode the instruction.
// This returns a list of undecoded bits of an instructions, for example,
// Inst{20} = 1 && Inst{3-0} == 0b1111 represents two islands of yet-to-be
// decoded bits in order to verify that the instruction matches the Opcode.
unsigned FilterChooser::getIslands(std::vector<unsigned> &StartBits,
    std::vector<unsigned> &EndBits, std::vector<uint64_t> &FieldVals,
    insn_t &Insn) {
  unsigned Num, BitNo;
  Num = BitNo = 0;

  uint64_t FieldVal = 0;

  // 0: Init
  // 1: Water (the bit value does not affect decoding)
  // 2: Island (well-known bit value needed for decoding)
  int State = 0;
  int Val = -1;

  for (unsigned i = 0; i < BitWidth; ++i) {
    Val = Value(Insn[i]);
    bool Filtered = PositionFiltered(i);
    switch (State) {
    default:
      assert(0 && "Unreachable code!");
      break;
    case 0:
    case 1:
      if (Filtered || Val == -1)
        State = 1; // Still in Water
      else {
        State = 2; // Into the Island
        BitNo = 0;
        StartBits.push_back(i);
        FieldVal = Val;
      }
      break;
    case 2:
      if (Filtered || Val == -1) {
        State = 1; // Into the Water
        EndBits.push_back(i - 1);
        FieldVals.push_back(FieldVal);
        ++Num;
      } else {
        State = 2; // Still in Island
        ++BitNo;
        FieldVal = FieldVal | Val << BitNo;
      }
      break;
    }
  }
  // If we are still in Island after the loop, do some housekeeping.
  if (State == 2) {
    EndBits.push_back(BitWidth - 1);
    FieldVals.push_back(FieldVal);
    ++Num;
  }

  assert(StartBits.size() == Num && EndBits.size() == Num &&
         FieldVals.size() == Num);
  return Num;
}

void FilterChooser::emitBinaryParser(raw_ostream &o, unsigned &Indentation,
                         OperandInfo &OpInfo) {
  std::string &Decoder = OpInfo.Decoder;

  if (OpInfo.numFields() == 1) {
    OperandInfo::iterator OI = OpInfo.begin();
    o.indent(Indentation) << "  tmp = fieldFromInstruction" << BitWidth
                            << "(insn, " << OI->Base << ", " << OI->Width
                            << ");\n";
  } else {
    o.indent(Indentation) << "  tmp = 0;\n";
    for (OperandInfo::iterator OI = OpInfo.begin(), OE = OpInfo.end();
         OI != OE; ++OI) {
      o.indent(Indentation) << "  tmp |= (fieldFromInstruction" << BitWidth
                            << "(insn, " << OI->Base << ", " << OI->Width 
                            << ") << " << OI->Offset << ");\n";
    }
  }

  if (Decoder != "")
    o.indent(Indentation) << "  if (!" << Decoder
                          << "(MI, tmp, Address, Decoder)) return false;\n";
  else
    o.indent(Indentation) << "  MI.addOperand(MCOperand::CreateImm(tmp));\n";

}

// Emits code to decode the singleton.  Return true if we have matched all the
// well-known bits.
bool FilterChooser::emitSingletonDecoder(raw_ostream &o, unsigned &Indentation,
                                         unsigned Opc) {
  std::vector<unsigned> StartBits;
  std::vector<unsigned> EndBits;
  std::vector<uint64_t> FieldVals;
  insn_t Insn;
  insnWithID(Insn, Opc);

  // Look for islands of undecoded bits of the singleton.
  getIslands(StartBits, EndBits, FieldVals, Insn);

  unsigned Size = StartBits.size();
  unsigned I, NumBits;

  // If we have matched all the well-known bits, just issue a return.
  if (Size == 0) {
    o.indent(Indentation) << "{\n";
    o.indent(Indentation) << "  MI.setOpcode(" << Opc << ");\n";
    std::vector<OperandInfo>& InsnOperands = Operands[Opc];
    for (std::vector<OperandInfo>::iterator
         I = InsnOperands.begin(), E = InsnOperands.end(); I != E; ++I) {
      // If a custom instruction decoder was specified, use that.
      if (I->numFields() == 0 && I->Decoder.size()) {
        o.indent(Indentation) << "  if (!" << I->Decoder
                              << "(MI, insn, Address, Decoder)) return false;\n";
        break;
      }

      emitBinaryParser(o, Indentation, *I);
    }

    o.indent(Indentation) << "  return true; // " << nameWithID(Opc)
                          << '\n';
    o.indent(Indentation) << "}\n";
    return true;
  }

  // Otherwise, there are more decodings to be done!

  // Emit code to match the island(s) for the singleton.
  o.indent(Indentation) << "// Check ";

  for (I = Size; I != 0; --I) {
    o << "Inst{" << EndBits[I-1] << '-' << StartBits[I-1] << "} ";
    if (I > 1)
      o << "&& ";
    else
      o << "for singleton decoding...\n";
  }

  o.indent(Indentation) << "if (";

  for (I = Size; I != 0; --I) {
    NumBits = EndBits[I-1] - StartBits[I-1] + 1;
    o << "fieldFromInstruction" << BitWidth << "(insn, "
      << StartBits[I-1] << ", " << NumBits
      << ") == " << FieldVals[I-1];
    if (I > 1)
      o << " && ";
    else
      o << ") {\n";
  }
  o.indent(Indentation) << "  MI.setOpcode(" << Opc << ");\n";
  std::vector<OperandInfo>& InsnOperands = Operands[Opc];
  for (std::vector<OperandInfo>::iterator
       I = InsnOperands.begin(), E = InsnOperands.end(); I != E; ++I) {
    // If a custom instruction decoder was specified, use that.
    if (I->numFields() == 0 && I->Decoder.size()) {
      o.indent(Indentation) << "  if (!" << I->Decoder
                            << "(MI, insn, Address, Decoder)) return false;\n";
      break;
    }

    emitBinaryParser(o, Indentation, *I);
  }
  o.indent(Indentation) << "  return true; // " << nameWithID(Opc)
                        << '\n';
  o.indent(Indentation) << "}\n";

  return false;
}

// Emits code to decode the singleton, and then to decode the rest.
void FilterChooser::emitSingletonDecoder(raw_ostream &o, unsigned &Indentation,
    Filter &Best) {

  unsigned Opc = Best.getSingletonOpc();

  emitSingletonDecoder(o, Indentation, Opc);

  // Emit code for the rest.
  o.indent(Indentation) << "else\n";

  Indentation += 2;
  Best.getVariableFC().emit(o, Indentation);
  Indentation -= 2;
}

// Assign a single filter and run with it.  Top level API client can initialize
// with a single filter to start the filtering process.
void FilterChooser::runSingleFilter(FilterChooser &owner, unsigned startBit,
    unsigned numBit, bool mixed) {
  Filters.clear();
  Filter F(*this, startBit, numBit, true);
  Filters.push_back(F);
  BestIndex = 0; // Sole Filter instance to choose from.
  bestFilter().recurse();
}

// reportRegion is a helper function for filterProcessor to mark a region as
// eligible for use as a filter region.
void FilterChooser::reportRegion(bitAttr_t RA, unsigned StartBit,
    unsigned BitIndex, bool AllowMixed) {
  if (RA == ATTR_MIXED && AllowMixed)
    Filters.push_back(Filter(*this, StartBit, BitIndex - StartBit, true));
  else if (RA == ATTR_ALL_SET && !AllowMixed)
    Filters.push_back(Filter(*this, StartBit, BitIndex - StartBit, false));
}

// FilterProcessor scans the well-known encoding bits of the instructions and
// builds up a list of candidate filters.  It chooses the best filter and
// recursively descends down the decoding tree.
bool FilterChooser::filterProcessor(bool AllowMixed, bool Greedy) {
  Filters.clear();
  BestIndex = -1;
  unsigned numInstructions = Opcodes.size();

  assert(numInstructions && "Filter created with no instructions");

  // No further filtering is necessary.
  if (numInstructions == 1)
    return true;

  // Heuristics.  See also doFilter()'s "Heuristics" comment when num of
  // instructions is 3.
  if (AllowMixed && !Greedy) {
    assert(numInstructions == 3);

    for (unsigned i = 0; i < Opcodes.size(); ++i) {
      std::vector<unsigned> StartBits;
      std::vector<unsigned> EndBits;
      std::vector<uint64_t> FieldVals;
      insn_t Insn;

      insnWithID(Insn, Opcodes[i]);

      // Look for islands of undecoded bits of any instruction.
      if (getIslands(StartBits, EndBits, FieldVals, Insn) > 0) {
        // Found an instruction with island(s).  Now just assign a filter.
        runSingleFilter(*this, StartBits[0], EndBits[0] - StartBits[0] + 1,
                        true);
        return true;
      }
    }
  }

  unsigned BitIndex, InsnIndex;

  // We maintain BIT_WIDTH copies of the bitAttrs automaton.
  // The automaton consumes the corresponding bit from each
  // instruction.
  //
  //   Input symbols: 0, 1, and _ (unset).
  //   States:        NONE, FILTERED, ALL_SET, ALL_UNSET, and MIXED.
  //   Initial state: NONE.
  //
  // (NONE) ------- [01] -> (ALL_SET)
  // (NONE) ------- _ ----> (ALL_UNSET)
  // (ALL_SET) ---- [01] -> (ALL_SET)
  // (ALL_SET) ---- _ ----> (MIXED)
  // (ALL_UNSET) -- [01] -> (MIXED)
  // (ALL_UNSET) -- _ ----> (ALL_UNSET)
  // (MIXED) ------ . ----> (MIXED)
  // (FILTERED)---- . ----> (FILTERED)

  std::vector<bitAttr_t> bitAttrs;

  // FILTERED bit positions provide no entropy and are not worthy of pursuing.
  // Filter::recurse() set either BIT_TRUE or BIT_FALSE for each position.
  for (BitIndex = 0; BitIndex < BitWidth; ++BitIndex)
    if (FilterBitValues[BitIndex] == BIT_TRUE ||
        FilterBitValues[BitIndex] == BIT_FALSE)
      bitAttrs.push_back(ATTR_FILTERED);
    else
      bitAttrs.push_back(ATTR_NONE);

  for (InsnIndex = 0; InsnIndex < numInstructions; ++InsnIndex) {
    insn_t insn;

    insnWithID(insn, Opcodes[InsnIndex]);

    for (BitIndex = 0; BitIndex < BitWidth; ++BitIndex) {
      switch (bitAttrs[BitIndex]) {
      case ATTR_NONE:
        if (insn[BitIndex] == BIT_UNSET)
          bitAttrs[BitIndex] = ATTR_ALL_UNSET;
        else
          bitAttrs[BitIndex] = ATTR_ALL_SET;
        break;
      case ATTR_ALL_SET:
        if (insn[BitIndex] == BIT_UNSET)
          bitAttrs[BitIndex] = ATTR_MIXED;
        break;
      case ATTR_ALL_UNSET:
        if (insn[BitIndex] != BIT_UNSET)
          bitAttrs[BitIndex] = ATTR_MIXED;
        break;
      case ATTR_MIXED:
      case ATTR_FILTERED:
        break;
      }
    }
  }

  // The regionAttr automaton consumes the bitAttrs automatons' state,
  // lowest-to-highest.
  //
  //   Input symbols: F(iltered), (all_)S(et), (all_)U(nset), M(ixed)
  //   States:        NONE, ALL_SET, MIXED
  //   Initial state: NONE
  //
  // (NONE) ----- F --> (NONE)
  // (NONE) ----- S --> (ALL_SET)     ; and set region start
  // (NONE) ----- U --> (NONE)
  // (NONE) ----- M --> (MIXED)       ; and set region start
  // (ALL_SET) -- F --> (NONE)        ; and report an ALL_SET region
  // (ALL_SET) -- S --> (ALL_SET)
  // (ALL_SET) -- U --> (NONE)        ; and report an ALL_SET region
  // (ALL_SET) -- M --> (MIXED)       ; and report an ALL_SET region
  // (MIXED) ---- F --> (NONE)        ; and report a MIXED region
  // (MIXED) ---- S --> (ALL_SET)     ; and report a MIXED region
  // (MIXED) ---- U --> (NONE)        ; and report a MIXED region
  // (MIXED) ---- M --> (MIXED)

  bitAttr_t RA = ATTR_NONE;
  unsigned StartBit = 0;

  for (BitIndex = 0; BitIndex < BitWidth; BitIndex++) {
    bitAttr_t bitAttr = bitAttrs[BitIndex];

    assert(bitAttr != ATTR_NONE && "Bit without attributes");

    switch (RA) {
    case ATTR_NONE:
      switch (bitAttr) {
      case ATTR_FILTERED:
        break;
      case ATTR_ALL_SET:
        StartBit = BitIndex;
        RA = ATTR_ALL_SET;
        break;
      case ATTR_ALL_UNSET:
        break;
      case ATTR_MIXED:
        StartBit = BitIndex;
        RA = ATTR_MIXED;
        break;
      default:
        assert(0 && "Unexpected bitAttr!");
      }
      break;
    case ATTR_ALL_SET:
      switch (bitAttr) {
      case ATTR_FILTERED:
        reportRegion(RA, StartBit, BitIndex, AllowMixed);
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        break;
      case ATTR_ALL_UNSET:
        reportRegion(RA, StartBit, BitIndex, AllowMixed);
        RA = ATTR_NONE;
        break;
      case ATTR_MIXED:
        reportRegion(RA, StartBit, BitIndex, AllowMixed);
        StartBit = BitIndex;
        RA = ATTR_MIXED;
        break;
      default:
        assert(0 && "Unexpected bitAttr!");
      }
      break;
    case ATTR_MIXED:
      switch (bitAttr) {
      case ATTR_FILTERED:
        reportRegion(RA, StartBit, BitIndex, AllowMixed);
        StartBit = BitIndex;
        RA = ATTR_NONE;
        break;
      case ATTR_ALL_SET:
        reportRegion(RA, StartBit, BitIndex, AllowMixed);
        StartBit = BitIndex;
        RA = ATTR_ALL_SET;
        break;
      case ATTR_ALL_UNSET:
        reportRegion(RA, StartBit, BitIndex, AllowMixed);
        RA = ATTR_NONE;
        break;
      case ATTR_MIXED:
        break;
      default:
        assert(0 && "Unexpected bitAttr!");
      }
      break;
    case ATTR_ALL_UNSET:
      assert(0 && "regionAttr state machine has no ATTR_UNSET state");
    case ATTR_FILTERED:
      assert(0 && "regionAttr state machine has no ATTR_FILTERED state");
    }
  }

  // At the end, if we're still in ALL_SET or MIXED states, report a region
  switch (RA) {
  case ATTR_NONE:
    break;
  case ATTR_FILTERED:
    break;
  case ATTR_ALL_SET:
    reportRegion(RA, StartBit, BitIndex, AllowMixed);
    break;
  case ATTR_ALL_UNSET:
    break;
  case ATTR_MIXED:
    reportRegion(RA, StartBit, BitIndex, AllowMixed);
    break;
  }

  // We have finished with the filter processings.  Now it's time to choose
  // the best performing filter.
  BestIndex = 0;
  bool AllUseless = true;
  unsigned BestScore = 0;

  for (unsigned i = 0, e = Filters.size(); i != e; ++i) {
    unsigned Usefulness = Filters[i].usefulness();

    if (Usefulness)
      AllUseless = false;

    if (Usefulness > BestScore) {
      BestIndex = i;
      BestScore = Usefulness;
    }
  }

  if (!AllUseless)
    bestFilter().recurse();

  return !AllUseless;
} // end of FilterChooser::filterProcessor(bool)

// Decides on the best configuration of filter(s) to use in order to decode
// the instructions.  A conflict of instructions may occur, in which case we
// dump the conflict set to the standard error.
void FilterChooser::doFilter() {
  unsigned Num = Opcodes.size();
  assert(Num && "FilterChooser created with no instructions");

  // Try regions of consecutive known bit values first.
  if (filterProcessor(false))
    return;

  // Then regions of mixed bits (both known and unitialized bit values allowed).
  if (filterProcessor(true))
    return;

  // Heuristics to cope with conflict set {t2CMPrs, t2SUBSrr, t2SUBSrs} where
  // no single instruction for the maximum ATTR_MIXED region Inst{14-4} has a
  // well-known encoding pattern.  In such case, we backtrack and scan for the
  // the very first consecutive ATTR_ALL_SET region and assign a filter to it.
  if (Num == 3 && filterProcessor(true, false))
    return;

  // If we come to here, the instruction decoding has failed.
  // Set the BestIndex to -1 to indicate so.
  BestIndex = -1;
}

// Emits code to decode our share of instructions.  Returns true if the
// emitted code causes a return, which occurs if we know how to decode
// the instruction at this level or the instruction is not decodeable.
bool FilterChooser::emit(raw_ostream &o, unsigned &Indentation) {
  if (Opcodes.size() == 1)
    // There is only one instruction in the set, which is great!
    // Call emitSingletonDecoder() to see whether there are any remaining
    // encodings bits.
    return emitSingletonDecoder(o, Indentation, Opcodes[0]);

  // Choose the best filter to do the decodings!
  if (BestIndex != -1) {
    Filter &Best = bestFilter();
    if (Best.getNumFiltered() == 1)
      emitSingletonDecoder(o, Indentation, Best);
    else
      bestFilter().emit(o, Indentation);
    return false;
  }

  // We don't know how to decode these instructions!  Return 0 and dump the
  // conflict set!
  o.indent(Indentation) << "return 0;" << " // Conflict set: ";
  for (int i = 0, N = Opcodes.size(); i < N; ++i) {
    o << nameWithID(Opcodes[i]);
    if (i < (N - 1))
      o << ", ";
    else
      o << '\n';
  }

  // Print out useful conflict information for postmortem analysis.
  errs() << "Decoding Conflict:\n";

  dumpStack(errs(), "\t\t");

  for (unsigned i = 0; i < Opcodes.size(); i++) {
    const std::string &Name = nameWithID(Opcodes[i]);

    errs() << '\t' << Name << " ";
    dumpBits(errs(),
             getBitsField(*AllInstructions[Opcodes[i]]->TheDef, "Inst"));
    errs() << '\n';
  }

  return true;
}

static bool populateInstruction(const CodeGenInstruction &CGI,
                                unsigned Opc,
                      std::map<unsigned, std::vector<OperandInfo> >& Operands){
  const Record &Def = *CGI.TheDef;
  // If all the bit positions are not specified; do not decode this instruction.
  // We are bound to fail!  For proper disassembly, the well-known encoding bits
  // of the instruction must be fully specified.
  //
  // This also removes pseudo instructions from considerations of disassembly,
  // which is a better design and less fragile than the name matchings.
  // Ignore "asm parser only" instructions.
  if (Def.getValueAsBit("isAsmParserOnly") ||
      Def.getValueAsBit("isCodeGenOnly"))
    return false;

  BitsInit &Bits = getBitsField(Def, "Inst");
  if (Bits.allInComplete()) return false;

  std::vector<OperandInfo> InsnOperands;

  // If the instruction has specified a custom decoding hook, use that instead
  // of trying to auto-generate the decoder.
  std::string InstDecoder = Def.getValueAsString("DecoderMethod");
  if (InstDecoder != "") {
    InsnOperands.push_back(OperandInfo(InstDecoder));
    Operands[Opc] = InsnOperands;
    return true;
  }

  // Generate a description of the operand of the instruction that we know
  // how to decode automatically.
  // FIXME: We'll need to have a way to manually override this as needed.

  // Gather the outputs/inputs of the instruction, so we can find their
  // positions in the encoding.  This assumes for now that they appear in the
  // MCInst in the order that they're listed.
  std::vector<std::pair<Init*, std::string> > InOutOperands;
  DagInit *Out  = Def.getValueAsDag("OutOperandList");
  DagInit *In  = Def.getValueAsDag("InOperandList");
  for (unsigned i = 0; i < Out->getNumArgs(); ++i)
    InOutOperands.push_back(std::make_pair(Out->getArg(i), Out->getArgName(i)));
  for (unsigned i = 0; i < In->getNumArgs(); ++i)
    InOutOperands.push_back(std::make_pair(In->getArg(i), In->getArgName(i)));

  // Search for tied operands, so that we can correctly instantiate
  // operands that are not explicitly represented in the encoding.
  std::map<std::string, std::string> TiedNames;
  for (unsigned i = 0; i < CGI.Operands.size(); ++i) {
    int tiedTo = CGI.Operands[i].getTiedRegister();
    if (tiedTo != -1) {
      TiedNames[InOutOperands[i].second] = InOutOperands[tiedTo].second;
      TiedNames[InOutOperands[tiedTo].second] = InOutOperands[i].second;
    }
  }

  // For each operand, see if we can figure out where it is encoded.
  for (std::vector<std::pair<Init*, std::string> >::iterator
       NI = InOutOperands.begin(), NE = InOutOperands.end(); NI != NE; ++NI) {
    std::string Decoder = "";

    // At this point, we can locate the field, but we need to know how to
    // interpret it.  As a first step, require the target to provide callbacks
    // for decoding register classes.
    // FIXME: This need to be extended to handle instructions with custom
    // decoder methods, and operands with (simple) MIOperandInfo's.
    TypedInit *TI = dynamic_cast<TypedInit*>(NI->first);
    RecordRecTy *Type = dynamic_cast<RecordRecTy*>(TI->getType());
    Record *TypeRecord = Type->getRecord();
    bool isReg = false;
    if (TypeRecord->isSubClassOf("RegisterOperand"))
      TypeRecord = TypeRecord->getValueAsDef("RegClass");
    if (TypeRecord->isSubClassOf("RegisterClass")) {
      Decoder = "Decode" + TypeRecord->getName() + "RegisterClass";
      isReg = true;
    }

    RecordVal *DecoderString = TypeRecord->getValue("DecoderMethod");
    StringInit *String = DecoderString ?
      dynamic_cast<StringInit*>(DecoderString->getValue()) : 0;
    if (!isReg && String && String->getValue() != "")
      Decoder = String->getValue();

    OperandInfo OpInfo(Decoder);
    unsigned Base = ~0U;
    unsigned Width = 0;
    unsigned Offset = 0;

    for (unsigned bi = 0; bi < Bits.getNumBits(); ++bi) {
      VarInit *Var = 0;
      VarBitInit *BI = dynamic_cast<VarBitInit*>(Bits.getBit(bi));
      if (BI)
        Var = dynamic_cast<VarInit*>(BI->getVariable());
      else
        Var = dynamic_cast<VarInit*>(Bits.getBit(bi));

      if (!Var) {
        if (Base != ~0U) {
          OpInfo.addField(Base, Width, Offset);
          Base = ~0U;
          Width = 0;
          Offset = 0;
        }
        continue;
      }

      if (Var->getName() != NI->second &&
          Var->getName() != TiedNames[NI->second]) {
        if (Base != ~0U) {
          OpInfo.addField(Base, Width, Offset);
          Base = ~0U;
          Width = 0;
          Offset = 0;
        }
        continue;
      }

      if (Base == ~0U) {
        Base = bi;
        Width = 1;
        Offset = BI ? BI->getBitNum() : 0;
      } else if (BI && BI->getBitNum() != Offset + Width) {
        OpInfo.addField(Base, Width, Offset);
        Base = bi;
        Width = 1;
        Offset = BI->getBitNum();
      } else {
        ++Width;
      }
    }

    if (Base != ~0U)
      OpInfo.addField(Base, Width, Offset);

    if (OpInfo.numFields() > 0)
      InsnOperands.push_back(OpInfo);
  }

  Operands[Opc] = InsnOperands;


#if 0
  DEBUG({
      // Dumps the instruction encoding bits.
      dumpBits(errs(), Bits);

      errs() << '\n';

      // Dumps the list of operand info.
      for (unsigned i = 0, e = CGI.Operands.size(); i != e; ++i) {
        const CGIOperandList::OperandInfo &Info = CGI.Operands[i];
        const std::string &OperandName = Info.Name;
        const Record &OperandDef = *Info.Rec;

        errs() << "\t" << OperandName << " (" << OperandDef.getName() << ")\n";
      }
    });
#endif

  return true;
}

static void emitHelper(llvm::raw_ostream &o, unsigned BitWidth) {
  unsigned Indentation = 0;
  std::string WidthStr = "uint" + utostr(BitWidth) + "_t";

  o << '\n';

  o.indent(Indentation) << "static " << WidthStr <<
    " fieldFromInstruction" << BitWidth <<
    "(" << WidthStr <<" insn, unsigned startBit, unsigned numBits)\n";

  o.indent(Indentation) << "{\n";

  ++Indentation; ++Indentation;
  o.indent(Indentation) << "assert(startBit + numBits <= " << BitWidth
                        << " && \"Instruction field out of bounds!\");\n";
  o << '\n';
  o.indent(Indentation) << WidthStr << " fieldMask;\n";
  o << '\n';
  o.indent(Indentation) << "if (numBits == " << BitWidth << ")\n";

  ++Indentation; ++Indentation;
  o.indent(Indentation) << "fieldMask = (" << WidthStr << ")-1;\n";
  --Indentation; --Indentation;

  o.indent(Indentation) << "else\n";

  ++Indentation; ++Indentation;
  o.indent(Indentation) << "fieldMask = ((1 << numBits) - 1) << startBit;\n";
  --Indentation; --Indentation;

  o << '\n';
  o.indent(Indentation) << "return (insn & fieldMask) >> startBit;\n";
  --Indentation; --Indentation;

  o.indent(Indentation) << "}\n";

  o << '\n';
}

// Emits disassembler code for instruction decoding.
void FixedLenDecoderEmitter::run(raw_ostream &o)
{
  o << "#include \"llvm/MC/MCInst.h\"\n";
  o << "#include \"llvm/Support/DataTypes.h\"\n";
  o << "#include <assert.h>\n";
  o << '\n';
  o << "namespace llvm {\n\n";

  // Parameterize the decoders based on namespace and instruction width.
  NumberedInstructions = Target.getInstructionsByEnumValue();
  std::map<std::pair<std::string, unsigned>,
           std::vector<unsigned> > OpcMap;
  std::map<unsigned, std::vector<OperandInfo> > Operands;

  for (unsigned i = 0; i < NumberedInstructions.size(); ++i) {
    const CodeGenInstruction *Inst = NumberedInstructions[i];
    Record *Def = Inst->TheDef;
    unsigned Size = Def->getValueAsInt("Size");
    if (Def->getValueAsString("Namespace") == "TargetOpcode" ||
        Def->getValueAsBit("isPseudo") ||
        Def->getValueAsBit("isAsmParserOnly") ||
        Def->getValueAsBit("isCodeGenOnly"))
      continue;

    std::string DecoderNamespace = Def->getValueAsString("DecoderNamespace");

    if (Size) {
      if (populateInstruction(*Inst, i, Operands)) {
        OpcMap[std::make_pair(DecoderNamespace, Size)].push_back(i);
      }
    }
  }

  std::set<unsigned> Sizes;
  for (std::map<std::pair<std::string, unsigned>,
                std::vector<unsigned> >::iterator
       I = OpcMap.begin(), E = OpcMap.end(); I != E; ++I) {
    // If we haven't visited this instruction width before, emit the
    // helper method to extract fields.
    if (!Sizes.count(I->first.second)) {
      emitHelper(o, 8*I->first.second);
      Sizes.insert(I->first.second);
    }

    // Emit the decoder for this namespace+width combination.
    FilterChooser FC(NumberedInstructions, I->second, Operands,
                     8*I->first.second);
    FC.emitTop(o, 0, I->first.first);
  }

  o << "\n} // End llvm namespace \n";
}
