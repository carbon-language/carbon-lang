//===------------ ARMDecoderEmitter.cpp - Decoder Generator ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the ARM Disassembler.
// It contains the tablegen backend that emits the decoder functions for ARM and
// Thumb.  The disassembler core includes the auto-generated file, invokes the
// decoder functions, and builds up the MCInst based on the decoded Opcode.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-decoder-emitter"

#include "ARMDecoderEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <map>
#include <string>

using namespace llvm;

/////////////////////////////////////////////////////
//                                                 //
//  Enums and Utilities for ARM Instruction Format //
//                                                 //
/////////////////////////////////////////////////////

#define ARM_FORMATS                   \
  ENTRY(ARM_FORMAT_PSEUDO,         0) \
  ENTRY(ARM_FORMAT_MULFRM,         1) \
  ENTRY(ARM_FORMAT_BRFRM,          2) \
  ENTRY(ARM_FORMAT_BRMISCFRM,      3) \
  ENTRY(ARM_FORMAT_DPFRM,          4) \
  ENTRY(ARM_FORMAT_DPSOREGFRM,     5) \
  ENTRY(ARM_FORMAT_LDFRM,          6) \
  ENTRY(ARM_FORMAT_STFRM,          7) \
  ENTRY(ARM_FORMAT_LDMISCFRM,      8) \
  ENTRY(ARM_FORMAT_STMISCFRM,      9) \
  ENTRY(ARM_FORMAT_LDSTMULFRM,    10) \
  ENTRY(ARM_FORMAT_LDSTEXFRM,     11) \
  ENTRY(ARM_FORMAT_ARITHMISCFRM,  12) \
  ENTRY(ARM_FORMAT_SATFRM,        13) \
  ENTRY(ARM_FORMAT_EXTFRM,        14) \
  ENTRY(ARM_FORMAT_VFPUNARYFRM,   15) \
  ENTRY(ARM_FORMAT_VFPBINARYFRM,  16) \
  ENTRY(ARM_FORMAT_VFPCONV1FRM,   17) \
  ENTRY(ARM_FORMAT_VFPCONV2FRM,   18) \
  ENTRY(ARM_FORMAT_VFPCONV3FRM,   19) \
  ENTRY(ARM_FORMAT_VFPCONV4FRM,   20) \
  ENTRY(ARM_FORMAT_VFPCONV5FRM,   21) \
  ENTRY(ARM_FORMAT_VFPLDSTFRM,    22) \
  ENTRY(ARM_FORMAT_VFPLDSTMULFRM, 23) \
  ENTRY(ARM_FORMAT_VFPMISCFRM,    24) \
  ENTRY(ARM_FORMAT_THUMBFRM,      25) \
  ENTRY(ARM_FORMAT_MISCFRM,       26) \
  ENTRY(ARM_FORMAT_NEONGETLNFRM,  27) \
  ENTRY(ARM_FORMAT_NEONSETLNFRM,  28) \
  ENTRY(ARM_FORMAT_NEONDUPFRM,    29) \
  ENTRY(ARM_FORMAT_NLdSt,         30) \
  ENTRY(ARM_FORMAT_N1RegModImm,   31) \
  ENTRY(ARM_FORMAT_N2Reg,         32) \
  ENTRY(ARM_FORMAT_NVCVT,         33) \
  ENTRY(ARM_FORMAT_NVecDupLn,     34) \
  ENTRY(ARM_FORMAT_N2RegVecShL,   35) \
  ENTRY(ARM_FORMAT_N2RegVecShR,   36) \
  ENTRY(ARM_FORMAT_N3Reg,         37) \
  ENTRY(ARM_FORMAT_N3RegVecSh,    38) \
  ENTRY(ARM_FORMAT_NVecExtract,   39) \
  ENTRY(ARM_FORMAT_NVecMulScalar, 40) \
  ENTRY(ARM_FORMAT_NVTBL,         41)

// ARM instruction format specifies the encoding used by the instruction.
#define ENTRY(n, v) n = v,
typedef enum {
  ARM_FORMATS
  ARM_FORMAT_NA
} ARMFormat;
#undef ENTRY

// Converts enum to const char*.
static const char *stringForARMFormat(ARMFormat form) {
#define ENTRY(n, v) case n: return #n;
  switch(form) {
    ARM_FORMATS
  case ARM_FORMAT_NA:
  default:
    return "";
  }
#undef ENTRY
}

enum {
  IndexModeNone = 0,
  IndexModePre  = 1,
  IndexModePost = 2,
  IndexModeUpd  = 3
};

/////////////////////////
//                     //
//  Utility functions  //
//                     //
/////////////////////////

/// byteFromBitsInit - Return the byte value from a BitsInit.
/// Called from getByteField().
static uint8_t byteFromBitsInit(BitsInit &init) {
  int width = init.getNumBits();

  assert(width <= 8 && "Field is too large for uint8_t!");

  int index;
  uint8_t mask = 0x01;

  uint8_t ret = 0;

  for (index = 0; index < width; index++) {
    if (static_cast<BitInit*>(init.getBit(index))->getValue())
      ret |= mask;

    mask <<= 1;
  }

  return ret;
}

static uint8_t getByteField(const Record &def, const char *str) {
  BitsInit *bits = def.getValueAsBitsInit(str);
  return byteFromBitsInit(*bits);
}

static BitsInit &getBitsField(const Record &def, const char *str) {
  BitsInit *bits = def.getValueAsBitsInit(str);
  return *bits;
}

/// sameStringExceptSuffix - Return true if the two strings differ only in RHS's
/// suffix.  ("VST4d8", "VST4d8_UPD", "_UPD") as input returns true.
static
bool sameStringExceptSuffix(const StringRef LHS, const StringRef RHS,
                            const StringRef Suffix) {

  if (RHS.startswith(LHS) && RHS.endswith(Suffix))
    return RHS.size() == LHS.size() + Suffix.size();

  return false;
}

/// thumbInstruction - Determine whether we have a Thumb instruction.
/// See also ARMInstrFormats.td.
static bool thumbInstruction(uint8_t Form) {
  return Form == ARM_FORMAT_THUMBFRM;
}

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

// Enums for the available target names.
typedef enum {
  TARGET_ARM = 0,
  TARGET_THUMB
} TARGET_NAME_t;

// FIXME: Possibly auto-detected?
#define BIT_WIDTH 32

// Forward declaration.
class ARMFilterChooser;

// Representation of the instruction to work on.
typedef bit_value_t insn_t[BIT_WIDTH];

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
class ARMFilter {
protected:
  ARMFilterChooser *Owner; // points to the FilterChooser who owns this filter
  unsigned StartBit; // the starting bit position
  unsigned NumBits; // number of bits to filter
  bool Mixed; // a mixed region contains both set and unset bits

  // Map of well-known segment value to the set of uid's with that value. 
  std::map<uint64_t, std::vector<unsigned> > FilteredInstructions;

  // Set of uid's with non-constant segment values.
  std::vector<unsigned> VariableInstructions;

  // Map of well-known segment value to its delegate.
  std::map<unsigned, ARMFilterChooser*> FilterChooserMap;

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
  ARMFilterChooser &getVariableFC() {
    assert(NumFiltered == 1);
    assert(FilterChooserMap.size() == 1);
    return *(FilterChooserMap.find((unsigned)-1)->second);
  }

  ARMFilter(const ARMFilter &f);
  ARMFilter(ARMFilterChooser &owner, unsigned startBit, unsigned numBits,
            bool mixed);

  ~ARMFilter();

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

/// ARMFilterChooser - FilterChooser chooses the best filter among a set of Filters
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
class ARMFilterChooser {
  static TARGET_NAME_t TargetName;

protected:
  friend class ARMFilter;

  // Vector of codegen instructions to choose our filter.
  const std::vector<const CodeGenInstruction*> &AllInstructions;

  // Vector of uid's for this filter chooser to work on.
  const std::vector<unsigned> Opcodes;

  // Vector of candidate filters.
  std::vector<ARMFilter> Filters;

  // Array of bit values passed down from our parent.
  // Set to all BIT_UNFILTERED's for Parent == NULL.
  bit_value_t FilterBitValues[BIT_WIDTH];

  // Links to the FilterChooser above us in the decoding tree.
  ARMFilterChooser *Parent;
  
  // Index of the best filter from Filters.
  int BestIndex;

public:
  static void setTargetName(TARGET_NAME_t tn) { TargetName = tn; }

  ARMFilterChooser(const ARMFilterChooser &FC) :
      AllInstructions(FC.AllInstructions), Opcodes(FC.Opcodes),
      Filters(FC.Filters), Parent(FC.Parent), BestIndex(FC.BestIndex) {
    memcpy(FilterBitValues, FC.FilterBitValues, sizeof(FilterBitValues));
  }

  ARMFilterChooser(const std::vector<const CodeGenInstruction*> &Insts,
                const std::vector<unsigned> &IDs) :
      AllInstructions(Insts), Opcodes(IDs), Filters(), Parent(NULL),
      BestIndex(-1) {
    for (unsigned i = 0; i < BIT_WIDTH; ++i)
      FilterBitValues[i] = BIT_UNFILTERED;

    doFilter();
  }

  ARMFilterChooser(const std::vector<const CodeGenInstruction*> &Insts,
                   const std::vector<unsigned> &IDs,
                   bit_value_t (&ParentFilterBitValues)[BIT_WIDTH],
                   ARMFilterChooser &parent) :
      AllInstructions(Insts), Opcodes(IDs), Filters(), Parent(&parent),
      BestIndex(-1) {
    for (unsigned i = 0; i < BIT_WIDTH; ++i)
      FilterBitValues[i] = ParentFilterBitValues[i];

    doFilter();
  }

  // The top level filter chooser has NULL as its parent.
  bool isTopLevel() { return Parent == NULL; }

  // This provides an opportunity for target specific code emission.
  void emitTopHook(raw_ostream &o);

  // Emit the top level typedef and decodeInstruction() function.
  void emitTop(raw_ostream &o, unsigned &Indentation);

  // This provides an opportunity for target specific code emission after
  // emitTop().
  void emitBot(raw_ostream &o, unsigned &Indentation);

protected:
  // Populates the insn given the uid.
  void insnWithID(insn_t &Insn, unsigned Opcode) const {
    BitsInit &Bits = getBitsField(*AllInstructions[Opcode]->TheDef, "Inst");

    for (unsigned i = 0; i < BIT_WIDTH; ++i)
      Insn[i] = bitFromBits(Bits, i);

    // Set Inst{21} to 1 (wback) when IndexModeBits == IndexModeUpd.
    Record *R = AllInstructions[Opcode]->TheDef;
    if (R->getValue("IndexModeBits") &&
        getByteField(*R, "IndexModeBits") == IndexModeUpd)
      Insn[21] = BIT_TRUE;
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
  void dumpFilterArray(raw_ostream &o, bit_value_t (&filter)[BIT_WIDTH]);

  /// dumpStack - dumpStack traverses the filter chooser chain and calls
  /// dumpFilterArray on each filter chooser up to the top level one.
  void dumpStack(raw_ostream &o, const char *prefix);

  ARMFilter &bestFilter() {
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

  // The purpose of this function is for the API client to detect possible
  // Load/Store Coprocessor instructions.  If the coprocessor number is of
  // the instruction is either 10 or 11, the decoder should not report the
  // instruction as LDC/LDC2/STC/STC2, but should match against Advanced SIMD or
  // VFP instructions.
  bool LdStCopEncoding1(unsigned Opc) {
    const std::string &Name = nameWithID(Opc);
    if (Name == "LDC_OFFSET" || Name == "LDC_OPTION" ||
        Name == "LDC_POST" || Name == "LDC_PRE" ||
        Name == "LDCL_OFFSET" || Name == "LDCL_OPTION" ||
        Name == "LDCL_POST" || Name == "LDCL_PRE" ||
        Name == "STC_OFFSET" || Name == "STC_OPTION" ||
        Name == "STC_POST" || Name == "STC_PRE" ||
        Name == "STCL_OFFSET" || Name == "STCL_OPTION" ||
        Name == "STCL_POST" || Name == "STCL_PRE")
      return true;
    else
      return false;
  }

  // Emits code to decode the singleton.  Return true if we have matched all the
  // well-known bits.
  bool emitSingletonDecoder(raw_ostream &o, unsigned &Indentation,unsigned Opc);

  // Emits code to decode the singleton, and then to decode the rest.
  void emitSingletonDecoder(raw_ostream &o, unsigned &Indentation,
                            ARMFilter &Best);

  // Assign a single filter and run with it.
  void runSingleFilter(ARMFilterChooser &owner, unsigned startBit,
                       unsigned numBit, bool mixed);

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

ARMFilter::ARMFilter(const ARMFilter &f) :
  Owner(f.Owner), StartBit(f.StartBit), NumBits(f.NumBits), Mixed(f.Mixed),
  FilteredInstructions(f.FilteredInstructions),
  VariableInstructions(f.VariableInstructions),
  FilterChooserMap(f.FilterChooserMap), NumFiltered(f.NumFiltered),
  LastOpcFiltered(f.LastOpcFiltered), NumVariable(f.NumVariable) {
}

ARMFilter::ARMFilter(ARMFilterChooser &owner, unsigned startBit, unsigned numBits,
    bool mixed) : Owner(&owner), StartBit(startBit), NumBits(numBits),
                  Mixed(mixed) {
  assert(StartBit + NumBits - 1 < BIT_WIDTH);

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

ARMFilter::~ARMFilter() {
  std::map<unsigned, ARMFilterChooser*>::iterator filterIterator;
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
void ARMFilter::recurse() {
  std::map<uint64_t, std::vector<unsigned> >::const_iterator mapIterator;

  bit_value_t BitValueArray[BIT_WIDTH];
  // Starts by inheriting our parent filter chooser's filter bit values.
  memcpy(BitValueArray, Owner->FilterBitValues, sizeof(BitValueArray));

  unsigned bitIndex;

  if (VariableInstructions.size()) {
    // Conservatively marks each segment position as BIT_UNSET.
    for (bitIndex = 0; bitIndex < NumBits; bitIndex++)
      BitValueArray[StartBit + bitIndex] = BIT_UNSET;

    // Delegates to an inferior filter chooser for further processing on this
    // group of instructions whose segment values are variable.
    FilterChooserMap.insert(std::pair<unsigned, ARMFilterChooser*>(
                              (unsigned)-1,
                              new ARMFilterChooser(Owner->AllInstructions,
                                                   VariableInstructions,
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
    FilterChooserMap.insert(std::pair<unsigned, ARMFilterChooser*>(
                              mapIterator->first,
                              new ARMFilterChooser(Owner->AllInstructions,
                                                   mapIterator->second,
                                                   BitValueArray,
                                                   *Owner)
                              ));
  }
}

// Emit code to decode instructions given a segment or segments of bits.
void ARMFilter::emit(raw_ostream &o, unsigned &Indentation) {
  o.indent(Indentation) << "// Check Inst{";

  if (NumBits > 1)
    o << (StartBit + NumBits - 1) << '-';

  o << StartBit << "} ...\n";

  o.indent(Indentation) << "switch (fieldFromInstruction(insn, "
                        << StartBit << ", " << NumBits << ")) {\n";

  std::map<unsigned, ARMFilterChooser*>::iterator filterIterator;

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
unsigned ARMFilter::usefulness() const {
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

// Define the symbol here.
TARGET_NAME_t ARMFilterChooser::TargetName;

// This provides an opportunity for target specific code emission.
void ARMFilterChooser::emitTopHook(raw_ostream &o) {
  if (TargetName == TARGET_ARM) {
    // Emit code that references the ARMFormat data type.
    o << "static const ARMFormat ARMFormats[] = {\n";
    for (unsigned i = 0, e = AllInstructions.size(); i != e; ++i) {
      const Record &Def = *(AllInstructions[i]->TheDef);
      const std::string &Name = Def.getName();
      if (Def.isSubClassOf("InstARM") || Def.isSubClassOf("InstThumb"))
        o.indent(2) << 
          stringForARMFormat((ARMFormat)getByteField(Def, "Form"));
      else
        o << "  ARM_FORMAT_NA";

      o << ",\t// Inst #" << i << " = " << Name << '\n';
    }
    o << "  ARM_FORMAT_NA\t// Unreachable.\n";
    o << "};\n\n";
  }
}

// Emit the top level typedef and decodeInstruction() function.
void ARMFilterChooser::emitTop(raw_ostream &o, unsigned &Indentation) {
  // Run the target specific emit hook.
  emitTopHook(o);

  switch (BIT_WIDTH) {
  case 8:
    o.indent(Indentation) << "typedef uint8_t field_t;\n";
    break;
  case 16:
    o.indent(Indentation) << "typedef uint16_t field_t;\n";
    break;
  case 32:
    o.indent(Indentation) << "typedef uint32_t field_t;\n";
    break;
  case 64:
    o.indent(Indentation) << "typedef uint64_t field_t;\n";
    break;
  default:
    assert(0 && "Unexpected instruction size!");
  }

  o << '\n';

  o.indent(Indentation) << "static field_t " <<
    "fieldFromInstruction(field_t insn, unsigned startBit, unsigned numBits)\n";

  o.indent(Indentation) << "{\n";

  ++Indentation; ++Indentation;
  o.indent(Indentation) << "assert(startBit + numBits <= " << BIT_WIDTH
                        << " && \"Instruction field out of bounds!\");\n";
  o << '\n';
  o.indent(Indentation) << "field_t fieldMask;\n";
  o << '\n';
  o.indent(Indentation) << "if (numBits == " << BIT_WIDTH << ")\n";

  ++Indentation; ++Indentation;
  o.indent(Indentation) << "fieldMask = (field_t)-1;\n";
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

  o.indent(Indentation) <<"static uint16_t decodeInstruction(field_t insn) {\n";

  ++Indentation; ++Indentation;
  // Emits code to decode the instructions.
  emit(o, Indentation);

  o << '\n';
  o.indent(Indentation) << "return 0;\n";
  --Indentation; --Indentation;

  o.indent(Indentation) << "}\n";

  o << '\n';
}

// This provides an opportunity for target specific code emission after
// emitTop().
void ARMFilterChooser::emitBot(raw_ostream &o, unsigned &Indentation) {
  if (TargetName != TARGET_THUMB) return;

  // Emit code that decodes the Thumb ISA.
  o.indent(Indentation)
    << "static uint16_t decodeThumbInstruction(field_t insn) {\n";

  ++Indentation; ++Indentation;

  // Emits code to decode the instructions.
  emit(o, Indentation);

  o << '\n';
  o.indent(Indentation) << "return 0;\n";

  --Indentation; --Indentation;

  o.indent(Indentation) << "}\n";
}

// Populates the field of the insn given the start position and the number of
// consecutive bits to scan for.
//
// Returns false if and on the first uninitialized bit value encountered.
// Returns true, otherwise.
bool ARMFilterChooser::fieldFromInsn(uint64_t &Field, insn_t &Insn,
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
void ARMFilterChooser::dumpFilterArray(raw_ostream &o,
    bit_value_t (&filter)[BIT_WIDTH]) {
  unsigned bitIndex;

  for (bitIndex = BIT_WIDTH; bitIndex > 0; bitIndex--) {
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
void ARMFilterChooser::dumpStack(raw_ostream &o, const char *prefix) {
  ARMFilterChooser *current = this;

  while (current) {
    o << prefix;
    dumpFilterArray(o, current->FilterBitValues);
    o << '\n';
    current = current->Parent;
  }
}

// Called from Filter::recurse() when singleton exists.  For debug purpose.
void ARMFilterChooser::SingletonExists(unsigned Opc) {
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
unsigned ARMFilterChooser::getIslands(std::vector<unsigned> &StartBits,
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

  for (unsigned i = 0; i < BIT_WIDTH; ++i) {
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
    EndBits.push_back(BIT_WIDTH - 1);
    FieldVals.push_back(FieldVal);
    ++Num;
  }

  assert(StartBits.size() == Num && EndBits.size() == Num &&
         FieldVals.size() == Num);
  return Num;
}

// Emits code to decode the singleton.  Return true if we have matched all the
// well-known bits.
bool ARMFilterChooser::emitSingletonDecoder(raw_ostream &o, unsigned &Indentation,
                                         unsigned Opc) {
  std::vector<unsigned> StartBits;
  std::vector<unsigned> EndBits;
  std::vector<uint64_t> FieldVals;
  insn_t Insn;
  insnWithID(Insn, Opc);

  // This provides a good opportunity to check for possible Ld/St Coprocessor
  // Opcode and escapes if the coproc # is either 10 or 11.  It is a NEON/VFP
  // instruction is disguise.
  if (TargetName == TARGET_ARM && LdStCopEncoding1(Opc)) {
    o.indent(Indentation);
    // A8.6.51 & A8.6.188
    // If coproc = 0b101?, i.e, slice(insn, 11, 8) = 10 or 11, escape.
    o << "if (fieldFromInstruction(insn, 9, 3) == 5) break; // fallthrough\n";
  }

  // Look for islands of undecoded bits of the singleton.
  getIslands(StartBits, EndBits, FieldVals, Insn);

  unsigned Size = StartBits.size();
  unsigned I, NumBits;

  // If we have matched all the well-known bits, just issue a return.
  if (Size == 0) {
    o.indent(Indentation) << "return " << Opc << "; // " << nameWithID(Opc)
                          << '\n';
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
    o << "fieldFromInstruction(insn, " << StartBits[I-1] << ", " << NumBits
      << ") == " << FieldVals[I-1];
    if (I > 1)
      o << " && ";
    else
      o << ")\n";
  }

  o.indent(Indentation) << "  return " << Opc << "; // " << nameWithID(Opc)
                        << '\n';

  return false;
}

// Emits code to decode the singleton, and then to decode the rest.
void ARMFilterChooser::emitSingletonDecoder(raw_ostream &o,
                                            unsigned &Indentation,
                                            ARMFilter &Best) {

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
void ARMFilterChooser::runSingleFilter(ARMFilterChooser &owner,
                                       unsigned startBit,
                                       unsigned numBit, bool mixed) {
  Filters.clear();
  ARMFilter F(*this, startBit, numBit, true);
  Filters.push_back(F);
  BestIndex = 0; // Sole Filter instance to choose from.
  bestFilter().recurse();
}

// reportRegion is a helper function for filterProcessor to mark a region as
// eligible for use as a filter region.
void ARMFilterChooser::reportRegion(bitAttr_t RA, unsigned StartBit,
                                    unsigned BitIndex, bool AllowMixed) {
  if (RA == ATTR_MIXED && AllowMixed)
    Filters.push_back(ARMFilter(*this, StartBit, BitIndex - StartBit, true));   
  else if (RA == ATTR_ALL_SET && !AllowMixed)
    Filters.push_back(ARMFilter(*this, StartBit, BitIndex - StartBit, false));
}

// FilterProcessor scans the well-known encoding bits of the instructions and
// builds up a list of candidate filters.  It chooses the best filter and
// recursively descends down the decoding tree.
bool ARMFilterChooser::filterProcessor(bool AllowMixed, bool Greedy) {
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

  bitAttr_t bitAttrs[BIT_WIDTH];

  // FILTERED bit positions provide no entropy and are not worthy of pursuing.
  // Filter::recurse() set either BIT_TRUE or BIT_FALSE for each position.
  for (BitIndex = 0; BitIndex < BIT_WIDTH; ++BitIndex)
    if (FilterBitValues[BitIndex] == BIT_TRUE ||
        FilterBitValues[BitIndex] == BIT_FALSE)
      bitAttrs[BitIndex] = ATTR_FILTERED;
    else
      bitAttrs[BitIndex] = ATTR_NONE;

  for (InsnIndex = 0; InsnIndex < numInstructions; ++InsnIndex) {
    insn_t insn;

    insnWithID(insn, Opcodes[InsnIndex]);

    for (BitIndex = 0; BitIndex < BIT_WIDTH; ++BitIndex) {
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

  for (BitIndex = 0; BitIndex < BIT_WIDTH; BitIndex++) {
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
void ARMFilterChooser::doFilter() {
  unsigned Num = Opcodes.size();
  assert(Num && "FilterChooser created with no instructions");

  // Heuristics: Use Inst{31-28} as the top level filter for ARM ISA.
  if (TargetName == TARGET_ARM && Parent == NULL) {
    runSingleFilter(*this, 28, 4, false);
    return;
  }

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
bool ARMFilterChooser::emit(raw_ostream &o, unsigned &Indentation) {
  if (Opcodes.size() == 1)
    // There is only one instruction in the set, which is great!
    // Call emitSingletonDecoder() to see whether there are any remaining
    // encodings bits.
    return emitSingletonDecoder(o, Indentation, Opcodes[0]);

  // Choose the best filter to do the decodings!
  if (BestIndex != -1) {
    ARMFilter &Best = bestFilter();
    if (Best.getNumFiltered() == 1)
      emitSingletonDecoder(o, Indentation, Best);
    else
      bestFilter().emit(o, Indentation);
    return false;
  }

  // If we reach here, there is a conflict in decoding.  Let's resolve the known
  // conflicts!
  if ((TargetName == TARGET_ARM || TargetName == TARGET_THUMB) &&
      Opcodes.size() == 2) {
    // Resolve the known conflict sets:
    //
    // 1. source registers are identical => VMOVDneon; otherwise => VORRd
    // 2. source registers are identical => VMOVQ; otherwise => VORRq
    // 3. LDR, LDRcp => return LDR for now.
    // FIXME: How can we distinguish between LDR and LDRcp?  Do we need to?
    // 4. tLDMIA, tLDMIA_UPD => Rn = Inst{10-8}, reglist = Inst{7-0},
    //    wback = registers<Rn> = 0
    // NOTE: (tLDM, tLDM_UPD) resolution must come before Advanced SIMD
    //       addressing mode resolution!!!
    // 5. VLD[234]LN*/VST[234]LN* vs. VLD[234]LN*_UPD/VST[234]LN*_UPD conflicts
    //    are resolved returning the non-UPD versions of the instructions if the
    //    Rm field, i.e., Inst{3-0} is 0b1111.  This is specified in A7.7.1
    //    Advanced SIMD addressing mode.
    const std::string &name1 = nameWithID(Opcodes[0]);
    const std::string &name2 = nameWithID(Opcodes[1]);
    if ((name1 == "VMOVDneon" && name2 == "VORRd") ||
        (name1 == "VMOVQ" && name2 == "VORRq")) {
      // Inserting the opening curly brace for this case block.
      --Indentation; --Indentation;
      o.indent(Indentation) << "{\n";
      ++Indentation; ++Indentation;

      o.indent(Indentation)
        << "field_t N = fieldFromInstruction(insn, 7, 1), "
        << "M = fieldFromInstruction(insn, 5, 1);\n";
      o.indent(Indentation)
        << "field_t Vn = fieldFromInstruction(insn, 16, 4), "
        << "Vm = fieldFromInstruction(insn, 0, 4);\n";
      o.indent(Indentation)
        << "return (N == M && Vn == Vm) ? "
        << Opcodes[0] << " /* " << name1 << " */ : "
        << Opcodes[1] << " /* " << name2 << " */ ;\n";

      // Inserting the closing curly brace for this case block.
      --Indentation; --Indentation;
      o.indent(Indentation) << "}\n";
      ++Indentation; ++Indentation;

      return true;
    }
    if (name1 == "LDR" && name2 == "LDRcp") {
      o.indent(Indentation)
        << "return " << Opcodes[0]
        << "; // Returning LDR for {LDR, LDRcp}\n";
      return true;
    }
    if (name1 == "tLDMIA" && name2 == "tLDMIA_UPD") {
      // Inserting the opening curly brace for this case block.
      --Indentation; --Indentation;
      o.indent(Indentation) << "{\n";
      ++Indentation; ++Indentation;
      
      o.indent(Indentation)
        << "unsigned Rn = fieldFromInstruction(insn, 8, 3), "
        << "list = fieldFromInstruction(insn, 0, 8);\n";
      o.indent(Indentation)
        << "return ((list >> Rn) & 1) == 0 ? "
        << Opcodes[1] << " /* " << name2 << " */ : "
        << Opcodes[0] << " /* " << name1 << " */ ;\n";

      // Inserting the closing curly brace for this case block.
      --Indentation; --Indentation;
      o.indent(Indentation) << "}\n";
      ++Indentation; ++Indentation;

      return true;
    }
    if (sameStringExceptSuffix(name1, name2, "_UPD")) {
      o.indent(Indentation)
        << "return fieldFromInstruction(insn, 0, 4) == 15 ? " << Opcodes[0]
        << " /* " << name1 << " */ : " << Opcodes[1] << "/* " << name2
        << " */ ; // Advanced SIMD addressing mode\n";
      return true;
    }

    // Otherwise, it does not belong to the known conflict sets.
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


////////////////////////////////////////////
//                                        //
//  ARMDEBackend                          //
//  (Helper class for ARMDecoderEmitter)  //
//                                        //
////////////////////////////////////////////

class ARMDecoderEmitter::ARMDEBackend {
public:
  ARMDEBackend(ARMDecoderEmitter &frontend, RecordKeeper &Records) :
    NumberedInstructions(),
    Opcodes(),
    Frontend(frontend),
    Target(Records),
    FC(NULL)
  {
    if (Target.getName() == "ARM")
      TargetName = TARGET_ARM;
    else {
      errs() << "Target name " << Target.getName() << " not recognized\n";
      assert(0 && "Unknown target");
    }

    // Populate the instructions for our TargetName.
    populateInstructions();
  }

  ~ARMDEBackend() {
    if (FC) {
      delete FC;
      FC = NULL;
    }
  }

  void getInstructionsByEnumValue(std::vector<const CodeGenInstruction*>
                                                &NumberedInstructions) {
    // We must emit the PHI opcode first...
    std::string Namespace = Target.getInstNamespace();
    assert(!Namespace.empty() && "No instructions defined.");

    NumberedInstructions = Target.getInstructionsByEnumValue();
  }

  bool populateInstruction(const CodeGenInstruction &CGI, TARGET_NAME_t TN);

  void populateInstructions();

  // Emits disassembler code for instruction decoding.  This delegates to the
  // FilterChooser instance to do the heavy lifting.
  void emit(raw_ostream &o);

protected:
  std::vector<const CodeGenInstruction*> NumberedInstructions;
  std::vector<unsigned> Opcodes;
  // Special case for the ARM chip, which supports ARM and Thumb ISAs.
  // Opcodes2 will be populated with the Thumb opcodes.
  std::vector<unsigned> Opcodes2;
  ARMDecoderEmitter &Frontend;
  CodeGenTarget Target;
  ARMFilterChooser *FC;

  TARGET_NAME_t TargetName;
};

bool ARMDecoderEmitter::
ARMDEBackend::populateInstruction(const CodeGenInstruction &CGI,
                                  TARGET_NAME_t TN) {
  const Record &Def = *CGI.TheDef;
  const StringRef Name = Def.getName();
  uint8_t Form = getByteField(Def, "Form");

  BitsInit &Bits = getBitsField(Def, "Inst");

  // If all the bit positions are not specified; do not decode this instruction.
  // We are bound to fail!  For proper disassembly, the well-known encoding bits
  // of the instruction must be fully specified.
  //
  // This also removes pseudo instructions from considerations of disassembly,
  // which is a better design and less fragile than the name matchings.
  if (Bits.allInComplete()) return false;

  // Ignore "asm parser only" instructions.
  if (Def.getValueAsBit("isAsmParserOnly"))
    return false;

  if (TN == TARGET_ARM) {
    if (Form == ARM_FORMAT_PSEUDO)
      return false;
    if (thumbInstruction(Form))
      return false;

    // Tail calls are other patterns that generate existing instructions.
    if (Name == "TCRETURNdi" || Name == "TCRETURNdiND" ||
        Name == "TCRETURNri" || Name == "TCRETURNriND" ||
        Name == "TAILJMPd"  || Name == "TAILJMPdt" ||
        Name == "TAILJMPdND" || Name == "TAILJMPdNDt" ||
        Name == "TAILJMPr"  || Name == "TAILJMPrND" ||
        Name == "MOVr_TC")
      return false;

    // Delegate ADR disassembly to the more generic ADDri/SUBri instructions.
    if (Name == "ADR")
      return false;

    //
    // The following special cases are for conflict resolutions.
    //

    // RSCSri and RSCSrs set the 's' bit, but are not predicated.  We are
    // better off using the generic RSCri and RSCrs instructions.
    if (Name == "RSCSri" || Name == "RSCSrs") return false;

    // A8-598: VEXT
    // Vector Extract extracts elements from the bottom end of the second
    // operand vector and the top end of the first, concatenates them and
    // places the result in the destination vector.  The elements of the
    // vectors are treated as being 8-bit bitfields.  There is no distinction
    // between data types.  The size of the operation can be specified in
    // assembler as vext.size.  If the value is 16, 32, or 64, the syntax is
    // a pseudo-instruction for a VEXT instruction specifying the equivalent
    // number of bytes.
    //
    // Variants VEXTd16, VEXTd32, VEXTd8, and VEXTdf are reduced to VEXTd8;
    // variants VEXTq16, VEXTq32, VEXTq8, and VEXTqf are reduced to VEXTq8.
    if (Name == "VEXTd16" || Name == "VEXTd32" || Name == "VEXTdf" ||
        Name == "VEXTq16" || Name == "VEXTq32" || Name == "VEXTqf")
      return false;
  } else if (TN == TARGET_THUMB) {
    if (!thumbInstruction(Form))
      return false;

    // A8.6.189 STM / STMIA / STMEA -- Encoding T1
    // There's only STMIA_UPD for Thumb1.
    if (Name == "tSTMIA")
      return false;

    // On Darwin R9 is call-clobbered.  Ignore the non-Darwin counterparts.
    if (Name == "tBL" || Name == "tBLXi" || Name == "tBLXr")
      return false;

    // A8.6.25 BX.  Use the generic tBX_Rm, ignore tBX_RET and tBX_RET_vararg.
    if (Name == "tBX_RET" || Name == "tBX_RET_vararg")
      return false;

    // Ignore the TPsoft (TLS) instructions, which conflict with tBLr9.
    if (Name == "tTPsoft" || Name == "t2TPsoft")
      return false;

    // Ignore tADR, prefer tADDrPCi.
    if (Name == "tADR")
      return false;

    // Delegate t2ADR disassembly to the more generic t2ADDri12/t2SUBri12
    // instructions.
    if (Name == "t2ADR")
      return false;

    // Ignore tADDrSP, tADDspr, and tPICADD, prefer the generic tADDhirr.
    // Ignore t2SUBrSPs, prefer the t2SUB[S]r[r|s].
    // Ignore t2ADDrSPs, prefer the t2ADD[S]r[r|s].
    // Ignore t2ADDrSPi/t2SUBrSPi, which have more generic couterparts.
    // Ignore t2ADDrSPi12/t2SUBrSPi12, which have more generic couterparts
    if (Name == "tADDrSP" || Name == "tADDspr" || Name == "tPICADD" ||
        Name == "t2SUBrSPs" || Name == "t2ADDrSPs" ||
        Name == "t2ADDrSPi" || Name == "t2SUBrSPi" ||
        Name == "t2ADDrSPi12" || Name == "t2SUBrSPi12")
      return false;

    // FIXME: Use ldr.n to work around a Darwin assembler bug.
    // Introduce a workaround with tLDRpciDIS opcode.
    if (Name == "tLDRpci")
      return false;

    // Ignore t2LDRDpci, prefer the generic t2LDRDi8, t2LDRD_PRE, t2LDRD_POST.
    if (Name == "t2LDRDpci")
      return false;

    // Resolve conflicts:
    //
    //   tBfar conflicts with tBLr9
    //   tPOP_RET/t2LDMIA_RET conflict with tPOP/t2LDM (ditto)
    //   tMOVCCi conflicts with tMOVi8
    //   tMOVCCr conflicts with tMOVgpr2gpr
    //   tLDRcp conflicts with tLDRspi
    //   t2MOVCCi16 conflicts with tMOVi16
    if (Name == "tBfar" ||
        Name == "tPOP_RET" || Name == "t2LDMIA_RET" ||
        Name == "tMOVCCi" || Name == "tMOVCCr" ||
        Name == "tLDRcp" || 
        Name == "t2MOVCCi16")
      return false;
  }

  DEBUG({
      // Dumps the instruction encoding format.
      switch (TargetName) {
      case TARGET_ARM:
      case TARGET_THUMB:
        errs() << Name << " " << stringForARMFormat((ARMFormat)Form);
        break;
      }

      errs() << " ";

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

  return true;
}

void ARMDecoderEmitter::ARMDEBackend::populateInstructions() {
  getInstructionsByEnumValue(NumberedInstructions);

  unsigned numUIDs = NumberedInstructions.size();
  if (TargetName == TARGET_ARM) {
    for (unsigned uid = 0; uid < numUIDs; uid++) {
      // filter out intrinsics
      if (!NumberedInstructions[uid]->TheDef->isSubClassOf("InstARM"))
        continue;

      if (populateInstruction(*NumberedInstructions[uid], TargetName))
        Opcodes.push_back(uid);
    }

    // Special handling for the ARM chip, which supports two modes of execution.
    // This branch handles the Thumb opcodes.
    for (unsigned uid = 0; uid < numUIDs; uid++) {
      // filter out intrinsics
      if (!NumberedInstructions[uid]->TheDef->isSubClassOf("InstARM")
          && !NumberedInstructions[uid]->TheDef->isSubClassOf("InstThumb"))
        continue;

      if (populateInstruction(*NumberedInstructions[uid], TARGET_THUMB))
        Opcodes2.push_back(uid);
    }

    return;
  }

  // For other targets.
  for (unsigned uid = 0; uid < numUIDs; uid++) {
    Record *R = NumberedInstructions[uid]->TheDef;
    if (R->getValueAsString("Namespace") == "TargetOpcode")
      continue;

    if (populateInstruction(*NumberedInstructions[uid], TargetName))
      Opcodes.push_back(uid);
  }
}

// Emits disassembler code for instruction decoding.  This delegates to the
// FilterChooser instance to do the heavy lifting.
void ARMDecoderEmitter::ARMDEBackend::emit(raw_ostream &o) {
  switch (TargetName) {
  case TARGET_ARM:
    Frontend.EmitSourceFileHeader("ARM/Thumb Decoders", o);
    break;
  default:
    assert(0 && "Unreachable code!");
  }

  o << "#include \"llvm/Support/DataTypes.h\"\n";
  o << "#include <assert.h>\n";
  o << '\n';
  o << "namespace llvm {\n\n";

  ARMFilterChooser::setTargetName(TargetName);

  switch (TargetName) {
  case TARGET_ARM: {
    // Emit common utility and ARM ISA decoder.
    FC = new ARMFilterChooser(NumberedInstructions, Opcodes);
    // Reset indentation level.
    unsigned Indentation = 0;
    FC->emitTop(o, Indentation);
    delete FC;

    // Emit Thumb ISA decoder as well.
    ARMFilterChooser::setTargetName(TARGET_THUMB);
    FC = new ARMFilterChooser(NumberedInstructions, Opcodes2);
    // Reset indentation level.
    Indentation = 0;
    FC->emitBot(o, Indentation);
    break;
  }
  default:
    assert(0 && "Unreachable code!");
  }

  o << "\n} // End llvm namespace \n";
}

/////////////////////////
//  Backend interface  //
/////////////////////////

void ARMDecoderEmitter::initBackend()
{
  Backend = new ARMDEBackend(*this, Records);
}

void ARMDecoderEmitter::run(raw_ostream &o)
{
  Backend->emit(o);
}

void ARMDecoderEmitter::shutdownBackend()
{
  delete Backend;
  Backend = NULL;
}
