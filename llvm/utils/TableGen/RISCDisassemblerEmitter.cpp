//===- RISCDisassemblerEmitter.cpp - Disassembler Generator ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FIXME: document
//
//===----------------------------------------------------------------------===//

#include "RISCDisassemblerEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <map>
#include <string>
#include <sstream>

#include <asl.h>

using namespace llvm;

#pragma mark Utility classes / structures

// LLVM coding style
#define INDENT_LEVEL 2

class Indenter {
public:
  Indenter() : fDepth(0) {
    memset(fString, ' ', sizeof(fString));
    fString[fDepth] = '\0';
  }

  void push() {
    if (fDepth < sizeof(fString) - INDENT_LEVEL) {
      fString[fDepth] = ' ';
      fDepth += INDENT_LEVEL;
      fString[fDepth] = '\0';
    }
  }

  void pop() {
    if(fDepth >= INDENT_LEVEL) {
      fString[fDepth] = ' ';
      fDepth -= INDENT_LEVEL;
      fString[fDepth] = '\0';
    }
  }

  const char* indent() const {
    return &fString[0];
  }
private:
  char fString[256];
  uint8_t fDepth;
};

#pragma mark Utility functions

#if 0
static int dprintf(int level, const char* format, ...) throw() {
  static aslmsg* fMessage = NULL;

  if (!fMessage) {
    fMessage = new aslmsg;
    *fMessage = asl_new(ASL_TYPE_MSG);
    asl_set(*fMessage, ASL_KEY_FACILITY, "com.apple.llvm.disassembler.emitter.risc");
  }

  va_list ap;

  va_start(ap, format);
  int ret = asl_vlog(NULL, *fMessage, level, format, ap);
  va_end(ap);

  return ret;
}
#endif

static uint8_t byteFromBitsInit(BitsInit& init) {
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

static uint8_t getByteField(const Record& def, const char* str) {
  BitsInit* bits = def.getValueAsBitsInit(str);
  return byteFromBitsInit(*bits);
}

static BitsInit& getBitsField(const Record& def, const char* str) {
  BitsInit* bits = def.getValueAsBitsInit(str);
  return *bits;
}

#if 0
static void binaryFromNumber(std::string& binary,
                             uint64_t number,
                             unsigned minimumWidth = 1) {
  unsigned bitIndex;
  bool emitting = false;

  binary = "0b";

  for (bitIndex = (sizeof(number) * 8); bitIndex > 0; bitIndex--) {
    uint64_t bit = (number & (1 << (bitIndex - 1))) >> (bitIndex - 1);

    if (bitIndex == minimumWidth)
      emitting = true;

    if (bit) {
      emitting = true;
      binary.append("1");
    } else if(emitting) {
      binary.append("0");
    }
  }
}
#endif

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
} bitValue_t;

static bitValue_t bitFromBits(BitsInit& bits, unsigned index) {
  if (BitInit* bit = dynamic_cast<BitInit*>(bits.getBit(index)))
    return bit->getValue() ? BIT_TRUE : BIT_FALSE;

  // The bit is uninitialized.
  return BIT_UNSET;
}

static void dumpBits(std::ostream& o, BitsInit& bits) {
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

#pragma mark Enums

#define ARM_FORMS                   \
  ENTRY(ARM_FORM_PSEUDO,         0) \
  ENTRY(ARM_FORM_MULFRM,         1) \
  ENTRY(ARM_FORM_BRFRM,          2) \
  ENTRY(ARM_FORM_BRMISCFRM,      3) \
  ENTRY(ARM_FORM_DPFRM,          4) \
  ENTRY(ARM_FORM_DPSOREGFRM,     5) \
  ENTRY(ARM_FORM_LDFRM,          6) \
  ENTRY(ARM_FORM_STFRM,          7) \
  ENTRY(ARM_FORM_LDMISCFRM,      8) \
  ENTRY(ARM_FORM_STMISCFRM,      9) \
  ENTRY(ARM_FORM_LDSTMULFRM,    10) \
  ENTRY(ARM_FORM_ARITHMISCFRM,  11) \
  ENTRY(ARM_FORM_EXTFRM,        12) \
  ENTRY(ARM_FORM_VFPUNARYFRM,   13) \
  ENTRY(ARM_FORM_VFPBINARYFRM,  14) \
  ENTRY(ARM_FORM_VFPCONV1FRM,   15) \
  ENTRY(ARM_FORM_VFPCONV2FRM,   16) \
  ENTRY(ARM_FORM_VFPCONV3FRM,   17) \
  ENTRY(ARM_FORM_VFPCONV4FRM,   18) \
  ENTRY(ARM_FORM_VFPCONV5FRM,   19) \
  ENTRY(ARM_FORM_VFPLDSTFRM,    20) \
  ENTRY(ARM_FORM_VFPLDSTMULFRM, 21) \
  ENTRY(ARM_FORM_VFPMISCFRM,    22) \
  ENTRY(ARM_FORM_THUMBFRM,      23) \
  ENTRY(ARM_FORM_NEONFRM,       24) \
  ENTRY(ARM_FORM_NEONGETLNFRM,  25) \
  ENTRY(ARM_FORM_NEONSETLNFRM,  26) \
  ENTRY(ARM_FORM_NEONDUPFRM,    27)

// ARM instruction format specifies the encoding used by the instruction.
#define ENTRY(n, v) n = v,
enum ARMForm {
  ARM_FORMS
  ARM_FORM_max
};
#undef ENTRY

// Converts enum to const char*.
static const char* stringWithARMForm(enum ARMForm form) {
#define ENTRY(n, v) case n: return #n;
  switch(form) {
    ARM_FORMS
  case ARM_FORM_max:
  default:
    return "";
  }
#undef ENTRY
}

static std::vector< std::vector<unsigned> > sConflicts;

class AbstractFilterChooser {
public:
  virtual void emitTop(std::ostream& o, Indenter& i) = 0;
};

template <unsigned tBitWidth>
class FilterChooser : public AbstractFilterChooser {
protected:
  // Representation of the instruction to work on.
  typedef uint8_t insn_t[tBitWidth];

  class Filter {
  protected:
    FilterChooser* fParent; // pointer without ownership
    unsigned fStartBit; // the starting bit position
    unsigned fNumBits; // number of bits to filter
    bool fMixed; // a mixed region contains set and unset bits

    // Map of well-known segment value to the set of uid's with that value. 
    std::map<uint64_t, std::vector<unsigned> > fFilteredInstructions;

    // Set of uid's with non-constant segment values.
    std::vector<unsigned> fVariableInstructions;

    // Map of well-known segment value to its delegate.
    std::map<unsigned, FilterChooser> fSubChoosers;
  public:
    Filter(const Filter& f) :
      fParent(f.fParent),
      fStartBit(f.fStartBit),
      fNumBits(f.fNumBits),
      fMixed(f.fMixed),
      fFilteredInstructions(f.fFilteredInstructions),
      fVariableInstructions(f.fVariableInstructions),
      fSubChoosers(f.fSubChoosers) { }

    Filter(FilterChooser& parent, unsigned startBit, unsigned numBits, bool mixed) :
      fParent(&parent),
      fStartBit(startBit),
      fNumBits(numBits),
      fMixed(mixed)
    {
      unsigned insnIndex;
      unsigned numInsns = fParent->fInstructionsToFilter.size();

      for (insnIndex = 0; insnIndex < numInsns; insnIndex++) {
        insn_t insn;

        // Populates the insn given the uid.
        fParent->insnWithID(insn, fParent->fInstructionsToFilter[insnIndex]);

        uint64_t field;
        // Scans the segment for possibly well-specified encoding bits.
        bool ok = fParent->fieldFromInsn(field, insn, fStartBit, fNumBits);

        if (ok) {
          // The encoding bits are well-known.  Lets add the uid of the
          // instruction into the bucket keyed off the constant field value.
          fFilteredInstructions[field].push_back(fParent->fInstructionsToFilter[insnIndex]);
        } else {
          // Some of the encoding bit(s) are unspecfied.  This contributes to
          // one additional member of "Variable" instructions.
          fVariableInstructions.push_back(fParent->fInstructionsToFilter[insnIndex]);
        }
      }

      assert((fFilteredInstructions.size() + fVariableInstructions.size() > 0)
             && "Filter returns no instruction categories");
    }

    // Divides the decoding task into sub tasks and delegates them to the
    // inferior FilterChooser's.
    void recurse() {
      std::map<uint64_t, std::vector<unsigned> >::const_iterator mapIterator;

      bitValue_t filtered[tBitWidth];
      // Starts by inheriting our parent filter chooser's bit values.
      memcpy(filtered, fParent->fFiltered, sizeof(filtered));

      unsigned bitIndex;

      for (mapIterator = fFilteredInstructions.begin();
           mapIterator != fFilteredInstructions.end();
           mapIterator++) {

        // Marks all the segment positions with either BIT_TRUE or BIT_FALSE.
        for (bitIndex = 0; bitIndex < fNumBits; bitIndex++) {
          if (mapIterator->first & (1 << bitIndex))
            filtered[fStartBit + bitIndex] = BIT_TRUE;
          else
            filtered[fStartBit + bitIndex] = BIT_FALSE;
        }

        // Delegates to an inferior filter chooser for futher processing on this
        // category of instructions.
        fSubChoosers.insert(std::pair<unsigned, FilterChooser>(
                              mapIterator->first,
                              FilterChooser(fParent->fAllInstructions,
                                            mapIterator->second,
                                            filtered,
                                            *fParent)
                              ));
      }

      if (fVariableInstructions.size()) {
        // Conservatively marks each segment position as BIT_UNSET.
        for (bitIndex = 0; bitIndex < fNumBits; bitIndex++)
          filtered[fStartBit + bitIndex] = BIT_UNSET;

        // Delegates to an inferior filter chooser for futher processing on this
        // group of instructions whose segment values are variable.
        fSubChoosers.insert(std::pair<unsigned, FilterChooser>(
                              (unsigned)-1,
                              FilterChooser(fParent->fAllInstructions,
                                            fVariableInstructions,
                                            filtered,
                                            *fParent)
                              ));
      }
    }

    // Emits code to decode instructions given a segment of bits.
    void emit(std::ostream& o, Indenter& i) {
      o << i.indent() << "// Check Inst{";

      if (fNumBits > 1)
        o << (fStartBit + fNumBits - 1) << '-';

      o << fStartBit << "} ...\n";

      o << i.indent() << "switch (fieldFromInstruction(insn, " << fStartBit << ", " << fNumBits << ")) {\n";
      i.push();

      typename std::map<unsigned, FilterChooser>::iterator filterIterator;

      for (filterIterator = fSubChoosers.begin();
           filterIterator != fSubChoosers.end();
           filterIterator++) {

        // Field value -1 implies a non-empty set of variable instructions.
        // See also recurse().
        if (filterIterator->first == (unsigned)-1)
          o << i.indent() << "default:\n";
        else
          o << i.indent() << "case " << filterIterator->first << ":\n";

        // We arrive at a category of instructions with the same segment value.
        // Now delegate to the sub filter chooser for further decodings.
        i.push();
        {
          bool finished = filterIterator->second.emit(o, i);
          if (!finished) o << i.indent() << "break;\n";
        }
        i.pop();
      }

      i.pop();
      o << i.indent() << "}\n";
    }

    // Returns the number of fanout produced by the filter.  More fanout implies
    // the filter distinguishes more categories of instructions.
    unsigned usefulness() const {
      if (fVariableInstructions.size())
        return fFilteredInstructions.size();
      else
        return fFilteredInstructions.size() - 1;
    }
  };

  friend class Filter;

  // Vector of codegen instructions to choose our filter.
  const std::vector<const CodeGenInstruction*>& fAllInstructions;

  // Vector of uid's for this filter chooser to work on.
  const std::vector<unsigned> fInstructionsToFilter;

  // Vector of candidate filters.
  std::vector<Filter> fFilters;

  // Array of bit values passed down from our parent.
  // Set to all BIT_UNFILTERED's for fRarent == NULL.
  bitValue_t fFiltered[tBitWidth];

  // Link to the FilterChooser in our parent chain.
  FilterChooser* fParent;
  
  // Index of the best filter from fFilters.
  int fBestIndex;

public:
  FilterChooser(const FilterChooser& fc) :
    AbstractFilterChooser(),
    fAllInstructions(fc.fAllInstructions),
    fInstructionsToFilter(fc.fInstructionsToFilter),
    fFilters(fc.fFilters),
    fParent(fc.fParent),
    fBestIndex(fc.fBestIndex)
  {
    memcpy(fFiltered, fc.fFiltered, sizeof(fFiltered));
  }

  FilterChooser(const std::vector<const CodeGenInstruction*>& allInstructions,
                const std::vector<unsigned>& instructionsToFilter) :
    fAllInstructions(allInstructions),
    fInstructionsToFilter(instructionsToFilter),
    fFilters(),
    fParent(NULL),
    fBestIndex(-1)
  {
    unsigned bitIndex;

    for (bitIndex = 0; bitIndex < tBitWidth; bitIndex++)
      fFiltered[bitIndex] = BIT_UNFILTERED;

    doFilter();
  }

  FilterChooser(const std::vector<const CodeGenInstruction*>& allInstructions,
                const std::vector<unsigned>& instructionsToFilter,
                bitValue_t (&lastFiltered)[tBitWidth],
                FilterChooser& parent) :
    fAllInstructions(allInstructions),
    fInstructionsToFilter(instructionsToFilter),
    fFilters(),
    fParent(&parent),
    fBestIndex(-1)
  {
    unsigned bitIndex;

    for (bitIndex = 0; bitIndex < tBitWidth; bitIndex++)
      fFiltered[bitIndex] = lastFiltered[bitIndex];

    doFilter();
  }

  void emitTop(std::ostream& o, Indenter& i) {
    switch(tBitWidth) {
    case 8:
      o << i.indent() << "typedef uint8_t field_t;\n";
      break;
    case 16:
      o << i.indent() << "typedef uint16_t field_t;\n";
      break;
    case 32:
      o << i.indent() << "typedef uint32_t field_t;\n";
      break;
    case 64:
      o << i.indent() << "typedef uint64_t field_t;\n";
      break;
    default:
      assert(0 && "Unexpected instruction size!");
    }

    o << '\n';

    o << i.indent() << "field_t fieldFromInstruction(field_t insn, unsigned startBit, unsigned numBits)\n";

    o << i.indent() << "{\n";
    i.push();
    {
      o << i.indent() << "assert(startBit + numBits <= " << tBitWidth << " && \"Instruction field out of bounds!\");\n";
      o << '\n';
      o << i.indent() << "field_t fieldMask;\n";
      o << '\n';
      o << i.indent() << "if (numBits == " << tBitWidth << ")\n";

      i.push();
      {
        o << i.indent() << "fieldMask = (field_t)-1;\n";
      }
      i.pop();

      o << i.indent() << "else\n";

      i.push();
      {
        o << i.indent() << "fieldMask = ((1 << numBits) - 1) << startBit;\n";
      }
      i.pop();

      o << '\n';
      o << i.indent() << "return (insn & fieldMask) >> numBits;\n";
    }
    i.pop();
    o << i.indent() << "}\n";

    o << '\n';

    o << i.indent() << "uint16_t decodeInstruction(field_t insn) {\n";

    i.push();
    {
      // Emits code to decode the instructions.
      emit(o, i);

      o << '\n';
      o << i.indent() << "return 0;\n";
    }
    i.pop();

    o << i.indent() << "}" << std::endl;
  }

protected:
  // Populates the insn given the uid.
  void insnWithID(insn_t& insn, unsigned insnID) const {
    BitsInit& bits = getBitsField(*fAllInstructions[insnID]->TheDef, "Inst");

    unsigned index;

    for (index = 0; index < tBitWidth; index++)
      insn[index] = bitFromBits(bits, index);
  }

  // Returns the record name.
  const std::string& nameWithID(unsigned insnID) const {
    return fAllInstructions[insnID]->TheDef->getName();
  }

  // Populates the field of the insn given the start position and the number of
  // consecutive bits to scan for.
  //
  // Returns false if and on the first uninitialized bit value encountered.
  // Returns true, otherwise.
  const bool fieldFromInsn(uint64_t& field, insn_t& insn, unsigned startBit,
                           unsigned numBits) const {
    unsigned bitIndex;

    field = 0;

    for (bitIndex = 0; bitIndex < numBits; bitIndex++) {
      if (insn[startBit + bitIndex] == BIT_UNSET)
        return false;

      if (insn[startBit + bitIndex] == BIT_TRUE)
        field = field | (1 << bitIndex);
    }

    return true;
  }

  void dumpFilterArray(std::ostream& o, bitValue_t (&filter)[tBitWidth]) {
    unsigned bitIndex;

    for (bitIndex = tBitWidth; bitIndex > 0; bitIndex--) {
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

  void dumpStack(std::ostream& o, const char* prefix) {
    FilterChooser* current = this;

    while (current) {
      o << prefix;

      dumpFilterArray(o, current->fFiltered);

      o << std::endl;

      current = current->fParent;
    }
  }

  Filter& bestFilter() {
    assert(fBestIndex != -1 && "fBestIndex not set");
    return fFilters[fBestIndex];
  }

  // States of our finite state machines.
  typedef enum {
    ATTR_NONE,
    ATTR_FILTERED,
    ATTR_ALL_SET,
    ATTR_ALL_UNSET,
    ATTR_MIXED
  } bitAttr_t;

  bool filterProcessor(bool allowMixed = true) {
    fFilters.clear();
    unsigned numInstructions = fInstructionsToFilter.size();

    assert(numInstructions && "Filter created with no instructions");

    // No further filtering is necessary.
    if (numInstructions == 1)
      return true;

    unsigned bitIndex, insnIndex;

    // We maintain tBitWidth copies of the bitAttrs automaton.
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

    bitAttr_t bitAttrs[tBitWidth];

    // FILTERED bit positions provide no entropy and are not worthy of pursuing.
    // Filter::recurse() set either BIT_TRUE or BIT_FALSE for each position.
    for (bitIndex = 0; bitIndex < tBitWidth; ++bitIndex)
      if (fFiltered[bitIndex] == BIT_TRUE || fFiltered[bitIndex] == BIT_FALSE)
        bitAttrs[bitIndex] = ATTR_FILTERED;
      else
        bitAttrs[bitIndex] = ATTR_NONE;

    for (insnIndex = 0; insnIndex < numInstructions; ++insnIndex) {
      insn_t insn;

      insnWithID(insn, fInstructionsToFilter[insnIndex]);

      for (bitIndex = 0; bitIndex < tBitWidth; ++bitIndex) {
        switch (bitAttrs[bitIndex]) {
        case ATTR_NONE:
          if (insn[bitIndex] == BIT_UNSET)
            bitAttrs[bitIndex] = ATTR_ALL_UNSET;
          else
            bitAttrs[bitIndex] = ATTR_ALL_SET;
          break;
        case ATTR_ALL_SET:
          if (insn[bitIndex] == BIT_UNSET)
            bitAttrs[bitIndex] = ATTR_MIXED;
          break;
        case ATTR_ALL_UNSET:
          if (insn[bitIndex] != BIT_UNSET)
            bitAttrs[bitIndex] = ATTR_MIXED;
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

    bitAttr_t regionAttr = ATTR_NONE;

    unsigned startBit = 0;

    for (bitIndex = 0; bitIndex < tBitWidth; bitIndex++) {
      bitAttr_t bitAttr = bitAttrs[bitIndex];

      assert(bitAttr != ATTR_NONE && "Bit without attributes");

#define SET_START                                                              \
      startBit = bitIndex;

#define REPORT_REGION                                                          \
      if (regionAttr == ATTR_MIXED && allowMixed)                              \
        fFilters.push_back(Filter(*this, startBit, bitIndex - startBit, true));\
      else if (regionAttr == ATTR_ALL_SET && !allowMixed)                      \
        fFilters.push_back(Filter(*this, startBit, bitIndex - startBit, false));

      switch (regionAttr) {
      case ATTR_NONE:
        switch (bitAttr) {
        case ATTR_FILTERED:
          break;
        case ATTR_ALL_SET:
          SET_START
          regionAttr = ATTR_ALL_SET;
          break;
        case ATTR_ALL_UNSET:
          break;
        case ATTR_MIXED:
          SET_START
          regionAttr = ATTR_MIXED;
          break;
        default:
          assert(0 && "Unexpected bitAttr!");
        }
        break;
      case ATTR_ALL_SET:
        switch (bitAttr) {
        case ATTR_FILTERED:
          REPORT_REGION
          regionAttr = ATTR_NONE;
          break;
        case ATTR_ALL_SET:
          break;
        case ATTR_ALL_UNSET:
          REPORT_REGION
          regionAttr = ATTR_NONE;
          break;
        case ATTR_MIXED:
          REPORT_REGION
          SET_START
          regionAttr = ATTR_MIXED;
          break;
        default:
          assert(0 && "Unexpected bitAttr!");
        }
        break;
      case ATTR_MIXED:
        switch (bitAttr) {
        case ATTR_FILTERED:
          REPORT_REGION
          SET_START
          regionAttr = ATTR_NONE;
          break;
        case ATTR_ALL_SET:
          REPORT_REGION
          SET_START
          regionAttr = ATTR_ALL_SET;
          break;
        case ATTR_ALL_UNSET:
          REPORT_REGION
          regionAttr = ATTR_NONE;
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

    switch (regionAttr) {
    case ATTR_NONE:
      break;
    case ATTR_FILTERED:
      break;
    case ATTR_ALL_SET:
      REPORT_REGION
      break;
    case ATTR_ALL_UNSET:
      break;
    case ATTR_MIXED:
      REPORT_REGION
      break;
    }

#undef SET_START
#undef REPORT_REGION

    bool allAreUseless = true;
    fBestIndex = 0;
    unsigned bestUsefulness = 0;
    unsigned filterIndex;
    unsigned numFilters = fFilters.size();

    for (filterIndex = 0; filterIndex < numFilters; ++filterIndex) {
      unsigned usefulness = fFilters[filterIndex].usefulness();

      if (usefulness)
        allAreUseless = false;

      if (usefulness > bestUsefulness) {
        fBestIndex = filterIndex;
        bestUsefulness = usefulness;
      }
    }

    if (!allAreUseless) {
      bestFilter().recurse();
    }

    return !allAreUseless;
  } // end of filterProcessor(bool)

  // Decides on the best configuration of filter(s) to use in order to decode
  // the instructions.  A conflict of instructions may occur, in which case we
  // dump the conflict set to the standard error.
  void doFilter() {
    unsigned numInstructions = fInstructionsToFilter.size();
    assert(numInstructions && "FilterChooser created with no instructions");

    if (filterProcessor(false))
      return;

    if (filterProcessor(true))
      return;

    // If we come to here, the instruction decoding has failed.
    // Print out the instructions in the conflict set...

    fBestIndex = -1;

    std::cerr << "Conflict:" << std::endl;

    unsigned insnIndex;

    dumpStack(std::cerr, "                    ");

    for (insnIndex = 0; insnIndex < numInstructions; insnIndex++) {
      const std::string& name = nameWithID(fInstructionsToFilter[insnIndex]);

      std::cerr << "    " << std::setw(15) << std::left << name << " ";
      dumpBits(std::cerr,
               getBitsField(*fAllInstructions[
                              fInstructionsToFilter[insnIndex]]->TheDef,
                            "Inst"));
      std::cerr << std::endl;
    }
  }

  // Emits code to decode our share of instructions.  Returns true if the
  // emitted code causes a return, which occurs if we known how to decode
  // the instruction at this level or the instruction is not decodeable.
  bool emit(std::ostream& o, Indenter& i) {
    if (fInstructionsToFilter.size() == 1) {
      // There is only one instruction in the set, which is great!
      // Lets return the decoded instruction.
      o << i.indent() << "return " << fInstructionsToFilter[0] << "; // "
        << nameWithID(fInstructionsToFilter[0]) << '\n';
      return true;
    } else if (fBestIndex == -1) {
      if (fInstructionsToFilter.size() == 2) {
        // Resolve the known conflicts sets:
        //
        // 1. source registers are identical => VMOVD; otherwise => VORRd
        // 2. source registers are identical => VMOVQ; otherwise => VORRq
        const std::string& name1 = nameWithID(fInstructionsToFilter[0]);
        const std::string& name2 = nameWithID(fInstructionsToFilter[1]);
        if ((name1 == "VMOVD" && name2 == "VORRd") ||
            (name1 == "VMOVQ" && name2 == "VORRq")) {
          // Inserting the opening curly brace for this case block.
          i.pop();
          o << i.indent() << "{\n";
          i.push();

          o << i.indent() <<
            "field_t N = fieldFromInstruction(insn, 7, 1), M = fieldFromInstruction(insn, 5, 1);\n";
          o << i.indent() <<
            "field_t Vn = fieldFromInstruction(insn, 16, 4), Vm = fieldFromInstruction(insn, 0, 4);\n";
          o << i.indent() << "return (N == M && Vn == Vm) ? "
            << fInstructionsToFilter[0] << " /* " << name1 << " */ : "
            << fInstructionsToFilter[1] << " /* " << name2 << " */ ;\n";

          // Inserting the closing curly brace for this case block.
          i.pop();
          o << i.indent() << "}\n";
          i.push();

          return true;
        }
        // Otherwise, it does not belong to the known conflict sets.
      }
      // We don't know how to decode this instruction!  Dump the conflict set!
      o << i.indent() << "return 0;" << " // Conflict set: ";
      for (int i = 0, N = fInstructionsToFilter.size(); i < N; ++i) {
        o << nameWithID(fInstructionsToFilter[i]);
        if (i < (N - 1))
          o << ", ";
        else
          o << '\n';
      }
      return true;
    } else {
      // Choose the best filter to do the decodings!
      bestFilter().emit(o, i);
      return false;
    }
  }
};

#pragma mark Backend

class RISCDisassemblerEmitter::RISCDEBackend {
public:
  RISCDEBackend(RISCDisassemblerEmitter& frontend) :
    fNumberedInstructions(),
    fTopLevelInstructions(),
    fFrontend(frontend),
    fTarget(),
    fFilterChooser(NULL)
  {
    populateInstructions();

    if (fTarget.getName() == "ARM") {
      fTargetName = TARGET_ARM;
    } else {
      std::cerr << "Target name " << fTarget.getName() << " not recognized" << std::endl;
      assert(0 && "Unknown target");
    }
  }

  ~RISCDEBackend() {
    if (fFilterChooser) {
      delete fFilterChooser;
      fFilterChooser = NULL;
    }
  }

  void getInstructionsByEnumValue(std::vector<const CodeGenInstruction*>& 
                                  numberedInstructions) {
    // Dig down to the proper namespace.  Code shamelessly stolen from
    // InstrEnumEmitter.cpp
    std::string Namespace;
    CodeGenTarget::inst_iterator II, E;

    for (II = fTarget.inst_begin(), E = fTarget.inst_end(); II != E; ++II)
      if (II->second.Namespace != "TargetInstrInfo") {
        Namespace = II->second.Namespace;
        break;
      }

    assert(!Namespace.empty() && "No instructions defined.");

    fTarget.getInstructionsByEnumValue(numberedInstructions);
  }

  bool populateInstruction(const CodeGenInstruction& insn) {
    const Record& def = *insn.TheDef;
    const std::string& name = def.getName();
    uint8_t form = getByteField(def, "Form");
    BitsInit& bits = getBitsField(def, "Inst");

    if (fTargetName == TARGET_ARM) {
      if (form == ARM_FORM_PSEUDO)
        return false;
      if (fTargetName == TARGET_ARM && name[0] == 't')
        return false;
      if (name.find("CMPz") != std::string::npos ||
          name.find("CMNz") != std::string::npos)
        return false;
      if (name.find("BX_RET") != std::string::npos ||
          name.find("BXr9") != std::string::npos ||
          name.find("BLXr9") != std::string::npos)
        return false;

      //
      // The following special cases are for conflict resolutions.
      //

      // RSCSri and RSCSrs set the 's' bit, but are not predicated.  We are
      // better off using the generic RSCri and RSCrs instructions.
      if (name == "RSCSri" || name == "RSCSrs") return false;

      // MOVCCr, MOVCCs, MOVCCi, FCYPScc, FCYPDcc, FNEGScc, and FNEGDcc are used
      // in the compiler to implement conditional moves.  We can ignore them in
      // favor of their more generic versions of instructions.
      // See also SDNode *ARMDAGToDAGISel::Select(SDValue Op).
      if (name == "MOVCCr" || name == "MOVCCs" || name == "MOVCCi" ||
          name == "FCPYScc" || name == "FCPYDcc" ||
          name == "FNEGScc" || name == "FNEGDcc")
        return false;

      // Ignore the *_sfp instructions when decoding.  They are used by the
      // compiler to implement scalar floating point operations using vector
      // operations in order to work around some performance issues.
      if (name.find("_sfp") != std::string::npos) return false;

      // LDM_RET is a special case of LDM (Load Multiple) where the registers
      // loaded include the PC, causing a branch to a loaded address.  Ignore
      // the LDM_RET instruction when decoding.
      if (name == "LDM_RET") return false;

      // Bcc is in a more generic form than B.  Ignore B when decoding.
      if (name == "B") return false;

      // Ignore the non-Darwin BL instructions and the TPsoft (TLS) instruction.
      if (name == "BL" || name == "BL_pred" || name == "TPsoft") return false;

      // Ignore VDUPf[d|q] instructions known to conflict with VDUP32[d-q] for
      // decoding.  The instruction duplicates an element from an ARM core
      // register into every element of the destination vector.  There is no
      // distinction between data types.
      if (name == "VDUPfd" || name == "VDUPfq") return false;
    }

    // Dumps the instruction encoding format.
    switch (fTargetName) {
    case TARGET_ARM:
      std::cerr << name << " " << stringWithARMForm((ARMForm)form);
      break;
    }

    std::cerr << " ";

    // Dumps the instruction encoding bits.
    dumpBits(std::cerr, bits);

    std::cerr << std::endl;

    // Dumps the list of operand info.
    for (unsigned i = 0, e = insn.OperandList.size(); i != e; ++i) {
      CodeGenInstruction::OperandInfo info = insn.OperandList[i];
      const std::string& operandName = info.Name;
      const Record& operandDef = *info.Rec;

      std::cerr << "\t" << operandName << " (" << operandDef.getName() << ") "
                << std::endl;
    }

    return true;
  }

  void populateInstructions() {
    getInstructionsByEnumValue(fNumberedInstructions);

    uint16_t numUIDs = fNumberedInstructions.size();
    uint16_t uid;

    const char* instClass;

    switch (fTargetName) {
    case TARGET_ARM:
      instClass = "InstARM";
    }

    for (uid = 0; uid < numUIDs; uid++) {
      // filter out intrinsics
      if (!fNumberedInstructions[uid]->TheDef->isSubClassOf(instClass))
        continue;

      if (populateInstruction(*fNumberedInstructions[uid]))
        fTopLevelInstructions.push_back(uid);
    }

    switch (fTargetName) {
    case TARGET_ARM:
      fFilterChooser = new FilterChooser<32>(fNumberedInstructions,
                                             fTopLevelInstructions);
    }
  }

  // Emits disassembler code for instruction decoding.  This delegates to the
  // FilterChooser instance to do the heavy lifting.
  void emit(std::ostream& o) {
    Indenter i;
    std::string s;
    raw_string_ostream ro(s);

    switch (fTargetName) {
    case TARGET_ARM:
      fFrontend.EmitSourceFileHeader("ARM Disassembler", ro);
    }

    ro.flush();
    o << s;

    o << i.indent() << "#include <inttypes.h>\n";
    o << i.indent() << "#include <assert.h>\n";
    o << '\n';

    fFilterChooser->emitTop(o, i);
  }

protected:
  std::vector<const CodeGenInstruction*> fNumberedInstructions;
  std::vector<unsigned> fTopLevelInstructions;
  RISCDisassemblerEmitter& fFrontend;
  CodeGenTarget fTarget;
  AbstractFilterChooser* fFilterChooser;

  enum {
    TARGET_ARM = 0
  } fTargetName;
};

#pragma mark Backend interface

void RISCDisassemblerEmitter::initBackend()
{
    fBackend = new RISCDEBackend(*this);
}

void RISCDisassemblerEmitter::run(raw_ostream& o)
{
  std::ostringstream so;
  fBackend->emit(so);
  o << so.str();
}

void RISCDisassemblerEmitter::shutdownBackend()
{
  delete fBackend;
}
