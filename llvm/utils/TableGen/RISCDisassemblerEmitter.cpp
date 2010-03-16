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

#define DEBUG_TYPE "risc-disassembler-emitter"

#include "RISCDisassemblerEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <iomanip>
#include <vector>
#include <cstdio>
#include <map>
#include <string>
#include <sstream>

using namespace llvm;

////////////////////////////////////
//  Utility classes / structures  //
////////////////////////////////////

// LLVM coding style
#define INDENT_LEVEL 2

/// Indenter - A little helper class to keep track of the indentation depth,
/// while the instance object is being passed around.
class Indenter {
public:
  Indenter() : depth(0) {}

  void push() {
    depth += INDENT_LEVEL;
  }

  void pop() {
    if (depth >= INDENT_LEVEL)
      depth -= INDENT_LEVEL;
  }

  // Conversion operator.
  operator int () {
    return depth;
  }
private:
  uint8_t depth;
};

/////////////////////////
//  Utility functions  //
/////////////////////////

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

/// sameStringExceptEndingChar - Return true if the two strings differ only in
/// the ending char.  ("VST4q8a", "VST4q8b", 'a', 'b') as input returns true.
static
bool sameStringExceptEndingChar(const std::string &LHS, const std::string &RHS,
                                char lhc, char rhc) {

  if (LHS.length() > 1 && RHS.length() > 1 && LHS.length() == RHS.length()) {
    unsigned length = LHS.length();
    return LHS.substr(0, length - 1) == RHS.substr(0, length - 1)
      && LHS[length - 1] == lhc && RHS[length - 1] == rhc;
  }

  return false;
}

/// thumbInstruction - Determine whether we have a Thumb instruction.
/// See also ARMInstrFormats.td.
static bool thumbInstruction(uint8_t Form) {
  return Form == 23;
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

static bit_value_t bitFromBits(BitsInit &bits, unsigned index) {
  if (BitInit *bit = dynamic_cast<BitInit*>(bits.getBit(index)))
    return bit->getValue() ? BIT_TRUE : BIT_FALSE;

  // The bit is uninitialized.
  return BIT_UNSET;
}

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

/////////////
//  Enums  //
/////////////

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
  ENTRY(ARM_FORMAT_ARITHMISCFRM,  11) \
  ENTRY(ARM_FORMAT_EXTFRM,        12) \
  ENTRY(ARM_FORMAT_VFPUNARYFRM,   13) \
  ENTRY(ARM_FORMAT_VFPBINARYFRM,  14) \
  ENTRY(ARM_FORMAT_VFPCONV1FRM,   15) \
  ENTRY(ARM_FORMAT_VFPCONV2FRM,   16) \
  ENTRY(ARM_FORMAT_VFPCONV3FRM,   17) \
  ENTRY(ARM_FORMAT_VFPCONV4FRM,   18) \
  ENTRY(ARM_FORMAT_VFPCONV5FRM,   19) \
  ENTRY(ARM_FORMAT_VFPLDSTFRM,    20) \
  ENTRY(ARM_FORMAT_VFPLDSTMULFRM, 21) \
  ENTRY(ARM_FORMAT_VFPMISCFRM,    22) \
  ENTRY(ARM_FORMAT_THUMBFRM,      23) \
  ENTRY(ARM_FORMAT_NEONFRM,       24) \
  ENTRY(ARM_FORMAT_NEONGETLNFRM,  25) \
  ENTRY(ARM_FORMAT_NEONSETLNFRM,  26) \
  ENTRY(ARM_FORMAT_NEONDUPFRM,    27) \
  ENTRY(ARM_FORMAT_LDSTEXFRM,     28) \
  ENTRY(ARM_FORMAT_MISCFRM,       29) \
  ENTRY(ARM_FORMAT_THUMBMISCFRM,  30)

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

#define NS_FORMATS                              \
  ENTRY(NS_FORMAT_NONE,                     0)  \
  ENTRY(NS_FORMAT_VLDSTLane,                1)  \
  ENTRY(NS_FORMAT_VLDSTLaneDbl,             2)  \
  ENTRY(NS_FORMAT_VLDSTRQ,                  3)  \
  ENTRY(NS_FORMAT_NVdImm,                   4)  \
  ENTRY(NS_FORMAT_NVdVmImm,                 5)  \
  ENTRY(NS_FORMAT_NVdVmImmVCVT,             6)  \
  ENTRY(NS_FORMAT_NVdVmImmVDupLane,         7)  \
  ENTRY(NS_FORMAT_NVdVmImmVSHLL,            8)  \
  ENTRY(NS_FORMAT_NVectorShuffle,           9)  \
  ENTRY(NS_FORMAT_NVectorShift,             10) \
  ENTRY(NS_FORMAT_NVectorShift2,            11) \
  ENTRY(NS_FORMAT_NVdVnVmImm,               12) \
  ENTRY(NS_FORMAT_NVdVnVmImmVectorShift,    13) \
  ENTRY(NS_FORMAT_NVdVnVmImmVectorExtract,  14) \
  ENTRY(NS_FORMAT_NVdVnVmImmMulScalar,      15) \
  ENTRY(NS_FORMAT_VTBL,                     16)

// NEON instruction sub-format further classify the NEONFrm instruction.
#define ENTRY(n, v) n = v,
typedef enum {
  NS_FORMATS
  NS_FORMAT_NA
} NSFormat;
#undef ENTRY

// Converts enum to const char*.
static const char *stringForNSFormat(NSFormat form) {
#define ENTRY(n, v) case n: return #n;
  switch(form) {
    NS_FORMATS
  case NS_FORMAT_NA:
  default:
    return "";
  }
#undef ENTRY
}

// Enums for the available target names.
typedef enum {
  TARGET_ARM = 0,
  TARGET_THUMB
} TARGET_NAME_t;

class AbstractFilterChooser {
public:
  static TARGET_NAME_t TargetName;
  static void setTargetName(TARGET_NAME_t tn) { TargetName = tn; }
  virtual ~AbstractFilterChooser() {}
  virtual void emitTop(raw_ostream &o, Indenter &i) = 0;
  virtual void emitBot(raw_ostream &o, Indenter &i) = 0;
};

// Define the symbol here.
TARGET_NAME_t AbstractFilterChooser::TargetName;

template <unsigned tBitWidth>
class FilterChooser : public AbstractFilterChooser {
protected:
  // Representation of the instruction to work on.
  typedef bit_value_t insn_t[tBitWidth];

  class Filter {
  protected:
    FilterChooser *Owner; // pointer without ownership
    unsigned StartBit; // the starting bit position
    unsigned NumBits; // number of bits to filter
    bool Mixed; // a mixed region contains both set and unset bits

    // Map of well-known segment value to the set of uid's with that value. 
    std::map<uint64_t, std::vector<unsigned> > FilteredInstructions;

    // Set of uid's with non-constant segment values.
    std::vector<unsigned> VariableInstructions;

    // Map of well-known segment value to its delegate.
    std::map<unsigned, FilterChooser> FilterChooserMap;

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
    FilterChooser &getVariableFC() {
      assert(NumFiltered == 1);
      assert(FilterChooserMap.size() == 1);
      return FilterChooserMap.find(-1)->second;
    }

    Filter(const Filter &f) :
      Owner(f.Owner),
      StartBit(f.StartBit),
      NumBits(f.NumBits),
      Mixed(f.Mixed),
      FilteredInstructions(f.FilteredInstructions),
      VariableInstructions(f.VariableInstructions),
      FilterChooserMap(f.FilterChooserMap),
      NumFiltered(f.NumFiltered),
      LastOpcFiltered(f.LastOpcFiltered),
      NumVariable(f.NumVariable) { }

    Filter(FilterChooser &owner, unsigned startBit, unsigned numBits,
           bool mixed) :
      Owner(&owner),
      StartBit(startBit),
      NumBits(numBits),
      Mixed(mixed)
    {
      assert(StartBit + NumBits - 1 < tBitWidth);

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

    // Divides the decoding task into sub tasks and delegates them to the
    // inferior FilterChooser's.
    //
    // A special case arises when there's only one entry in the filtered
    // instructions.  In order to unambiguously decode the singleton, we need to
    // match the remaining undecoded encoding bits against the singleton.
    void recurse() {
      std::map<uint64_t, std::vector<unsigned> >::const_iterator mapIterator;

      bit_value_t BitValueArray[tBitWidth];
      // Starts by inheriting our parent filter chooser's filter bit values.
      memcpy(BitValueArray, Owner->FilterBitValues, sizeof(BitValueArray));

      unsigned bitIndex;

      if (VariableInstructions.size()) {
        // Conservatively marks each segment position as BIT_UNSET.
        for (bitIndex = 0; bitIndex < NumBits; bitIndex++)
          BitValueArray[StartBit + bitIndex] = BIT_UNSET;

        // Delegates to an inferior filter chooser for futher processing on this
        // group of instructions whose segment values are variable.
        FilterChooserMap.insert(std::pair<unsigned, FilterChooser>(
                                  (unsigned)-1,
                                  FilterChooser(Owner->AllInstructions,
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
          if (mapIterator->first & (1 << bitIndex))
            BitValueArray[StartBit + bitIndex] = BIT_TRUE;
          else
            BitValueArray[StartBit + bitIndex] = BIT_FALSE;
        }

        // Delegates to an inferior filter chooser for futher processing on this
        // category of instructions.
        FilterChooserMap.insert(std::pair<unsigned, FilterChooser>(
                                  mapIterator->first,
                                  FilterChooser(Owner->AllInstructions,
                                                mapIterator->second,
                                                BitValueArray,
                                                *Owner)
                                  ));
      }
    }

    // Emit code to decode instructions given a segment or segments of bits.
    void emit(raw_ostream &o, Indenter &i) {
      o.indent(i) << "// Check Inst{";

      if (NumBits > 1)
        o << (StartBit + NumBits - 1) << '-';

      o << StartBit << "} ...\n";

      o.indent(i) << "switch (fieldFromInstruction(insn, "
                  << StartBit << ", " << NumBits << ")) {\n";

      typename std::map<unsigned, FilterChooser>::iterator filterIterator;

      bool DefaultCase = false;
      for (filterIterator = FilterChooserMap.begin();
           filterIterator != FilterChooserMap.end();
           filterIterator++) {

        // Field value -1 implies a non-empty set of variable instructions.
        // See also recurse().
        if (filterIterator->first == (unsigned)-1) {
          DefaultCase = true;

          o.indent(i) << "default:\n";
          o.indent(i) << "  break; // fallthrough\n";

          // Closing curly brace for the switch statement.
          // This is unconventional because we want the default processing to be
          // performed for the fallthrough cases as well, i.e., when the "cases"
          // did not prove a decoded instruction.
          o.indent(i) << "}\n";

        } else {
          o.indent(i) << "case " << filterIterator->first << ":\n";
        }

        // We arrive at a category of instructions with the same segment value.
        // Now delegate to the sub filter chooser for further decodings.
        // The case may fallthrough, which happens if the remaining well-known
        // encoding bits do not match exactly.
        if (!DefaultCase) i.push();
        {
          bool finished = filterIterator->second.emit(o, i);
          // For top level default case, there's no need for a break statement.
          if (Owner->isTopLevel() && DefaultCase)
            break;
          if (!finished)
            o.indent(i) << "break;\n";
        }
        if (!DefaultCase) i.pop();
      }

      // If there is no default case, we still need to supply a closing brace.
      if (!DefaultCase) {
        // Closing curly brace for the switch statement.
        o.indent(i) << "}\n";
      }
    }

    // Returns the number of fanout produced by the filter.  More fanout implies
    // the filter distinguishes more categories of instructions.
    unsigned usefulness() const {
      if (VariableInstructions.size())
        return FilteredInstructions.size();
      else
        return FilteredInstructions.size() + 1;
    }
  }; // End of inner class Filter

  friend class Filter;

  // Vector of codegen instructions to choose our filter.
  const std::vector<const CodeGenInstruction*> &AllInstructions;

  // Vector of uid's for this filter chooser to work on.
  const std::vector<unsigned> Opcodes;

  // Vector of candidate filters.
  std::vector<Filter> Filters;

  // Array of bit values passed down from our parent.
  // Set to all BIT_UNFILTERED's for Parent == NULL.
  bit_value_t FilterBitValues[tBitWidth];

  // Links to the FilterChooser above us in the decoding tree.
  FilterChooser *Parent;
  
  // Index of the best filter from Filters.
  int BestIndex;

public:
  FilterChooser(const FilterChooser &FC) :
    AbstractFilterChooser(),
    AllInstructions(FC.AllInstructions),
    Opcodes(FC.Opcodes),
    Filters(FC.Filters),
    Parent(FC.Parent),
    BestIndex(FC.BestIndex)
  {
    memcpy(FilterBitValues, FC.FilterBitValues, sizeof(FilterBitValues));
  }

  FilterChooser(const std::vector<const CodeGenInstruction*> &Insts,
                const std::vector<unsigned> &IDs) :
    AllInstructions(Insts),
    Opcodes(IDs),
    Filters(),
    Parent(NULL),
    BestIndex(-1)
  {
    for (unsigned i = 0; i < tBitWidth; ++i)
      FilterBitValues[i] = BIT_UNFILTERED;

    doFilter();
  }

  FilterChooser(const std::vector<const CodeGenInstruction*> &Insts,
                const std::vector<unsigned> &IDs,
                bit_value_t (&ParentFilterBitValues)[tBitWidth],
                FilterChooser &parent) :
    AllInstructions(Insts),
    Opcodes(IDs),
    Filters(),
    Parent(&parent),
    BestIndex(-1)
  {
    for (unsigned i = 0; i < tBitWidth; ++i)
      FilterBitValues[i] = ParentFilterBitValues[i];

    doFilter();
  }

  // The top level filter chooser has NULL as its parent.
  bool isTopLevel() { return Parent == NULL; }

  // This provides an opportunity for target specific code emission.
  void emitTopHook(raw_ostream &o, Indenter &i) {
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

      // And emit code that references the NSFormat data type.
      // This is meaningful only for NEONFrm instructions.
      o << "static const NSFormat NSFormats[] = {\n";
      for (unsigned i = 0, e = AllInstructions.size(); i != e; ++i) {
        const Record &Def = *(AllInstructions[i]->TheDef);
        const std::string &Name = Def.getName();
        if (Def.isSubClassOf("NeonI") || Def.isSubClassOf("NeonXI"))
          o.indent(2) << 
            stringForNSFormat((NSFormat)getByteField(Def, "NSForm"));
        else
          o << "  NS_FORMAT_NA";

        o << ",\t// Inst #" << i << " = " << Name << '\n';
      }
      o << "  NS_FORMAT_NA\t// Unreachable.\n";
      o << "};\n\n";
    }
  }

  // Emit the top level typedef and decodeInstruction() function.
  void emitTop(raw_ostream &o, Indenter &i) {

    // Run the target specific emit hook.
    emitTopHook(o, i);

    switch(tBitWidth) {
    case 8:
      o.indent(i) << "typedef uint8_t field_t;\n";
      break;
    case 16:
      o.indent(i) << "typedef uint16_t field_t;\n";
      break;
    case 32:
      o.indent(i) << "typedef uint32_t field_t;\n";
      break;
    case 64:
      o.indent(i) << "typedef uint64_t field_t;\n";
      break;
    default:
      assert(0 && "Unexpected instruction size!");
    }

    o << '\n';

    o.indent(i) << "static field_t " <<
    "fieldFromInstruction(field_t insn, unsigned startBit, unsigned numBits)\n";

    o.indent(i) << "{\n";
    i.push();
    {
      o.indent(i) << "assert(startBit + numBits <= " << tBitWidth
                  << " && \"Instruction field out of bounds!\");\n";
      o << '\n';
      o.indent(i) << "field_t fieldMask;\n";
      o << '\n';
      o.indent(i) << "if (numBits == " << tBitWidth << ")\n";

      i.push();
      {
        o.indent(i) << "fieldMask = (field_t)-1;\n";
      }
      i.pop();

      o.indent(i) << "else\n";

      i.push();
      {
        o.indent(i) << "fieldMask = ((1 << numBits) - 1) << startBit;\n";
      }
      i.pop();

      o << '\n';
      o.indent(i) << "return (insn & fieldMask) >> startBit;\n";
    }
    i.pop();
    o.indent(i) << "}\n";

    o << '\n';

    o.indent(i) << "static uint16_t decodeInstruction(field_t insn) {\n";

    i.push();
    {
      // Emits code to decode the instructions.
      emit(o, i);

      o << '\n';
      o.indent(i) << "return 0;\n";
    }
    i.pop();

    o.indent(i) << "}\n";

    o << '\n';

  }

  // This provides an opportunity for target specific code emission after
  // emitTop().
  void emitBot(raw_ostream &o, Indenter &i) {
    if (TargetName == TARGET_THUMB) {
      // Emit code that decodes the Thumb ISA.
      o.indent(i)
        << "static uint16_t decodeThumbInstruction(field_t insn) {\n";

      i.push();
      {
        // Emits code to decode the instructions.
        emit(o, i);

        o << '\n';
        o.indent(i) << "return 0;\n";
      }
      i.pop();

      o.indent(i) << "}\n";
    }
  }

protected:
  // Populates the insn given the uid.
  void insnWithID(insn_t &Insn, unsigned Opcode) const {
    assert(Opcode > 10);
    BitsInit &Bits = getBitsField(*AllInstructions[Opcode]->TheDef, "Inst");

    for (unsigned i = 0; i < tBitWidth; ++i)
      Insn[i] = bitFromBits(Bits, i);
  }

  // Returns the record name.
  const std::string &nameWithID(unsigned Opcode) const {
    return AllInstructions[Opcode]->TheDef->getName();
  }

  // Populates the field of the insn given the start position and the number of
  // consecutive bits to scan for.
  //
  // Returns false if and on the first uninitialized bit value encountered.
  // Returns true, otherwise.
  const bool fieldFromInsn(uint64_t &Field, insn_t &Insn, unsigned StartBit,
                           unsigned NumBits) const {
    Field = 0;

    for (unsigned i = 0; i < NumBits; ++i) {
      if (Insn[StartBit + i] == BIT_UNSET)
        return false;

      if (Insn[StartBit + i] == BIT_TRUE)
        Field = Field | (1 << i);
    }

    return true;
  }

  void dumpFilterArray(raw_ostream &o, bit_value_t (&filter)[tBitWidth]) {
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

  void dumpStack(raw_ostream &o, const char *prefix) {
    FilterChooser *current = this;

    while (current) {
      o << prefix;

      dumpFilterArray(o, current->FilterBitValues);

      o << '\n';

      current = current->Parent;
    }
  }

  Filter &bestFilter() {
    assert(BestIndex != -1 && "BestIndex not set");
    return Filters[BestIndex];
  }

  // States of our finite state machines.
  typedef enum {
    ATTR_NONE,
    ATTR_FILTERED,
    ATTR_ALL_SET,
    ATTR_ALL_UNSET,
    ATTR_MIXED
  } bitAttr_t;

  // Called from Filter::recurse() when singleton exists.  For debug purpose.
  void SingletonExists(unsigned Opc) {

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

  bool ValueSet(bit_value_t V) {
    return (V == BIT_TRUE || V == BIT_FALSE);
  }
  bool ValueNotSet(bit_value_t V) {
    return (V == BIT_UNSET);
  }
  int Value(bit_value_t V) {
    return ValueNotSet(V) ? -1 : (V == BIT_FALSE ? 0 : 1);
  }
  bool PositionFiltered(unsigned i) {
    return ValueSet(FilterBitValues[i]);
  }

  // Calculates the island(s) needed to decode the instruction.
  unsigned getIslands(std::vector<unsigned> &StartBits,
                      std::vector<unsigned> &EndBits,
                      std::vector<uint64_t> &FieldVals, insn_t &Insn)
  {
    unsigned Num, BitNo;
    Num = BitNo = 0;

    uint64_t FieldVal = 0;

    // 0: Init
    // 1: Water
    // 2: Island
    int State = 0;
    int Val = -1;

    for (unsigned i = 0; i < tBitWidth; ++i) {
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
      EndBits.push_back(tBitWidth - 1);
      FieldVals.push_back(FieldVal);
      ++Num;
    }

    /*
    printf("StartBits.size()=%u,EndBits.size()=%u,FieldVals.size()=%u,Num=%u\n",
          (unsigned)StartBits.size(), (unsigned)EndBits.size(),
          (unsigned)FieldVals.size(), Num);
    */

    assert(StartBits.size() == Num && EndBits.size() == Num &&
           FieldVals.size() == Num);

    return Num;
  }

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
  bool emitSingletonDecoder(raw_ostream &o, Indenter &i, unsigned Opc) {

    std::vector<unsigned> StartBits;
    std::vector<unsigned> EndBits;
    std::vector<uint64_t> FieldVals;
    insn_t Insn;
    insnWithID(Insn, Opc);

    if (TargetName == TARGET_ARM && LdStCopEncoding1(Opc)) {
      o.indent(i);
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
      o.indent(i) << "return " << Opc << "; // " << nameWithID(Opc) << '\n';
      return true;
    }

    // Otherwise, there are more decodings to be done!

    // Emit code to match the island(s) for the singleton.
    o.indent(i) << "// Check ";

    for (I = Size; I != 0; --I) {
      o << "Inst{" << EndBits[I-1] << '-' << StartBits[I-1] << "} ";
      if (I > 1)
        o << "&& ";
      else
        o << "for singleton decoding...\n";
    }

    o.indent(i) << "if (";

    for (I = Size; I != 0; --I) {
      NumBits = EndBits[I-1] - StartBits[I-1] + 1;
      o << "fieldFromInstruction(insn, " << StartBits[I-1] << ", " << NumBits
        << ") == " << FieldVals[I-1];
      if (I > 1)
        o << " && ";
      else
        o << ")\n";
    }

    o.indent(i) << "  return " << Opc << "; // " << nameWithID(Opc) << '\n';

    return false;
  }

  // Emits code to decode the singleton, and then to decode the rest.
  void emitSingletonDecoder(raw_ostream &o, Indenter &i, Filter & Best) {

    unsigned Opc = Best.getSingletonOpc();

    emitSingletonDecoder(o, i, Opc);

    // Emit code for the rest.
    o.indent(i) << "else\n";
    i.push();
    {
      Best.getVariableFC().emit(o, i);
    }
    i.pop();
  }

  // Assign a single filter and run with it.
  void runSingleFilter(FilterChooser &owner, unsigned startBit, unsigned numBit,
                       bool mixed) {
    Filters.clear();
    Filter F(*this, startBit, numBit, true);
    Filters.push_back(F);
    BestIndex = 0; // Sole Filter instance to choose from.
    bestFilter().recurse();
  }

  bool filterProcessor(bool AllowMixed, bool Greedy = true) {
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
      if (FilterBitValues[bitIndex] == BIT_TRUE ||
          FilterBitValues[bitIndex] == BIT_FALSE)
        bitAttrs[bitIndex] = ATTR_FILTERED;
      else
        bitAttrs[bitIndex] = ATTR_NONE;

    for (insnIndex = 0; insnIndex < numInstructions; ++insnIndex) {
      insn_t insn;

      insnWithID(insn, Opcodes[insnIndex]);

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
      if (regionAttr == ATTR_MIXED && AllowMixed)                              \
        Filters.push_back(Filter(*this, startBit, bitIndex - startBit, true)); \
      else if (regionAttr == ATTR_ALL_SET && !AllowMixed)                      \
        Filters.push_back(Filter(*this, startBit, bitIndex - startBit, false));

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

    if (!AllUseless) {
      bestFilter().recurse();
    }

    return !AllUseless;
  } // end of filterProcessor(bool)

  // Decides on the best configuration of filter(s) to use in order to decode
  // the instructions.  A conflict of instructions may occur, in which case we
  // dump the conflict set to the standard error.
  void doFilter() {
    unsigned Num = Opcodes.size();
    assert(Num && "FilterChooser created with no instructions");

    // Heuristics: Use Inst{31-28} as the top level filter for ARM ISA.
    if (TargetName == TARGET_ARM && Parent == NULL) {
      runSingleFilter(*this, 28, 4, false);
      return;
    }

    if (filterProcessor(false))
      return;

    if (filterProcessor(true))
      return;

    // Heuristics to cope with conflict set {t2CMPrs, t2SUBSrr, t2SUBSrs} where
    // no single instruction for the maximum ATTR_MIXED region Inst{14-4} has a
    // well-known encoding pattern.  In such case, we backtrack and scan for the
    // the very first consecutive ATTR_ALL_SET region and assign a filter to it.
    if (Num == 3 && filterProcessor(true, false))
      return;

    // If we come to here, the instruction decoding has failed.
    // Print out the instructions in the conflict set...

    BestIndex = -1;

    DEBUG({
        errs() << "Conflict:\n";

        dumpStack(errs(), "\t\t");

        for (unsigned i = 0; i < Num; i++) {
          const std::string &Name = nameWithID(Opcodes[i]);

          errs() << '\t' << Name << " ";
          dumpBits(errs(),
                   getBitsField(*AllInstructions[Opcodes[i]]->TheDef, "Inst"));
          errs() << '\n';
        }
      });
  }

  // Emits code to decode our share of instructions.  Returns true if the
  // emitted code causes a return, which occurs if we know how to decode
  // the instruction at this level or the instruction is not decodeable.
  bool emit(raw_ostream &o, Indenter &i) {
    if (Opcodes.size() == 1) {
      // There is only one instruction in the set, which is great!
      // Call emitSingletonDecoder() to see whether there are any remaining
      // encodings bits.
      return emitSingletonDecoder(o, i, Opcodes[0]);

    } else if (BestIndex == -1) {
      if (TargetName == TARGET_ARM && Opcodes.size() == 2) {
        // Resolve the known conflict sets:
        //
        // 1. source registers are identical => VMOVDneon; otherwise => VORRd
        // 2. source registers are identical => VMOVQ; otherwise => VORRq
        // 3. LDR, LDRcp => return LDR for now.
        // FIXME: How can we distinguish between LDR and LDRcp?  Do we need to?
        // 4. VLD[234]LN*a/VST[234]LN*a vs. VLD[234]LN*b/VST[234]LN*b conflicts
        //    are resolved returning the 'a' versions of the instructions.  Note
        //    that the difference between a/b is that the former is for double-
        //    spaced even registers while the latter is for double-spaced odd
        //    registers.  This is for codegen instruction selection purpose.
        //    For disassembly, it does not matter.
        const std::string &name1 = nameWithID(Opcodes[0]);
        const std::string &name2 = nameWithID(Opcodes[1]);
        if ((name1 == "VMOVDneon" && name2 == "VORRd") ||
            (name1 == "VMOVQ" && name2 == "VORRq")) {
          // Inserting the opening curly brace for this case block.
          i.pop();
          o.indent(i) << "{\n";
          i.push();

          o.indent(i) << "field_t N = fieldFromInstruction(insn, 7, 1), "
                      << "M = fieldFromInstruction(insn, 5, 1);\n";
          o.indent(i) << "field_t Vn = fieldFromInstruction(insn, 16, 4), "
                      << "Vm = fieldFromInstruction(insn, 0, 4);\n";
          o.indent(i) << "return (N == M && Vn == Vm) ? "
                      << Opcodes[0] << " /* " << name1 << " */ : "
                      << Opcodes[1] << " /* " << name2 << " */ ;\n";

          // Inserting the closing curly brace for this case block.
          i.pop();
          o.indent(i) << "}\n";
          i.push();

          return true;
        }
        if (name1 == "LDR" && name2 == "LDRcp") {
          o.indent(i) << "return " << Opcodes[0]
                      << "; // Returning LDR for {LDR, LDRcp}\n";
          return true;
        }
        if (sameStringExceptEndingChar(name1, name2, 'a', 'b')) {
          o.indent(i) << "return " << Opcodes[0] << "; // Returning " << name1
                      << " for {" << name1 << ", " << name2 << "}\n";
          return true;
        }

        // Otherwise, it does not belong to the known conflict sets.
      }
      // We don't know how to decode these instructions!  Dump the conflict set!
      o.indent(i) << "return 0;" << " // Conflict set: ";
      for (int i = 0, N = Opcodes.size(); i < N; ++i) {
        o << nameWithID(Opcodes[i]);
        if (i < (N - 1))
          o << ", ";
        else
          o << '\n';
      }
      return true;
    } else {
      // Choose the best filter to do the decodings!
      Filter &Best = bestFilter();
      if (Best.getNumFiltered() == 1)
        emitSingletonDecoder(o, i, Best);
      else
        bestFilter().emit(o, i);
      return false;
    }
  }
};

///////////////
//  Backend  //
///////////////

class RISCDisassemblerEmitter::RISCDEBackend {
public:
  RISCDEBackend(RISCDisassemblerEmitter &frontend) :
    NumberedInstructions(),
    Opcodes(),
    Frontend(frontend),
    Target(),
    AFC(NULL)
  {
    populateInstructions();

    if (Target.getName() == "ARM") {
      TargetName = TARGET_ARM;
    } else {
      errs() << "Target name " << Target.getName() << " not recognized\n";
      assert(0 && "Unknown target");
    }
  }

  ~RISCDEBackend() {
    if (AFC) {
      delete AFC;
      AFC = NULL;
    }
  }

  void getInstructionsByEnumValue(std::vector<const CodeGenInstruction*>
                                                &NumberedInstructions) {
    // Dig down to the proper namespace.  Code shamelessly stolen from
    // InstrEnumEmitter.cpp
    std::string Namespace;
    CodeGenTarget::inst_iterator II, E;

    for (II = Target.inst_begin(), E = Target.inst_end(); II != E; ++II)
      if (II->second.Namespace != "TargetInstrInfo") {
        Namespace = II->second.Namespace;
        break;
      }

    assert(!Namespace.empty() && "No instructions defined.");

    Target.getInstructionsByEnumValue(NumberedInstructions);
  }

  bool populateInstruction(const CodeGenInstruction &CGI, TARGET_NAME_t TN) {
    const Record &Def = *CGI.TheDef;
    const std::string &Name = Def.getName();
    uint8_t Form = getByteField(Def, "Form");
    BitsInit &Bits = getBitsField(Def, "Inst");

    if (TN == TARGET_ARM) {
      // FIXME: what about Int_MemBarrierV6 and Int_SyncBarrierV6?
      if ((Name != "Int_MemBarrierV7" && Name != "Int_SyncBarrierV7") &&
          Form == ARM_FORMAT_PSEUDO)
        return false;
      if (thumbInstruction(Form))
        return false;
      if (Name.find("CMPz") != std::string::npos /* ||
          Name.find("CMNz") != std::string::npos */)
        return false;

      // Ignore pseudo instructions.
      if (Name == "BXr9" || Name == "BMOVPCRX" || Name == "BMOVPCRXr9")
        return false;

      // VLDRQ/VSTRQ can be hanlded with the more generic VLDMD/VSTMD.
      if (Name == "VLDRQ" || Name == "VSTRQ")
        return false;

      //
      // The following special cases are for conflict resolutions.
      //

      // RSCSri and RSCSrs set the 's' bit, but are not predicated.  We are
      // better off using the generic RSCri and RSCrs instructions.
      if (Name == "RSCSri" || Name == "RSCSrs") return false;

      // MOVCCr, MOVCCs, MOVCCi, FCYPScc, FCYPDcc, FNEGScc, and FNEGDcc are used
      // in the compiler to implement conditional moves.  We can ignore them in
      // favor of their more generic versions of instructions.
      // See also SDNode *ARMDAGToDAGISel::Select(SDValue Op).
      if (Name == "MOVCCr" || Name == "MOVCCs" || Name == "MOVCCi" ||
          Name == "FCPYScc" || Name == "FCPYDcc" ||
          Name == "FNEGScc" || Name == "FNEGDcc")
        return false;

      // Ditto for VMOVDcc, VMOVScc, VNEGDcc, and VNEGScc.
      if (Name == "VMOVDcc" || Name == "VMOVScc" || Name == "VNEGDcc" ||
          Name == "VNEGScc")
        return false;

      // Ignore the *_sfp instructions when decoding.  They are used by the
      // compiler to implement scalar floating point operations using vector
      // operations in order to work around some performance issues.
      if (Name.find("_sfp") != std::string::npos) return false;

      // LLVM added LDM/STM_UPD which conflicts with LDM/STM.
      // Ditto for VLDMS_UPD, VLDMD_UPD, VSTMS_UPD, VSTMD_UPD.
      if (Name == "LDM_UPD" || Name == "STM_UPD" || Name == "VLDMS_UPD" ||
          Name == "VLDMD_UPD" || Name == "VSTMS_UPD" || Name == "VSTMD_UPD")
        return false;

      // LDM_RET is a special case of LDM (Load Multiple) where the registers
      // loaded include the PC, causing a branch to a loaded address.  Ignore
      // the LDM_RET instruction when decoding.
      if (Name == "LDM_RET") return false;

      // Bcc is in a more generic form than B.  Ignore B when decoding.
      if (Name == "B") return false;

      // Ignore the non-Darwin BL instructions and the TPsoft (TLS) instruction.
      if (Name == "BL" || Name == "BL_pred" || Name == "BLX" || Name == "BX" ||
          Name == "TPsoft")
        return false;

      // Ignore VDUPf[d|q] instructions known to conflict with VDUP32[d-q] for
      // decoding.  The instruction duplicates an element from an ARM core
      // register into every element of the destination vector.  There is no
      // distinction between data types.
      if (Name == "VDUPfd" || Name == "VDUPfq") return false;

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

      // Vector Reverse is similar to Vector Extract.  There is no distinction
      // between data types, other than size.
      //
      // VREV64df is equivalent to VREV64d32.
      // VREV64qf is equivalent to VREV64q32.
      if (Name == "VREV64df" || Name == "VREV64qf") return false;

      // VDUPLNfd is equivalent to VDUPLN32d; VDUPfdf is specialized VDUPLN32d.
      // VDUPLNfq is equivalent to VDUPLN32q; VDUPfqf is specialized VDUPLN32q.
      // VLD1df is equivalent to VLD1d32.
      // VLD1qf is equivalent to VLD1q32.
      // VLD2d64 is equivalent to VLD1q64.
      // VST1df is equivalent to VST1d32.
      // VST1qf is equivalent to VST1q32.
      // VST2d64 is equivalent to VST1q64.
      if (Name == "VDUPLNfd" || Name == "VDUPfdf" ||
          Name == "VDUPLNfq" || Name == "VDUPfqf" ||
          Name == "VLD1df" || Name == "VLD1qf" || Name == "VLD2d64" ||
          Name == "VST1df" || Name == "VST1qf" || Name == "VST2d64")
        return false;
    } else if (TN == TARGET_THUMB) {
      if (!thumbInstruction(Form))
        return false;

      // Ignore pseudo instructions.
      if (Name == "tInt_eh_sjlj_setjmp" || Name == "t2Int_eh_sjlj_setjmp" ||
          Name == "t2MOVi32imm" || Name == "tBX" || Name == "tBXr9")
        return false;

      // LLVM added tLDM_UPD which conflicts with tLDM.
      if (Name == "tLDM_UPD")
        return false;

      // On Darwin R9 is call-clobbered.  Ignore the non-Darwin counterparts.
      if (Name == "tBL" || Name == "tBLXi" || Name == "tBLXr")
        return false;

      // Ignore the TPsoft (TLS) instructions, which conflict with tBLr9.
      if (Name == "tTPsoft" || Name == "t2TPsoft")
        return false;

      // Ignore tLEApcrel and tLEApcrelJT, prefer tADDrPCi.
      if (Name == "tLEApcrel" || Name == "tLEApcrelJT")
        return false;

      // Ignore t2LEApcrel, prefer the generic t2ADD* for disassembly printing.
      if (Name == "t2LEApcrel")
        return false;

      // Ignore tADDrSP, tADDspr, and tPICADD, prefer the generic tADDhirr.
      // Ignore t2SUBrSPs, prefer the t2SUB[S]r[r|s].
      // Ignore t2ADDrSPs, prefer the t2ADD[S]r[r|s].
      if (Name == "tADDrSP" || Name == "tADDspr" || Name == "tPICADD" ||
          Name == "t2SUBrSPs" || Name == "t2ADDrSPs")
        return false;

      // Ignore t2LDRDpci, prefer the generic t2LDRDi8, t2LDRD_PRE, t2LDRD_POST.
      if (Name == "t2LDRDpci")
        return false;

      // Ignore t2TBB, t2TBH and prefer the generic t2TBBgen, t2TBHgen.
      if (Name == "t2TBB" || Name == "t2TBH")
        return false;

      // Resolve conflicts:
      //
      //   tBfar conflicts with tBLr9
      //   tCMNz conflicts with tCMN (with assembly format strings being equal)
      //   tPOP_RET/t2LDM_RET conflict with tPOP/t2LDM (ditto)
      //   tMOVCCi conflicts with tMOVi8
      //   tMOVCCr conflicts with tMOVgpr2gpr
      //   tBR_JTr conflicts with tBRIND
      //   tSpill conflicts with tSTRspi
      //   tLDRcp conflicts with tLDRspi
      //   tRestore conflicts with tLDRspi
      //   t2LEApcrelJT conflicts with t2LEApcrel
      //   t2ADDrSPi/t2SUBrSPi have more generic couterparts
      if (Name == "tBfar" ||
          /* Name == "tCMNz" || */ Name == "tCMPzi8" || Name == "tCMPzr" ||
          Name == "tCMPzhir" || /* Name == "t2CMNzrr" || Name == "t2CMNzrs" ||
          Name == "t2CMNzri" || */ Name == "t2CMPzrr" || Name == "t2CMPzrs" ||
          Name == "t2CMPzri" || Name == "tPOP_RET" || Name == "t2LDM_RET" ||
          Name == "tMOVCCi" || Name == "tMOVCCr" || Name == "tBR_JTr" ||
          Name == "tSpill" || Name == "tLDRcp" || Name == "tRestore" ||
          Name == "t2LEApcrelJT" || Name == "t2ADDrSPi" || Name == "t2SUBrSPi")
        return false;
    }

    // Dumps the instruction encoding format.
    switch (TargetName) {
    case TARGET_ARM:
    case TARGET_THUMB:
      DEBUG(errs() << Name << " " << stringForARMFormat((ARMFormat)Form));
      break;
    }

    DEBUG({
        errs() << " ";

        // Dumps the instruction encoding bits.
        dumpBits(errs(), Bits);

        errs() << '\n';

        // Dumps the list of operand info.
        for (unsigned i = 0, e = CGI.OperandList.size(); i != e; ++i) {
          CodeGenInstruction::OperandInfo Info = CGI.OperandList[i];
          const std::string &OperandName = Info.Name;
          const Record &OperandDef = *Info.Rec;

          errs() << "\t" << OperandName << " (" << OperandDef.getName() << ")\n";
        }
      });

    return true;
  }

  void populateInstructions() {
    getInstructionsByEnumValue(NumberedInstructions);

    uint16_t numUIDs = NumberedInstructions.size();
    uint16_t uid;

    const char *instClass = NULL;

    switch (TargetName) {
    case TARGET_ARM:
      instClass = "InstARM";
      break;
    default:
      assert(0 && "Unreachable code!");
    }

    for (uid = 0; uid < numUIDs; uid++) {
      // filter out intrinsics
      if (!NumberedInstructions[uid]->TheDef->isSubClassOf(instClass))
        continue;

      if (populateInstruction(*NumberedInstructions[uid], TargetName))
        Opcodes.push_back(uid);
    }

    // Special handling for the ARM chip, which supports two modes of execution.
    // This branch handles the Thumb opcodes.
    if (TargetName == TARGET_ARM) {
      for (uid = 0; uid < numUIDs; uid++) {
        // filter out intrinsics
        if (!NumberedInstructions[uid]->TheDef->isSubClassOf("InstARM")
            && !NumberedInstructions[uid]->TheDef->isSubClassOf("InstThumb"))
          continue;

          if (populateInstruction(*NumberedInstructions[uid], TARGET_THUMB))
            Opcodes2.push_back(uid);
      }
    }
  }

  // Emits disassembler code for instruction decoding.  This delegates to the
  // FilterChooser instance to do the heavy lifting.
  void emit(raw_ostream &o) {
    Indenter i;
    std::string s;
    raw_string_ostream ro(s);

    switch (TargetName) {
    case TARGET_ARM:
      Frontend.EmitSourceFileHeader("ARM Disassembler", ro);
      break;
    default:
      assert(0 && "Unreachable code!");
    }

    ro.flush();
    o << s;

    o.indent(i) << "#include <inttypes.h>\n";
    o.indent(i) << "#include <assert.h>\n";
    o << '\n';
    o << "namespace llvm {\n\n";

    AbstractFilterChooser::setTargetName(TargetName);

    switch (TargetName) {
    case TARGET_ARM: {
      // Emit common utility and ARM ISA decoder.
      AFC = new FilterChooser<32>(NumberedInstructions, Opcodes);
      AFC->emitTop(o, i);
      delete AFC;

      // Emit Thumb ISA decoder as well.
      AbstractFilterChooser::setTargetName(TARGET_THUMB);
      AFC = new FilterChooser<32>(NumberedInstructions, Opcodes2);
      AFC->emitBot(o, i);
      break;
    }
    default:
      assert(0 && "Unreachable code!");
    }

    o << "\n} // End llvm namespace \n";
  }

protected:
  std::vector<const CodeGenInstruction*> NumberedInstructions;
  std::vector<unsigned> Opcodes;
  // Special case for the ARM chip, which supports ARM and Thumb ISAs.
  // Opcodes2 will be populated with the Thumb opcodes.
  std::vector<unsigned> Opcodes2;
  RISCDisassemblerEmitter &Frontend;
  CodeGenTarget Target;
  AbstractFilterChooser *AFC;

  TARGET_NAME_t TargetName;
};

/////////////////////////
//  Backend interface  //
/////////////////////////

void RISCDisassemblerEmitter::initBackend()
{
    Backend = new RISCDEBackend(*this);
}

void RISCDisassemblerEmitter::run(raw_ostream &o)
{
  Backend->emit(o);
}

void RISCDisassemblerEmitter::shutdownBackend()
{
  delete Backend;
  Backend = NULL;
}
