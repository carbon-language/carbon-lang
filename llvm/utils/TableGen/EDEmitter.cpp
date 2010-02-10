//===- EDEmitter.cpp - Generate instruction descriptions for ED -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of each
// instruction in a format that the enhanced disassembler can use to tokenize
// and parse instructions.
//
//===----------------------------------------------------------------------===//

#include "EDEmitter.h"

#include "AsmWriterInst.h"
#include "CodeGenTarget.h"
#include "Record.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <string>

#define MAX_OPERANDS 5
#define MAX_SYNTAXES 2

using namespace llvm;

///////////////////////////////////////////////////////////
// Support classes for emitting nested C data structures //
///////////////////////////////////////////////////////////

namespace {
  
  class EnumEmitter {
  private:
    std::string Name;
    std::vector<std::string> Entries;
  public:
    EnumEmitter(const char *N) : Name(N) { 
    }
    int addEntry(const char *e) { 
      Entries.push_back(std::string(e));
      return Entries.size() - 1; 
    }
    void emit(raw_ostream &o, unsigned int &i) {
      o.indent(i) << "enum " << Name.c_str() << " {" << "\n";
      i += 2;
      
      unsigned int index = 0;
      unsigned int numEntries = Entries.size();
      for(index = 0; index < numEntries; ++index) {
        o.indent(i) << Entries[index];
        if(index < (numEntries - 1))
          o << ",";
        o << "\n";
      }
      
      i -= 2;
      o.indent(i) << "};" << "\n";
    }
    
    void emitAsFlags(raw_ostream &o, unsigned int &i) {
      o.indent(i) << "enum " << Name.c_str() << " {" << "\n";
      i += 2;
      
      unsigned int index = 0;
      unsigned int numEntries = Entries.size();
      unsigned int flag = 1;
      for (index = 0; index < numEntries; ++index) {
        o.indent(i) << Entries[index] << " = " << format("0x%x", flag);
        if (index < (numEntries - 1))
          o << ",";
        o << "\n";
        flag <<= 1;
      }
      
      i -= 2;
      o.indent(i) << "};" << "\n";
    }
  };

  class StructEmitter {
  private:
    std::string Name;
    std::vector<std::string> MemberTypes;
    std::vector<std::string> MemberNames;
  public:
    StructEmitter(const char *N) : Name(N) {
    }
    void addMember(const char *t, const char *n) {
      MemberTypes.push_back(std::string(t));
      MemberNames.push_back(std::string(n));
    }
    void emit(raw_ostream &o, unsigned int &i) {
      o.indent(i) << "struct " << Name.c_str() << " {" << "\n";
      i += 2;
      
      unsigned int index = 0;
      unsigned int numMembers = MemberTypes.size();
      for (index = 0; index < numMembers; ++index) {
        o.indent(i) << MemberTypes[index] << " " << MemberNames[index] << ";";
        o << "\n";
      }
      
      i -= 2;
      o.indent(i) << "};" << "\n";
    }
  };
  
  class ConstantEmitter {
  public:
    virtual ~ConstantEmitter() { }
    virtual void emit(raw_ostream &o, unsigned int &i) = 0;
  };
  
  class LiteralConstantEmitter : public ConstantEmitter {
  private:
    std::string Literal;
  public:
    LiteralConstantEmitter(const char *literal) : Literal(literal) {
    }
    LiteralConstantEmitter(int literal) {
      char buf[256];
      snprintf(buf, 256, "%d", literal);
      Literal = buf;
    }
    void emit(raw_ostream &o, unsigned int &i) {
      o << Literal;
    }
  };
  
  class CompoundConstantEmitter : public ConstantEmitter {
  private:
    std::vector<ConstantEmitter*> Entries;
  public:
    CompoundConstantEmitter() {
    }
    ~CompoundConstantEmitter() {
      unsigned int index;
      unsigned int numEntries = Entries.size();
      for (index = 0; index < numEntries; ++index) {
        delete Entries[index];
      }
    }
    CompoundConstantEmitter &addEntry(ConstantEmitter *e) {
      Entries.push_back(e);
      return *this;
    }
    void emit(raw_ostream &o, unsigned int &i) {
      o << "{" << "\n";
      i += 2;
  
      unsigned int index;
      unsigned int numEntries = Entries.size();
      for (index = 0; index < numEntries; ++index) {
        o.indent(i);
        Entries[index]->emit(o, i);
        if (index < (numEntries - 1))
          o << ",";
        o << "\n";
      }
      
      i -= 2;
      o.indent(i) << "}";
    }
  };
  
  class FlagsConstantEmitter : public ConstantEmitter {
  private:
    std::vector<std::string> Flags;
  public:
    FlagsConstantEmitter() {
    }
    FlagsConstantEmitter &addEntry(const char *f) {
      Flags.push_back(std::string(f));
      return *this;
    }
    void emit(raw_ostream &o, unsigned int &i) {
      unsigned int index;
      unsigned int numFlags = Flags.size();
      if (numFlags == 0)
        o << "0";
      
      for (index = 0; index < numFlags; ++index) {
        o << Flags[index].c_str();
        if (index < (numFlags - 1))
          o << " | ";
      }
    }
  };
}

EDEmitter::EDEmitter(RecordKeeper &R) : Records(R) {
}

/// populateOperandOrder - Accepts a CodeGenInstruction and generates its
///   AsmWriterInst for the desired assembly syntax, giving an ordered list of
///   operands in the order they appear in the printed instruction.  Then, for
///   each entry in that list, determines the index of the same operand in the
///   CodeGenInstruction, and emits the resulting mapping into an array, filling
///   in unused slots with -1.
///
/// @arg operandOrder - The array that will be populated with the operand
///                     mapping.  Each entry will contain -1 (invalid index
///                     into the operands present in the AsmString) or a number
///                     representing an index in the operand descriptor array.
/// @arg inst         - The instruction to use when looking up the operands
/// @arg syntax       - The syntax to use, according to LLVM's enumeration
void populateOperandOrder(CompoundConstantEmitter *operandOrder,
                          const CodeGenInstruction &inst,
                          unsigned syntax) {
  unsigned int numArgs = 0;
  
  AsmWriterInst awInst(inst, syntax, -1, -1);
  
  std::vector<AsmWriterOperand>::iterator operandIterator;
  
  for (operandIterator = awInst.Operands.begin();
       operandIterator != awInst.Operands.end();
       ++operandIterator) {
    if (operandIterator->OperandType == 
        AsmWriterOperand::isMachineInstrOperand) {
      char buf[2];
      snprintf(buf, sizeof(buf), "%u", operandIterator->CGIOpNo);
      operandOrder->addEntry(new LiteralConstantEmitter(buf));
      numArgs++;
    }
  }
  
  for(; numArgs < MAX_OPERANDS; numArgs++) {
    operandOrder->addEntry(new LiteralConstantEmitter("-1"));
  }
}

/////////////////////////////////////////////////////
// Support functions for handling X86 instructions //
/////////////////////////////////////////////////////

#define ADDFLAG(flag) flags->addEntry(flag)

#define REG(str) if (name == str) { ADDFLAG("kOperandFlagRegister"); return 0; }
#define MEM(str) if (name == str) { ADDFLAG("kOperandFlagMemory"); return 0; }
#define LEA(str) if (name == str) { ADDFLAG("kOperandFlagEffectiveAddress"); \
                                    return 0; }
#define IMM(str) if (name == str) { ADDFLAG("kOperandFlagImmediate"); \
                                    return 0; }
#define PCR(str) if (name == str) { ADDFLAG("kOperandFlagMemory"); \
                                    ADDFLAG("kOperandFlagPCRelative"); \
                                    return 0; }

/// X86FlagFromOpName - Processes the name of a single X86 operand (which is
///   actually its type) and translates it into an operand flag
///
/// @arg flags    - The flags object to add the flag to
/// @arg name     - The name of the operand
static int X86FlagFromOpName(FlagsConstantEmitter *flags,
                             const std::string &name) {
  REG("GR8");
  REG("GR8_NOREX");
  REG("GR16");
  REG("GR32");
  REG("GR32_NOREX");
  REG("FR32");
  REG("RFP32");
  REG("GR64");
  REG("FR64");
  REG("VR64");
  REG("RFP64");
  REG("RFP80");
  REG("VR128");
  REG("RST");
  REG("SEGMENT_REG");
  REG("DEBUG_REG");
  REG("CONTROL_REG_32");
  REG("CONTROL_REG_64");
  
  MEM("i8mem");
  MEM("i8mem_NOREX");
  MEM("i16mem");
  MEM("i32mem");
  MEM("f32mem");
  MEM("ssmem");
  MEM("opaque32mem");
  MEM("opaque48mem");
  MEM("i64mem");
  MEM("f64mem");
  MEM("sdmem");
  MEM("f80mem");
  MEM("opaque80mem");
  MEM("i128mem");
  MEM("f128mem");
  MEM("opaque512mem");
  
  LEA("lea32mem");
  LEA("lea64_32mem");
  LEA("lea64mem");
  
  IMM("i8imm");
  IMM("i16imm");
  IMM("i16i8imm");
  IMM("i32imm");
  IMM("i32imm_pcrel");
  IMM("i32i8imm");
  IMM("i64imm");
  IMM("i64i8imm");
  IMM("i64i32imm");
  IMM("i64i32imm_pcrel");
  IMM("SSECC");
  
  PCR("brtarget8");
  PCR("offset8");
  PCR("offset16");
  PCR("offset32");
  PCR("offset64");
  PCR("brtarget");
  
  return 1;
}

#undef REG
#undef MEM
#undef LEA
#undef IMM
#undef PCR
#undef ADDFLAG

/// X86PopulateOperands - Handles all the operands in an X86 instruction, adding
///   the appropriate flags to their descriptors
///
/// @operandFlags - A reference the array of operand flag objects
/// @inst         - The instruction to use as a source of information
static void X86PopulateOperands(
  FlagsConstantEmitter *(&operandFlags)[MAX_OPERANDS],
  const CodeGenInstruction &inst) {
  if (!inst.TheDef->isSubClassOf("X86Inst"))
    return;
  
  unsigned int index;
  unsigned int numOperands = inst.OperandList.size();
  
  for (index = 0; index < numOperands; ++index) {
    const CodeGenInstruction::OperandInfo &operandInfo = 
      inst.OperandList[index];
    Record &rec = *operandInfo.Rec;
    
    if (X86FlagFromOpName(operandFlags[index], rec.getName())) {
      errs() << "Operand type: " << rec.getName().c_str() << "\n";
      errs() << "Operand name: " << operandInfo.Name.c_str() << "\n";
      errs() << "Instruction mame: " << inst.TheDef->getName().c_str() << "\n";
      llvm_unreachable("Unhandled type");
    }
  }
}

/// decorate1 - Decorates a named operand with a new flag
///
/// @operandFlags - The array of operand flag objects, which don't have names
/// @inst         - The CodeGenInstruction, which provides a way to translate
///                 between names and operand indices
/// @opName       - The name of the operand
/// @flag         - The name of the flag to add
static inline void decorate1(FlagsConstantEmitter *(&operandFlags)[MAX_OPERANDS],
                             const CodeGenInstruction &inst,
                             const char *opName,
                             const char *opFlag) {
  unsigned opIndex;
  
  opIndex = inst.getOperandNamed(std::string(opName));
  
  operandFlags[opIndex]->addEntry(opFlag);
}

#define DECORATE1(opName, opFlag) decorate1(operandFlags, inst, opName, opFlag)

#define MOV(source, target) {                       \
  instFlags.addEntry("kInstructionFlagMove");       \
  DECORATE1(source, "kOperandFlagSource");          \
  DECORATE1(target, "kOperandFlagTarget");          \
}

#define BRANCH(target) {                            \
  instFlags.addEntry("kInstructionFlagBranch");     \
  DECORATE1(target, "kOperandFlagTarget");          \
}

#define PUSH(source) {                              \
  instFlags.addEntry("kInstructionFlagPush");       \
  DECORATE1(source, "kOperandFlagSource");          \
}

#define POP(target) {                               \
  instFlags.addEntry("kInstructionFlagPop");        \
  DECORATE1(target, "kOperandFlagTarget");          \
}

#define CALL(target) {                              \
  instFlags.addEntry("kInstructionFlagCall");       \
  DECORATE1(target, "kOperandFlagTarget");          \
}

#define RETURN() {                                  \
  instFlags.addEntry("kInstructionFlagReturn");     \
}

/// X86ExtractSemantics - Performs various checks on the name of an X86
///   instruction to determine what sort of an instruction it is and then adds 
///   the appropriate flags to the instruction and its operands
///
/// @arg instFlags    - A reference to the flags for the instruction as a whole
/// @arg operandFlags - A reference to the array of operand flag object pointers
/// @arg inst         - A reference to the original instruction
static void X86ExtractSemantics(FlagsConstantEmitter &instFlags,
                                FlagsConstantEmitter *(&operandFlags)[MAX_OPERANDS],
                                const CodeGenInstruction &inst) {
  const std::string &name = inst.TheDef->getName();
    
  if (name.find("MOV") != name.npos) {
    if (name.find("MOV_V") != name.npos) {
      // ignore (this is a pseudoinstruction)
    }
    else if (name.find("MASK") != name.npos) {
      // ignore (this is a masking move)
    }
    else if (name.find("r0") != name.npos) {
      // ignore (this is a pseudoinstruction)
    }
    else if (name.find("PS") != name.npos ||
             name.find("PD") != name.npos) {
      // ignore (this is a shuffling move)
    }
    else if (name.find("MOVS") != name.npos) {
      // ignore (this is a string move)
    }
    else if (name.find("_F") != name.npos) {
      // TODO handle _F moves to ST(0)
    }
    else if (name.find("a") != name.npos) {
      // TODO handle moves to/from %ax
    }
    else if (name.find("CMOV") != name.npos) {
      MOV("src2", "dst");
    }
    else if (name.find("PC") != name.npos) {
      MOV("label", "reg")
    }
    else {
      MOV("src", "dst");
    }
  }
  
  if (name.find("JMP") != name.npos ||
      name.find("J") == 0) {
    if (name.find("FAR") != name.npos && name.find("i") != name.npos) {
      BRANCH("off");
    }
    else {
      BRANCH("dst");
    }
  }
  
  if (name.find("PUSH") != name.npos) {
    if (name.find("FS") != name.npos ||
        name.find("GS") != name.npos) {
      instFlags.addEntry("kInstructionFlagPush");
      // TODO add support for fixed operands
    }
    else if (name.find("F") != name.npos) {
      // ignore (this pushes onto the FP stack)
    }
    else if (name[name.length() - 1] == 'm') {
      PUSH("src");
    }
    else if (name.find("i") != name.npos) {
      PUSH("imm");
    }
    else {
      PUSH("reg");
    }
  }
  
  if (name.find("POP") != name.npos) {
    if (name.find("POPCNT") != name.npos) {
      // ignore (not a real pop)
    }
    else if (name.find("FS") != name.npos ||
             name.find("GS") != name.npos) {
      instFlags.addEntry("kInstructionFlagPop");
      // TODO add support for fixed operands
    }
    else if (name.find("F") != name.npos) {
      // ignore (this pops from the FP stack)
    }
    else if (name[name.length() - 1] == 'm') {
      POP("dst");
    }
    else {
      POP("reg");
    }
  }
  
  if (name.find("CALL") != name.npos) {
    if (name.find("ADJ") != name.npos) {
      // ignore (not a call)
    }
    else if (name.find("SYSCALL") != name.npos) {
      // ignore (doesn't go anywhere we know about)
    }
    else if (name.find("VMCALL") != name.npos) {
      // ignore (rather different semantics than a regular call)
    }
    else if (name.find("FAR") != name.npos && name.find("i") != name.npos) {
      CALL("off");
    }
    else {
      CALL("dst");
    }
  }
  
  if (name.find("RET") != name.npos) {
    RETURN();
  }
}

#undef MOV
#undef BRANCH
#undef PUSH
#undef POP
#undef CALL
#undef RETURN

#undef COND_DECORATE_2
#undef COND_DECORATE_1
#undef DECORATE1

/// populateInstInfo - Fills an array of InstInfos with information about each 
///   instruction in a target
///
/// @arg infoArray  - The array of InstInfo objects to populate
/// @arg target     - The CodeGenTarget to use as a source of instructions
static void populateInstInfo(CompoundConstantEmitter &infoArray,
                             CodeGenTarget &target) {
  std::vector<const CodeGenInstruction*> numberedInstructions;
  target.getInstructionsByEnumValue(numberedInstructions);
  
  unsigned int index;
  unsigned int numInstructions = numberedInstructions.size();
  
  for (index = 0; index < numInstructions; ++index) {
    const CodeGenInstruction& inst = *numberedInstructions[index];
    
    CompoundConstantEmitter *infoStruct = new CompoundConstantEmitter;
    infoArray.addEntry(infoStruct);
    
    FlagsConstantEmitter *instFlags = new FlagsConstantEmitter;
    infoStruct->addEntry(instFlags);
    
    LiteralConstantEmitter *numOperandsEmitter = 
      new LiteralConstantEmitter(inst.OperandList.size());
    infoStruct->addEntry(numOperandsEmitter);
                         
    CompoundConstantEmitter *operandFlagArray = new CompoundConstantEmitter;
    infoStruct->addEntry(operandFlagArray);
        
    FlagsConstantEmitter *operandFlags[MAX_OPERANDS];
    
    for (unsigned operandIndex = 0; operandIndex < MAX_OPERANDS; ++operandIndex) {
      operandFlags[operandIndex] = new FlagsConstantEmitter;
      operandFlagArray->addEntry(operandFlags[operandIndex]);
    }
 
    unsigned numSyntaxes = 0;
    
    if (target.getName() == "X86") {
      X86PopulateOperands(operandFlags, inst);
      X86ExtractSemantics(*instFlags, operandFlags, inst);
      numSyntaxes = 2;
    }
    
    CompoundConstantEmitter *operandOrderArray = new CompoundConstantEmitter;
    infoStruct->addEntry(operandOrderArray);
    
    for (unsigned syntaxIndex = 0; syntaxIndex < MAX_SYNTAXES; ++syntaxIndex) {
      CompoundConstantEmitter *operandOrder = new CompoundConstantEmitter;
      operandOrderArray->addEntry(operandOrder);
      
      if (syntaxIndex < numSyntaxes) {
        populateOperandOrder(operandOrder, inst, syntaxIndex);
      }
      else {
        for (unsigned operandIndex = 0; 
             operandIndex < MAX_OPERANDS; 
             ++operandIndex) {
          operandOrder->addEntry(new LiteralConstantEmitter("-1"));
        }
      }
    }
  }
}

void EDEmitter::run(raw_ostream &o) {
  unsigned int i = 0;
  
  CompoundConstantEmitter infoArray;
  CodeGenTarget target;
  
  populateInstInfo(infoArray, target);
  
  o << "InstInfo instInfo" << target.getName().c_str() << "[] = ";
  infoArray.emit(o, i);
  o << ";" << "\n";
}

void EDEmitter::runHeader(raw_ostream &o) {
  EmitSourceFileHeader("Enhanced Disassembly Info Header", o);
  
  o << "#ifndef EDInfo_" << "\n";
  o << "#define EDInfo_" << "\n";
  o << "\n";
  o << "#include <inttypes.h>" << "\n";
  o << "\n";
  o << "#define MAX_OPERANDS " << format("%d", MAX_OPERANDS) << "\n";
  o << "#define MAX_SYNTAXES " << format("%d", MAX_SYNTAXES) << "\n";
  o << "\n";
  
  unsigned int i = 0;
  
  EnumEmitter operandFlags("OperandFlags");
  operandFlags.addEntry("kOperandFlagImmediate");
  operandFlags.addEntry("kOperandFlagRegister");
  operandFlags.addEntry("kOperandFlagMemory");
  operandFlags.addEntry("kOperandFlagEffectiveAddress");
  operandFlags.addEntry("kOperandFlagPCRelative");
  operandFlags.addEntry("kOperandFlagSource");
  operandFlags.addEntry("kOperandFlagTarget");
  operandFlags.emitAsFlags(o, i);
  
  o << "\n";
  
  EnumEmitter instructionFlags("InstructionFlags");
  instructionFlags.addEntry("kInstructionFlagMove");
  instructionFlags.addEntry("kInstructionFlagBranch");
  instructionFlags.addEntry("kInstructionFlagPush");
  instructionFlags.addEntry("kInstructionFlagPop");
  instructionFlags.addEntry("kInstructionFlagCall");
  instructionFlags.addEntry("kInstructionFlagReturn");
  instructionFlags.emitAsFlags(o, i);
  
  o << "\n";
  
  StructEmitter instInfo("InstInfo");
  instInfo.addMember("uint32_t", "instructionFlags");
  instInfo.addMember("uint8_t", "numOperands");
  instInfo.addMember("uint8_t", "operandFlags[MAX_OPERANDS]");
  instInfo.addMember("const char", "operandOrders[MAX_SYNTAXES][MAX_OPERANDS]");
  instInfo.emit(o, i);
  
  o << "\n";
  o << "#endif" << "\n";
}
