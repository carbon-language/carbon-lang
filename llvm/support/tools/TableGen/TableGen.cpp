//===- TableGen.cpp - Top-Level TableGen implementation -------------------===//
//
// TableGen is a tool which can be used to build up a description of something,
// then invoke one or more "tablegen backends" to emit information about the
// description in some predefined format.  In practice, this is used by the LLVM
// code generators to automate generation of a code generator through a
// high-level description of the target.
//
//===----------------------------------------------------------------------===//

#include "Record.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include "CodeEmitterGen.h"
#include "RegisterInfoEmitter.h"
#include <algorithm>
#include <fstream>

enum ActionType {
  PrintRecords,
  GenEmitter,
  GenRegister, GenRegisterHeader,
  PrintEnums,
  Parse,
};

namespace {
  cl::opt<ActionType>
  Action(cl::desc("Action to perform:"),
         cl::values(clEnumValN(PrintRecords, "print-records",
                               "Print all records to stdout (default)"),
                    clEnumValN(GenEmitter, "gen-emitter",
                               "Generate machine code emitter"),
                    clEnumValN(GenRegister, "gen-register-desc",
                               "Generate a register info description"),
                    clEnumValN(GenRegisterHeader, "gen-register-desc-header",
                               "Generate a register info description header"),
                    clEnumValN(PrintEnums, "print-enums",
                               "Print enum values for a class"),
                    clEnumValN(Parse, "parse",
                               "Interpret machine code (testing only)"),
                    0));

  cl::opt<std::string>
  Class("class", cl::desc("Print Enum list for this class"),
        cl::value_desc("class name"));

  cl::opt<std::string>
  OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
                 cl::init("-"));

  cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));
}


void ParseFile(const std::string &Filename);

RecordKeeper Records;

static Init *getBit(Record *R, unsigned BitNo) {
  const std::vector<RecordVal> &V = R->getValues();
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    if (V[i].getPrefix()) {
      assert(dynamic_cast<BitsInit*>(V[i].getValue()) &&
	     "Can only handle fields of bits<> type!");
      BitsInit *I = (BitsInit*)V[i].getValue();
      if (BitNo < I->getNumBits())
	return I->getBit(BitNo);
      BitNo -= I->getNumBits();
    }

  std::cerr << "Cannot find requested bit!\n";
  abort();
  return 0;
}

static unsigned getNumBits(Record *R) {
  const std::vector<RecordVal> &V = R->getValues();
  unsigned Num = 0;
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    if (V[i].getPrefix()) {
      assert(dynamic_cast<BitsInit*>(V[i].getValue()) &&
	     "Can only handle fields of bits<> type!");
      Num += ((BitsInit*)V[i].getValue())->getNumBits();
    }
  return Num;
}

static bool BitsAreFixed(Record *I1, Record *I2, unsigned BitNo) {
  return dynamic_cast<BitInit*>(getBit(I1, BitNo)) &&
         dynamic_cast<BitInit*>(getBit(I2, BitNo));
}

static bool BitsAreEqual(Record *I1, Record *I2, unsigned BitNo) {
  BitInit *Bit1 = dynamic_cast<BitInit*>(getBit(I1, BitNo));
  BitInit *Bit2 = dynamic_cast<BitInit*>(getBit(I2, BitNo));

  return Bit1 && Bit2 && Bit1->getValue() == Bit2->getValue();
}

static bool BitRangesEqual(Record *I1, Record *I2,
			   unsigned Start, unsigned End) {
  for (unsigned i = Start; i != End; ++i)
    if (!BitsAreEqual(I1, I2, i))
      return false;
  return true;
}

static unsigned getFirstFixedBit(Record *R, unsigned FirstFixedBit) {
  // Look for the first bit of the pair that are required to be 0 or 1.
  while (!dynamic_cast<BitInit*>(getBit(R, FirstFixedBit)))
    ++FirstFixedBit;
  return FirstFixedBit;
}

static void FindInstDifferences(Record *I1, Record *I2,
				unsigned FirstFixedBit, unsigned MaxBits,
				unsigned &FirstVaryingBitOverall,
				unsigned &LastFixedBitOverall) {
  // Compare the first instruction to the rest of the instructions, looking for
  // fields that differ.
  //
  unsigned FirstVaryingBit = FirstFixedBit;
  while (FirstVaryingBit < MaxBits && BitsAreEqual(I1, I2, FirstVaryingBit))
    ++FirstVaryingBit;

  unsigned LastFixedBit = FirstVaryingBit;
  while (LastFixedBit < MaxBits && BitsAreFixed(I1, I2, LastFixedBit))
    ++LastFixedBit;
  
  if (FirstVaryingBit < FirstVaryingBitOverall)
    FirstVaryingBitOverall = FirstVaryingBit;
  if (LastFixedBit < LastFixedBitOverall)
    LastFixedBitOverall = LastFixedBit;
}

static bool getBitValue(Record *R, unsigned BitNo) {
  Init *I = getBit(R, BitNo);
  assert(dynamic_cast<BitInit*>(I) && "Bit should be fixed!");
  return ((BitInit*)I)->getValue();
}

struct BitComparator {
  unsigned BitBegin, BitEnd;
  BitComparator(unsigned B, unsigned E) : BitBegin(B), BitEnd(E) {}

  bool operator()(Record *R1, Record *R2) { // Return true if R1 is less than R2
    for (unsigned i = BitBegin; i != BitEnd; ++i) {
      bool V1 = getBitValue(R1, i), V2 = getBitValue(R2, i);
      if (V1 < V2)
	return true;
      else if (V2 < V1)
	return false;
    }
    return false;
  }
};

static void PrintRange(std::vector<Record*>::iterator I, 
		       std::vector<Record*>::iterator E) {
  while (I != E) std::cerr << **I++;
}

static bool getMemoryBit(unsigned char *M, unsigned i) {
  return (M[i/8] & (1 << (i&7))) != 0;
}

static unsigned getFirstFixedBitInSequence(std::vector<Record*>::iterator IB,
					   std::vector<Record*>::iterator IE,
					   unsigned StartBit) {
  unsigned FirstFixedBit = 0;
  for (std::vector<Record*>::iterator I = IB; I != IE; ++I)
    FirstFixedBit = std::max(FirstFixedBit, getFirstFixedBit(*I, StartBit));
  return FirstFixedBit;
}

// ParseMachineCode - Try to split the vector of instructions (which is
// intentially taken by-copy) in half, narrowing down the possible instructions
// that we may have found.  Eventually, this list will get pared down to zero or
// one instruction, in which case we have a match or failure.
//
static Record *ParseMachineCode(std::vector<Record*>::iterator InstsB, 
				std::vector<Record*>::iterator InstsE,
				unsigned char *M) {
  assert(InstsB != InstsE && "Empty range?");
  if (InstsB+1 == InstsE) {
    // Only a single instruction, see if we match it...
    Record *Inst = *InstsB;
    for (unsigned i = 0, e = getNumBits(Inst); i != e; ++i)
      if (BitInit *BI = dynamic_cast<BitInit*>(getBit(Inst, i)))
	if (getMemoryBit(M, i) != BI->getValue())
	  throw std::string("Parse failed!\n");
    return Inst;
  }

  unsigned MaxBits = ~0;
  for (std::vector<Record*>::iterator I = InstsB; I != InstsE; ++I)
    MaxBits = std::min(MaxBits, getNumBits(*I));

  unsigned FirstFixedBit = getFirstFixedBitInSequence(InstsB, InstsE, 0);
  unsigned FirstVaryingBit, LastFixedBit;
  do {
    FirstVaryingBit = ~0;
    LastFixedBit = ~0;
    for (std::vector<Record*>::iterator I = InstsB+1; I != InstsE; ++I)
      FindInstDifferences(*InstsB, *I, FirstFixedBit, MaxBits,
			  FirstVaryingBit, LastFixedBit);
    if (FirstVaryingBit == MaxBits) {
      std::cerr << "ERROR: Could not find bit to distinguish between "
		<< "the following entries!\n";
      PrintRange(InstsB, InstsE);
    }

#if 0
    std::cerr << "FVB: " << FirstVaryingBit << " - " << LastFixedBit
	      << ": " << InstsE-InstsB << "\n";
#endif

    FirstFixedBit = getFirstFixedBitInSequence(InstsB, InstsE, FirstVaryingBit);
  } while (FirstVaryingBit != FirstFixedBit);

  //std::cerr << "\n\nXXXXXXXXXXXXXXXXX\n\n";
  //PrintRange(InstsB, InstsE);

  // Sort the Insts list so that the entries have all of the bits in the range
  // [FirstVaryingBit,LastFixedBit) sorted.  These bits are all guaranteed to be
  // set to either 0 or 1 (BitInit values), which simplifies things.
  //
  std::sort(InstsB, InstsE, BitComparator(FirstVaryingBit, LastFixedBit));

  // Once the list is sorted by these bits, split the bit list into smaller
  // lists, and recurse on each one.
  //
  std::vector<Record*>::iterator RangeBegin = InstsB;
  Record *Match = 0;
  while (RangeBegin != InstsE) {
    std::vector<Record*>::iterator RangeEnd = RangeBegin+1;
    while (RangeEnd != InstsE &&
          BitRangesEqual(*RangeBegin, *RangeEnd, FirstVaryingBit, LastFixedBit))
      ++RangeEnd;
    
    // We just identified a range of equal instructions.  If this range is the
    // input range, we were not able to distinguish between the instructions in
    // the set.  Print an error and exit!
    //
    if (RangeBegin == InstsB && RangeEnd == InstsE) {
      std::cerr << "Error: Could not distinguish among the following insts!:\n";
      PrintRange(InstsB, InstsE);
      abort();
    }
    
#if 0
    std::cerr << "FVB: " << FirstVaryingBit << " - " << LastFixedBit
	      << ": [" << RangeEnd-RangeBegin << "] - ";
    for (int i = LastFixedBit-1; i >= (int)FirstVaryingBit; --i)
      std::cerr << (int)((BitInit*)getBit(*RangeBegin, i))->getValue() << " ";
    std::cerr << "\n";
#endif

    if (Record *R = ParseMachineCode(RangeBegin, RangeEnd, M)) {
      if (Match) {
	std::cerr << "Error: Multiple matches found:\n";
	PrintRange(InstsB, InstsE);
      }

      assert(Match == 0 && "Multiple matches??");
      Match = R;
    }
    RangeBegin = RangeEnd;
  }

  return Match;
}

static void PrintValue(Record *I, unsigned char *Ptr, const RecordVal &Val) {
  assert(dynamic_cast<BitsInit*>(Val.getValue()) &&
	 "Can only handle undefined bits<> types!");
  BitsInit *BI = (BitsInit*)Val.getValue();
  assert(BI->getNumBits() <= 32 && "Can only handle fields up to 32 bits!");

  unsigned Value = 0;
  const std::vector<RecordVal> &Vals = I->getValues();

  // Start by filling in fixed values...
  for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i)
    if (BitInit *B = dynamic_cast<BitInit*>(BI->getBit(i)))
      Value |= B->getValue() << i;
  
  // Loop over all of the fields in the instruction adding in any
  // contributions to this value (due to bit references).
  //
  unsigned Offset = 0;
  for (unsigned f = 0, e = Vals.size(); f != e; ++f)
    if (Vals[f].getPrefix()) {
      BitsInit *FieldInitializer = (BitsInit*)Vals[f].getValue();
      if (&Vals[f] == &Val) {
	// Read the bits directly now...
	for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i)
	  Value |= getMemoryBit(Ptr, Offset+i) << i;
	break;
      }
      
      // Scan through the field looking for bit initializers of the current
      // variable...
      for (unsigned i = 0, e = FieldInitializer->getNumBits(); i != e; ++i)
	if (VarBitInit *VBI =
	    dynamic_cast<VarBitInit*>(FieldInitializer->getBit(i))) {
          TypedInit *TI = VBI->getVariable();
          if (VarInit *VI = dynamic_cast<VarInit*>(TI)) {
            if (VI->getName() == Val.getName())
              Value |= getMemoryBit(Ptr, Offset+i) << VBI->getBitNum();
          } else if (FieldInit *FI = dynamic_cast<FieldInit*>(TI)) {
            // FIXME: implement this!
            std::cerr << "FIELD INIT not implemented yet!\n";
          }
	}	
      Offset += FieldInitializer->getNumBits();
    }

  std::cout << "0x" << std::hex << Value << std::dec;
}

static void PrintInstruction(Record *I, unsigned char *Ptr) {
  std::cout << "Inst " << getNumBits(I)/8 << " bytes: "
	    << "\t" << I->getName() << "\t" << *I->getValue("Name")->getValue()
	    << "\t";
  
  const std::vector<RecordVal> &Vals = I->getValues();
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    if (!Vals[i].getValue()->isComplete()) {
      std::cout << Vals[i].getName() << "=";
      PrintValue(I, Ptr, Vals[i]);
      std::cout << "\t";
    }
  
  std::cout << "\n";// << *I;
}

static void ParseMachineCode() {
  // X86 code
  unsigned char Buffer[] = {
                             0x55,             // push EBP
			     0x89, 0xE5,       // mov EBP, ESP
			     //0x83, 0xEC, 0x08, // sub ESP, 0x8
			     0xE8, 1, 2, 3, 4, // call +0x04030201
			     0x89, 0xEC,       // mov ESP, EBP
			     0x5D,             // pop EBP
			     0xC3,             // ret
			     0x90,             // nop
			     0xC9,             // leave
			     0x89, 0xF6,       // mov ESI, ESI
			     0x68, 1, 2, 3, 4, // push 0x04030201
			     0x5e,             // pop ESI
			     0xFF, 0xD0,       // call EAX
			     0xB8, 1, 2, 3, 4, // mov EAX, 0x04030201
			     0x85, 0xC0,       // test EAX, EAX
			     0xF4,             // hlt
  };

#if 0
  // SparcV9 code
  unsigned char Buffer[] = { 0xbf, 0xe0, 0x20, 0x1f, 0x1, 0x0, 0x0, 0x1, 
                             0x0, 0x0, 0x0, 0x0, 0xc1, 0x0, 0x20, 0x1, 0x1,
                             0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                             0x0, 0x0, 0x0, 0x0, 0x0, 0x40, 0x0, 0x0, 0x0, 0x1,
                             0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                             0x0, 0x0, 0xaf, 0xe8, 0x20, 0x17
  };
#endif

  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");

  unsigned char *BuffPtr = Buffer;
  while (1) {
    Record *R = ParseMachineCode(Insts.begin(), Insts.end(), BuffPtr);
    PrintInstruction(R, BuffPtr);

    unsigned Bits = getNumBits(R);
    assert((Bits & 7) == 0 && "Instruction is not an even number of bytes!");
    BuffPtr += Bits/8;
  }
}


int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  ParseFile(InputFilename);

  std::ostream *Out = &std::cout;
  if (OutputFilename != "-") {
    Out = new std::ofstream(OutputFilename.c_str());

    if (!Out->good()) {
      std::cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
      return 1;
    }

    // Make sure the file gets removed if *gasp* tablegen crashes...
    RemoveFileOnSignal(OutputFilename);
  }

  try {
    switch (Action) {
    case PrintRecords:
      *Out << Records;           // No argument, dump all contents
      break;
    case Parse:
      ParseMachineCode();
      break;
    case GenEmitter:
      CodeEmitterGen(Records).run(*Out);
      break;
    case GenRegister:
      RegisterInfoEmitter(Records).run(*Out);
      break;
    case GenRegisterHeader:
      RegisterInfoEmitter(Records).runHeader(*Out);
      break;
    case PrintEnums:
      std::vector<Record*> Recs = Records.getAllDerivedDefinitions(Class);
      for (unsigned i = 0, e = Recs.size(); i != e; ++i)
        *Out << Recs[i] << ", ";
      *Out << "\n";
      break;
    }
  } catch (const std::string &Error) {
    std::cerr << Error << "\n";
    if (Out != &std::cout) delete Out;
    return 1;
  }

  if (Out != &std::cout) delete Out;
  return 0;
}
