//===-- Writer.cpp - Library for writing VM bytecode files -------*- C++ -*--=//
//
// This library implements the functionality defined in llvm/Bytecode/Writer.h
//
// This library uses the Analysis library to figure out offsets for
// variables in the method tables...
//
// Note that this file uses an unusual technique of outputting all the bytecode
// to a vector of unsigned char's, then copies the vector to an ostream.  The
// reason for this is that we must do "seeking" in the stream to do back-
// patching, and some very important ostreams that we want to support (like
// pipes) do not support seeking.  :( :( :(
//
// The choice of the vector data structure is influenced by the extremely fast
// "append" speed, plus the free "seek"/replace in the middle of the stream.
//
// Note that the performance of this library is not terribly important, because
// it shouldn't be used by JIT type applications... so it is not a huge focus
// at least.  :)
//
//===----------------------------------------------------------------------===//

#include "WriterInternals.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include <string.h>
#include <algorithm>

BytecodeWriter::BytecodeWriter(vector<unsigned char> &o, const Module *M) 
  : Out(o), Table(M, false) {

  outputSignature();

  // Emit the top level CLASS block.
  BytecodeBlock ModuleBlock(BytecodeFormat::Module, Out);

  // Output largest ID of first "primitive" type:
  output_vbr((unsigned)Type::FirstDerivedTyID, Out);
  align32(Out);

  // Do the whole module now!
  processModule(M);

  // If needed, output the symbol table for the class...
  if (M->hasSymbolTable())
    outputSymbolTable(*M->getSymbolTable());
}

// TODO: REMOVE
#include "llvm/Assembly/Writer.h"

bool BytecodeWriter::processConstPool(const ConstantPool &CP, bool isMethod) {
  BytecodeBlock *CPool = new BytecodeBlock(BytecodeFormat::ConstantPool, Out);

  unsigned NumPlanes = Table.getNumPlanes();

  for (unsigned pno = 0; pno < NumPlanes; pno++) {
    const vector<const Value*> &Plane = Table.getPlane(pno);
    if (Plane.empty()) continue;          // Skip empty type planes...

    unsigned ValNo = 0;   // Don't reemit module constants
    if (isMethod) ValNo = Table.getModuleLevel(pno);
    
    unsigned NumConstants = 0;
    for (unsigned vn = ValNo; vn < Plane.size(); vn++)
      if (Plane[vn]->isConstant())
	NumConstants++;

    if (NumConstants == 0) continue;  // Skip empty type planes...

    // Output type header: [num entries][type id number]
    //
    output_vbr(NumConstants, Out);

    // Output the Type ID Number...
    int Slot = Table.getValSlot(Plane.front()->getType());
    assert (Slot != -1 && "Type in constant pool but not in method!!");
    output_vbr((unsigned)Slot, Out);

    //cerr << "NC: " << NumConstants << " Slot = " << hex << Slot << endl;

    for (; ValNo < Plane.size(); ValNo++) {
      const Value *V = Plane[ValNo];
      if (const ConstPoolVal *CPV = V->castConstant()) {
	//cerr << "Serializing value: <" << V->getType() << ">: " 
	//     << ((const ConstPoolVal*)V)->getStrValue() << ":" 
	//     << Out.size() << "\n";
	outputConstant(CPV);
      }
    }
  }

  delete CPool;  // End bytecode block section!

  if (!isMethod) // The ModuleInfoBlock follows directly after the c-pool
    outputModuleInfoBlock(CP.getParent()->castModuleAsserting());

  return false;
}

void BytecodeWriter::outputModuleInfoBlock(const Module *M) {
  BytecodeBlock ModuleInfoBlock(BytecodeFormat::ModuleGlobalInfo, Out);
  
  // Output the types of the methods in this class
  for (Module::const_iterator I = M->begin(), End = M->end(); I != End; ++I) {
    int Slot = Table.getValSlot((*I)->getType());
    assert(Slot != -1 && "Module const pool is broken!");
    assert(Slot >= Type::FirstDerivedTyID && "Derived type not in range!");
    output_vbr((unsigned)Slot, Out);
  }
  output_vbr((unsigned)Table.getValSlot(Type::VoidTy), Out);
  align32(Out);
}

bool BytecodeWriter::processMethod(const Method *M) {
  BytecodeBlock MethodBlock(BytecodeFormat::Method, Out);

  Table.incorporateMethod(M);

  if (ModuleAnalyzer::processMethod(M)) return true;
  
  // If needed, output the symbol table for the method...
  if (M->hasSymbolTable())
    outputSymbolTable(*M->getSymbolTable());

  Table.purgeMethod();
  return false;
}


bool BytecodeWriter::processBasicBlock(const BasicBlock *BB) {
  BytecodeBlock MethodBlock(BytecodeFormat::BasicBlock, Out);
  return ModuleAnalyzer::processBasicBlock(BB);
}

void BytecodeWriter::outputSymbolTable(const SymbolTable &MST) {
  BytecodeBlock MethodBlock(BytecodeFormat::SymbolTable, Out);

  for (SymbolTable::const_iterator TI = MST.begin(); TI != MST.end(); ++TI) {
    SymbolTable::type_const_iterator I = MST.type_begin(TI->first);
    SymbolTable::type_const_iterator End = MST.type_end(TI->first);
    int Slot;
    
    if (I == End) continue;  // Don't mess with an absent type...

    // Symtab block header: [num entries][type id number]
    output_vbr(MST.type_size(TI->first), Out);

    Slot = Table.getValSlot(TI->first);
    assert(Slot != -1 && "Type in symtab, but not in table!");
    output_vbr((unsigned)Slot, Out);

    for (; I != End; ++I) {
      // Symtab entry: [def slot #][name]
      Slot = Table.getValSlot(I->second);
      assert (Slot != -1 && "Value in symtab but not in method!!");
      output_vbr((unsigned)Slot, Out);
      output(I->first, Out, false); // Don't force alignment...
    }
  }
}

void WriteBytecodeToFile(const Module *C, ostream &Out) {
  assert(C && "You can't write a null class!!");

  vector<unsigned char> Buffer;

  // This object populates buffer for us...
  BytecodeWriter BCW(Buffer, C);

  // Okay, write the vector out to the ostream now...
  Out.write(&Buffer[0], Buffer.size());
  Out.flush();
}
