//===--- Bitcode/Writer/Writer.cpp - Bitcode Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Bitcode writer implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "ValueEnumerator.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

static const unsigned CurVersion = 0;

static void WriteStringRecord(unsigned Code, const std::string &Str, 
                              unsigned AbbrevToUse, BitstreamWriter &Stream) {
  SmallVector<unsigned, 64> Vals;
  
  // Code: [strlen, strchar x N]
  Vals.push_back(Str.size());
  for (unsigned i = 0, e = Str.size(); i != e; ++i)
    Vals.push_back(Str[i]);
    
  // Emit the finished record.
  Stream.EmitRecord(Code, Vals, AbbrevToUse);
}


/// WriteTypeTable - Write out the type table for a module.
static void WriteTypeTable(const ValueEnumerator &VE, BitstreamWriter &Stream) {
  const ValueEnumerator::TypeList &TypeList = VE.getTypes();
  
  Stream.EnterSubblock(bitc::TYPE_BLOCK_ID, 4 /*count from # abbrevs */);
  SmallVector<uint64_t, 64> TypeVals;
  
  // FIXME: Set up abbrevs now that we know the width of the type fields, etc.
  
  // Emit an entry count so the reader can reserve space.
  TypeVals.push_back(TypeList.size());
  Stream.EmitRecord(bitc::TYPE_CODE_NUMENTRY, TypeVals);
  TypeVals.clear();
  
  // Loop over all of the types, emitting each in turn.
  for (unsigned i = 0, e = TypeList.size(); i != e; ++i) {
    const Type *T = TypeList[i].first;
    int AbbrevToUse = 0;
    unsigned Code = 0;
    
    switch (T->getTypeID()) {
    case Type::PackedStructTyID: // FIXME: Delete Type::PackedStructTyID.
    default: assert(0 && "Unknown type!");
    case Type::VoidTyID:   Code = bitc::TYPE_CODE_VOID;   break;
    case Type::FloatTyID:  Code = bitc::TYPE_CODE_FLOAT;  break;
    case Type::DoubleTyID: Code = bitc::TYPE_CODE_DOUBLE; break;
    case Type::LabelTyID:  Code = bitc::TYPE_CODE_LABEL;  break;
    case Type::OpaqueTyID: Code = bitc::TYPE_CODE_OPAQUE; break;
    case Type::IntegerTyID:
      // INTEGER: [width]
      Code = bitc::TYPE_CODE_INTEGER;
      TypeVals.push_back(cast<IntegerType>(T)->getBitWidth());
      break;
    case Type::PointerTyID:
      // POINTER: [pointee type]
      Code = bitc::TYPE_CODE_POINTER;
      TypeVals.push_back(VE.getTypeID(cast<PointerType>(T)->getElementType()));
      break;

    case Type::FunctionTyID: {
      const FunctionType *FT = cast<FunctionType>(T);
      // FUNCTION: [isvararg, #pararms, paramty x N]
      Code = bitc::TYPE_CODE_FUNCTION;
      TypeVals.push_back(FT->isVarArg());
      TypeVals.push_back(VE.getTypeID(FT->getReturnType()));
      // FIXME: PARAM ATTR ID!
      TypeVals.push_back(FT->getNumParams());
      for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i)
        TypeVals.push_back(VE.getTypeID(FT->getParamType(i)));
      break;
    }
    case Type::StructTyID: {
      const StructType *ST = cast<StructType>(T);
      // STRUCT: [ispacked, #elts, eltty x N]
      Code = bitc::TYPE_CODE_STRUCT;
      TypeVals.push_back(ST->isPacked());
      TypeVals.push_back(ST->getNumElements());
      // Output all of the element types...
      for (StructType::element_iterator I = ST->element_begin(),
           E = ST->element_end(); I != E; ++I)
        TypeVals.push_back(VE.getTypeID(*I));
      break;
    }
    case Type::ArrayTyID: {
      const ArrayType *AT = cast<ArrayType>(T);
      // ARRAY: [numelts, eltty]
      Code = bitc::TYPE_CODE_ARRAY;
      TypeVals.push_back(AT->getNumElements());
      TypeVals.push_back(VE.getTypeID(AT->getElementType()));
      break;
    }
    case Type::VectorTyID: {
      const VectorType *VT = cast<VectorType>(T);
      // VECTOR [numelts, eltty]
      Code = bitc::TYPE_CODE_VECTOR;
      TypeVals.push_back(VT->getNumElements());
      TypeVals.push_back(VE.getTypeID(VT->getElementType()));
      break;
    }
    }

    // Emit the finished record.
    Stream.EmitRecord(Code, TypeVals, AbbrevToUse);
    TypeVals.clear();
  }
  
  Stream.ExitBlock();
}

/// WriteTypeSymbolTable - Emit a block for the specified type symtab.
static void WriteTypeSymbolTable(const TypeSymbolTable &TST,
                                 const ValueEnumerator &VE,
                                 BitstreamWriter &Stream) {
  if (TST.empty()) return;

  Stream.EnterSubblock(bitc::TYPE_SYMTAB_BLOCK_ID, 3);

  // FIXME: Set up the abbrev, we know how many types there are!
  // FIXME: We know if the type names can use 7-bit ascii.
  
  SmallVector<unsigned, 64> NameVals;

  for (TypeSymbolTable::const_iterator TI = TST.begin(), TE = TST.end(); 
       TI != TE; ++TI) {
    unsigned AbbrevToUse = 0;

    // TST_ENTRY: [typeid, namelen, namechar x N]
    NameVals.push_back(VE.getTypeID(TI->second));
    
    const std::string &Str = TI->first;
    NameVals.push_back(Str.size());
    for (unsigned i = 0, e = Str.size(); i != e; ++i)
      NameVals.push_back(Str[i]);
  
    // Emit the finished record.
    Stream.EmitRecord(bitc::TST_ENTRY_CODE, NameVals, AbbrevToUse);
    NameVals.clear();
  }

  Stream.ExitBlock();
}

static unsigned getEncodedLinkage(const GlobalValue *GV) {
  switch (GV->getLinkage()) {
  default: assert(0 && "Invalid linkage!");
  case GlobalValue::ExternalLinkage:     return 0;
  case GlobalValue::WeakLinkage:         return 1;
  case GlobalValue::AppendingLinkage:    return 2;
  case GlobalValue::InternalLinkage:     return 3;
  case GlobalValue::LinkOnceLinkage:     return 4;
  case GlobalValue::DLLImportLinkage:    return 5;
  case GlobalValue::DLLExportLinkage:    return 6;
  case GlobalValue::ExternalWeakLinkage: return 7;
  }
}

static unsigned getEncodedVisibility(const GlobalValue *GV) {
  switch (GV->getVisibility()) {
  default: assert(0 && "Invalid visibility!");
  case GlobalValue::DefaultVisibility: return 0;
  case GlobalValue::HiddenVisibility:  return 1;
  }
}

// Emit top-level description of module, including target triple, inline asm,
// descriptors for global variables, and function prototype info.
static void WriteModuleInfo(const Module *M, const ValueEnumerator &VE,
                            BitstreamWriter &Stream) {
  // Emit the list of dependent libraries for the Module.
  for (Module::lib_iterator I = M->lib_begin(), E = M->lib_end(); I != E; ++I)
    WriteStringRecord(bitc::MODULE_CODE_DEPLIB, *I, 0/*TODO*/, Stream);

  // Emit various pieces of data attached to a module.
  if (!M->getTargetTriple().empty())
    WriteStringRecord(bitc::MODULE_CODE_TRIPLE, M->getTargetTriple(),
                      0/*TODO*/, Stream);
  if (!M->getDataLayout().empty())
    WriteStringRecord(bitc::MODULE_CODE_DATALAYOUT, M->getDataLayout(),
                      0/*TODO*/, Stream);
  if (!M->getModuleInlineAsm().empty())
    WriteStringRecord(bitc::MODULE_CODE_ASM, M->getModuleInlineAsm(),
                      0/*TODO*/, Stream);

  // Emit information about sections, computing how many there are.  Also
  // compute the maximum alignment value.
  std::map<std::string, unsigned> SectionMap;
  unsigned MaxAlignment = 0;
  unsigned MaxGlobalType = 0;
  for (Module::const_global_iterator GV = M->global_begin(),E = M->global_end();
       GV != E; ++GV) {
    MaxAlignment = std::max(MaxAlignment, GV->getAlignment());
    MaxGlobalType = std::max(MaxGlobalType, VE.getTypeID(GV->getType()));
    
    if (!GV->hasSection()) continue;
    // Give section names unique ID's.
    unsigned &Entry = SectionMap[GV->getSection()];
    if (Entry != 0) continue;
    WriteStringRecord(bitc::MODULE_CODE_SECTIONNAME, GV->getSection(),
                      0/*TODO*/, Stream);
    Entry = SectionMap.size();
  }
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {
    MaxAlignment = std::max(MaxAlignment, F->getAlignment());
    if (!F->hasSection()) continue;
    // Give section names unique ID's.
    unsigned &Entry = SectionMap[F->getSection()];
    if (Entry != 0) continue;
    WriteStringRecord(bitc::MODULE_CODE_SECTIONNAME, F->getSection(),
                      0/*TODO*/, Stream);
    Entry = SectionMap.size();
  }
  
  // Emit abbrev for globals, now that we know # sections and max alignment.
  unsigned SimpleGVarAbbrev = 0;
  if (!M->global_empty()) { 
    // Add an abbrev for common globals with no visibility or thread localness.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::MODULE_CODE_GLOBALVAR));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::FixedWidth,
                              Log2_32_Ceil(MaxGlobalType+1)));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::FixedWidth, 1)); // Constant.
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));        // Initializer.
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::FixedWidth, 3)); // Linkage.
    if (MaxAlignment == 0)                                     // Alignment.
      Abbv->Add(BitCodeAbbrevOp(0));
    else {
      unsigned MaxEncAlignment = Log2_32(MaxAlignment)+1;
      Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::FixedWidth,
                               Log2_32_Ceil(MaxEncAlignment+1)));
    }
    if (SectionMap.empty())                                    // Section.
      Abbv->Add(BitCodeAbbrevOp(0));
    else
      Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::FixedWidth,
                               Log2_32_Ceil(SectionMap.size())));
    // Don't bother emitting vis + thread local.
    SimpleGVarAbbrev = Stream.EmitAbbrev(Abbv);
  }
  
  // Emit the global variable information.
  SmallVector<unsigned, 64> Vals;
  for (Module::const_global_iterator GV = M->global_begin(),E = M->global_end();
       GV != E; ++GV) {
    unsigned AbbrevToUse = 0;

    // GLOBALVAR: [type, isconst, initid, 
    //             linkage, alignment, section, visibility, threadlocal]
    Vals.push_back(VE.getTypeID(GV->getType()));
    Vals.push_back(GV->isConstant());
    Vals.push_back(GV->isDeclaration() ? 0 :
                   (VE.getValueID(GV->getInitializer()) + 1));
    Vals.push_back(getEncodedLinkage(GV));
    Vals.push_back(Log2_32(GV->getAlignment())+1);
    Vals.push_back(GV->hasSection() ? SectionMap[GV->getSection()] : 0);
    if (GV->isThreadLocal() || 
        GV->getVisibility() != GlobalValue::DefaultVisibility) {
      Vals.push_back(getEncodedVisibility(GV));
      Vals.push_back(GV->isThreadLocal());
    } else {
      AbbrevToUse = SimpleGVarAbbrev;
    }
    
    Stream.EmitRecord(bitc::MODULE_CODE_GLOBALVAR, Vals, AbbrevToUse);
    Vals.clear();
  }

  // Emit the function proto information.
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {
    // FUNCTION:  [type, callingconv, isproto, linkage, alignment, section,
    //             visibility]
    Vals.push_back(VE.getTypeID(F->getType()));
    Vals.push_back(F->getCallingConv());
    Vals.push_back(F->isDeclaration());
    Vals.push_back(getEncodedLinkage(F));
    Vals.push_back(Log2_32(F->getAlignment())+1);
    Vals.push_back(F->hasSection() ? SectionMap[F->getSection()] : 0);
    Vals.push_back(getEncodedVisibility(F));
    
    unsigned AbbrevToUse = 0;
    Stream.EmitRecord(bitc::MODULE_CODE_FUNCTION, Vals, AbbrevToUse);
    Vals.clear();
  }
}


/// WriteModule - Emit the specified module to the bitstream.
static void WriteModule(const Module *M, BitstreamWriter &Stream) {
  Stream.EnterSubblock(bitc::MODULE_BLOCK_ID, 3);
  
  // Emit the version number if it is non-zero.
  if (CurVersion) {
    SmallVector<unsigned, 1> VersionVals;
    VersionVals.push_back(CurVersion);
    Stream.EmitRecord(bitc::MODULE_CODE_VERSION, VersionVals);
  }
  
  // Analyze the module, enumerating globals, functions, etc.
  ValueEnumerator VE(M);
  
  // Emit information describing all of the types in the module.
  WriteTypeTable(VE, Stream);
  
  // FIXME: Emit constants.
  
  // Emit top-level description of module, including target triple, inline asm,
  // descriptors for global variables, and function prototype info.
  WriteModuleInfo(M, VE, Stream);
  
  // Emit the type symbol table information.
  WriteTypeSymbolTable(M->getTypeSymbolTable(), VE, Stream);
  Stream.ExitBlock();
}

/// WriteBitcodeToFile - Write the specified module to the specified output
/// stream.
void llvm::WriteBitcodeToFile(const Module *M, std::ostream &Out) {
  std::vector<unsigned char> Buffer;
  BitstreamWriter Stream(Buffer);
  
  Buffer.reserve(256*1024);
  
  // Emit the file header.
  Stream.Emit((unsigned)'B', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit(0x0, 4);
  Stream.Emit(0xC, 4);
  Stream.Emit(0xE, 4);
  Stream.Emit(0xD, 4);

  // Emit the module.
  WriteModule(M, Stream);
  
  // Write the generated bitstream to "Out".
  Out.write((char*)&Buffer.front(), Buffer.size());
}
