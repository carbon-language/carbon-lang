
//===-- PIC16DebugInfo.cpp - Implementation for PIC16 Debug Information ======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the helper functions for representing debug information.
//
//===----------------------------------------------------------------------===//

#include "PIC16.h"
#include "PIC16ABINames.h"
#include "PIC16DebugInfo.h" 
#include "llvm/GlobalVariable.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/DebugLoc.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

/// PopulateDebugInfo - Populate the TypeNo, Aux[] and TagName from Ty.
///
void PIC16DbgInfo::PopulateDebugInfo (DIType Ty, unsigned short &TypeNo,
                                      bool &HasAux, int Aux[], 
                                      std::string &TagName) {
  if (Ty.isBasicType())
    PopulateBasicTypeInfo (Ty, TypeNo);
  else if (Ty.isCompositeType())
    PopulateCompositeTypeInfo (Ty, TypeNo, HasAux, Aux, TagName);
  else if (Ty.isDerivedType())
    PopulateDerivedTypeInfo (Ty, TypeNo, HasAux, Aux, TagName);
  else {
    TypeNo = PIC16Dbg::T_NULL;
    HasAux = false;
  }
  return;
}

/// PopulateBasicTypeInfo- Populate TypeNo for basic type from Ty.
///
void PIC16DbgInfo::PopulateBasicTypeInfo (DIType Ty, unsigned short &TypeNo) {
  std::string Name = Ty.getName();
  unsigned short BaseTy = GetTypeDebugNumber(Name);
  TypeNo = TypeNo << PIC16Dbg::S_BASIC;
  TypeNo = TypeNo | (0xffff & BaseTy);
}

/// PopulateDerivedTypeInfo - Populate TypeNo, Aux[], TagName for derived type 
/// from Ty. Derived types are mostly pointers.
///
void PIC16DbgInfo::PopulateDerivedTypeInfo (DIType Ty, unsigned short &TypeNo,
                                            bool &HasAux, int Aux[],
                                            std::string &TagName) {

  switch(Ty.getTag())
  {
    case dwarf::DW_TAG_pointer_type:
      TypeNo = TypeNo << PIC16Dbg::S_DERIVED;
      TypeNo = TypeNo | PIC16Dbg::DT_PTR;
      break;
    default:
      TypeNo = TypeNo << PIC16Dbg::S_DERIVED;
  }
  
  // We also need to encode the information about the base type of
  // pointer in TypeNo.
  DIType BaseType = DIDerivedType(Ty.getNode()).getTypeDerivedFrom();
  PopulateDebugInfo(BaseType, TypeNo, HasAux, Aux, TagName);
}

/// PopulateArrayTypeInfo - Populate TypeNo, Aux[] for array from Ty.
void PIC16DbgInfo::PopulateArrayTypeInfo (DIType Ty, unsigned short &TypeNo,
                                          bool &HasAux, int Aux[],
                                          std::string &TagName) {

  DICompositeType CTy = DICompositeType(Ty.getNode());
  DIArray Elements = CTy.getTypeArray();
  unsigned short size = 1;
  unsigned short Dimension[4]={0,0,0,0};
  for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Elements.getElement(i);
    if (Element.getTag() == dwarf::DW_TAG_subrange_type) {
      TypeNo = TypeNo << PIC16Dbg::S_DERIVED;
      TypeNo = TypeNo | PIC16Dbg::DT_ARY;
      DISubrange SubRange = DISubrange(Element.getNode());
      Dimension[i] = SubRange.getHi() - SubRange.getLo() + 1;
      // Each dimension is represented by 2 bytes starting at byte 9.
      Aux[8+i*2+0] = Dimension[i];
      Aux[8+i*2+1] = Dimension[i] >> 8;
      size = size * Dimension[i];
    }
  }
  HasAux = true;
  // In auxillary entry for array, 7th and 8th byte represent array size.
  Aux[6] = size & 0xff;
  Aux[7] = size >> 8;
  DIType BaseType = CTy.getTypeDerivedFrom();
  PopulateDebugInfo(BaseType, TypeNo, HasAux, Aux, TagName);
}

/// PopulateStructOrUnionTypeInfo - Populate TypeNo, Aux[] , TagName for 
/// structure or union.
///
void PIC16DbgInfo::PopulateStructOrUnionTypeInfo (DIType Ty, 
                                                  unsigned short &TypeNo,
                                                  bool &HasAux, int Aux[],
                                                  std::string &TagName) {
  DICompositeType CTy = DICompositeType(Ty.getNode());
  TypeNo = TypeNo << PIC16Dbg::S_BASIC;
  if (Ty.getTag() == dwarf::DW_TAG_structure_type)
    TypeNo = TypeNo | PIC16Dbg::T_STRUCT;
  else
    TypeNo = TypeNo | PIC16Dbg::T_UNION;
  TagName = CTy.getName();
  // UniqueSuffix is .number where number is obtained from
  // llvm.dbg.composite<number>.
  // FIXME: This will break when composite type is not represented by
  // llvm.dbg.composite* global variable. Since we need to revisit 
  // PIC16DebugInfo implementation anyways after the MDNodes based 
  // framework is done, let us continue with the way it is.
  std::string UniqueSuffix = "." + Ty.getNode()->getNameStr().substr(18);
  TagName += UniqueSuffix;
  unsigned short size = CTy.getSizeInBits()/8;
  // 7th and 8th byte represent size.
  HasAux = true;
  Aux[6] = size & 0xff;
  Aux[7] = size >> 8;
}

/// PopulateEnumTypeInfo - Populate TypeNo for enum from Ty.
void PIC16DbgInfo::PopulateEnumTypeInfo (DIType Ty, unsigned short &TypeNo) {
  TypeNo = TypeNo << PIC16Dbg::S_BASIC;
  TypeNo = TypeNo | PIC16Dbg::T_ENUM;
}

/// PopulateCompositeTypeInfo - Populate TypeNo, Aux[] and TagName for 
/// composite types from Ty.
///
void PIC16DbgInfo::PopulateCompositeTypeInfo (DIType Ty, unsigned short &TypeNo,
                                              bool &HasAux, int Aux[],
                                              std::string &TagName) {
  switch (Ty.getTag()) {
    case dwarf::DW_TAG_array_type: {
      PopulateArrayTypeInfo (Ty, TypeNo, HasAux, Aux, TagName);
      break;
    }
    case dwarf:: DW_TAG_union_type:
    case dwarf::DW_TAG_structure_type: {
      PopulateStructOrUnionTypeInfo (Ty, TypeNo, HasAux, Aux, TagName);
      break;
    }
    case dwarf::DW_TAG_enumeration_type: {
      PopulateEnumTypeInfo (Ty, TypeNo);
      break;
    }
    default:
      TypeNo = TypeNo << PIC16Dbg::S_DERIVED;
  }
}

/// GetTypeDebugNumber - Get debug type number for given type.
///
unsigned PIC16DbgInfo::GetTypeDebugNumber(std::string &type)  {
  if (type == "char")
    return PIC16Dbg::T_CHAR;
  else if (type == "short")
    return PIC16Dbg::T_SHORT;
  else if (type == "int")
    return PIC16Dbg::T_INT;
  else if (type == "long")
    return PIC16Dbg::T_LONG;
  else if (type == "unsigned char")
    return PIC16Dbg::T_UCHAR;
  else if (type == "unsigned short")
    return PIC16Dbg::T_USHORT;
  else if (type == "unsigned int")
    return PIC16Dbg::T_UINT;
  else if (type == "unsigned long")
    return PIC16Dbg::T_ULONG;
  else
    return 0;
}
 
/// GetStorageClass - Get storage class for give debug variable.
///
short PIC16DbgInfo::getStorageClass(DIGlobalVariable DIGV) {
  short ClassNo;
  if (PAN::isLocalName(DIGV.getName())) {
    // Generating C_AUTO here fails due to error in linker. Change it once
    // linker is fixed.
    ClassNo = PIC16Dbg::C_STAT;
  }
  else if (DIGV.isLocalToUnit())
    ClassNo = PIC16Dbg::C_STAT;
  else
    ClassNo = PIC16Dbg::C_EXT;
  return ClassNo;
}

/// BeginModule - Emit necessary debug info to start a Module and do other
/// required initializations.
void PIC16DbgInfo::BeginModule(Module &M) {
  // Emit file directive for module.
  DebugInfoFinder DbgFinder;
  DbgFinder.processModule(M);
  if (DbgFinder.compile_unit_count() != 0) {
    // FIXME : What if more then one CUs are present in a module ?
    MDNode *CU = *DbgFinder.compile_unit_begin();
    EmitDebugDirectives = true;
    SwitchToCU(CU);
  }
  // Emit debug info for decls of composite types.
  EmitCompositeTypeDecls(M);
}

/// Helper to find first valid debug loc for a function.
///
static const DebugLoc GetDebugLocForFunction(const MachineFunction &MF) {
  DebugLoc DL;
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      DL = II->getDebugLoc();
      if (!DL.isUnknown())
        return DL;
    }
  }
  return DL;
}

/// BeginFunction - Emit necessary debug info to start a function.
///
void PIC16DbgInfo::BeginFunction(const MachineFunction &MF) {
  if (! EmitDebugDirectives) return;
  
  // Retreive the first valid debug Loc and process it.
  const DebugLoc &DL = GetDebugLocForFunction(MF);
  // Emit debug info only if valid debug info is available.
  if (!DL.isUnknown()) {
    ChangeDebugLoc(MF, DL, true);
    EmitFunctBeginDI(MF.getFunction());
  } 
  // Set current line to 0 so that.line directive is genearted after .bf.
  CurLine = 0;
}

/// ChangeDebugLoc - Take necessary steps when DebugLoc changes.
/// CurFile and CurLine may change as a result of this.
///
void PIC16DbgInfo::ChangeDebugLoc(const MachineFunction &MF,  
                                  const DebugLoc &DL, bool IsInBeginFunction) {
  if (!EmitDebugDirectives) return;
  assert(!DL.isUnknown() && "can't change to invalid debug loc");

  SwitchToCU(DL.getScope(MF.getFunction()->getContext()));
  SwitchToLine(DL.getLine(), IsInBeginFunction);
}

/// SwitchToLine - Emit line directive for a new line.
///
void PIC16DbgInfo::SwitchToLine(unsigned Line, bool IsInBeginFunction) {
  if (CurLine == Line) return;
  if (!IsInBeginFunction)
    OS.EmitRawText("\n\t.line " + Twine(Line));
  CurLine = Line;
}

/// EndFunction - Emit .ef for end of function.
///
void PIC16DbgInfo::EndFunction(const MachineFunction &MF) {
  if (! EmitDebugDirectives) return;
  const DebugLoc &DL = GetDebugLocForFunction(MF);
  // Emit debug info only if valid debug info is available.
  if (!DL.isUnknown())
    EmitFunctEndDI(MF.getFunction(), CurLine);
}

/// EndModule - Emit .eof for end of module.
///
void PIC16DbgInfo::EndModule(Module &M) {
  if (! EmitDebugDirectives) return;
  EmitVarDebugInfo(M);
  if (CurFile != "") OS.EmitRawText(StringRef("\n\t.eof"));
}
 
/// EmitCompositeTypeElements - Emit debug information for members of a 
/// composite type.
/// 
void PIC16DbgInfo::EmitCompositeTypeElements (DICompositeType CTy,
                                              std::string SuffixNo) {
  unsigned long Value = 0;
  DIArray Elements = CTy.getTypeArray();
  for (unsigned i = 0, N = Elements.getNumElements(); i < N; i++) {
    DIDescriptor Element = Elements.getElement(i);
    unsigned short TypeNo = 0;
    bool HasAux = false;
    int ElementAux[PIC16Dbg::AuxSize] = { 0 };
    std::string TagName = "";
    DIDerivedType DITy(Element.getNode());
    unsigned short ElementSize = DITy.getSizeInBits()/8;
    // Get mangleddd name for this structure/union  element.
    std::string MangMemName = DITy.getName().str() + SuffixNo;
    PopulateDebugInfo(DITy, TypeNo, HasAux, ElementAux, TagName);
    short Class = 0;
    if( CTy.getTag() == dwarf::DW_TAG_union_type)
      Class = PIC16Dbg::C_MOU;
    else if  (CTy.getTag() == dwarf::DW_TAG_structure_type)
      Class = PIC16Dbg::C_MOS;
    EmitSymbol(MangMemName.c_str(), Class, TypeNo, Value);
    if (CTy.getTag() == dwarf::DW_TAG_structure_type)
      Value += ElementSize;
    if (HasAux)
      EmitAuxEntry(MangMemName.c_str(), ElementAux, PIC16Dbg::AuxSize, TagName);
  }
}

/// EmitCompositeTypeDecls - Emit composite type declarations like structure 
/// and union declarations.
///
void PIC16DbgInfo::EmitCompositeTypeDecls(Module &M) {
  DebugInfoFinder DbgFinder;
  DbgFinder.processModule(M);
  for (DebugInfoFinder::iterator I = DbgFinder.type_begin(),
         E = DbgFinder.type_end(); I != E; ++I) {
    DICompositeType CTy(*I);
    if (!CTy.Verify())
      continue;
    if (CTy.getTag() == dwarf::DW_TAG_union_type ||
        CTy.getTag() == dwarf::DW_TAG_structure_type ) {
      // Get the number after llvm.dbg.composite and make UniqueSuffix from 
      // it.
      std::string DIVar = CTy.getNode()->getNameStr();
      std::string UniqueSuffix = "." + DIVar.substr(18);
      std::string MangledCTyName = CTy.getName().str() + UniqueSuffix;
      unsigned short size = CTy.getSizeInBits()/8;
      int Aux[PIC16Dbg::AuxSize] = {0};
      // 7th and 8th byte represent size of structure/union.
      Aux[6] = size & 0xff;
      Aux[7] = size >> 8;
      // Emit .def for structure/union tag.
      if( CTy.getTag() == dwarf::DW_TAG_union_type)
        EmitSymbol(MangledCTyName.c_str(), PIC16Dbg::C_UNTAG);
      else if  (CTy.getTag() == dwarf::DW_TAG_structure_type) 
        EmitSymbol(MangledCTyName.c_str(), PIC16Dbg::C_STRTAG);
      
      // Emit auxiliary debug information for structure/union tag. 
      EmitAuxEntry(MangledCTyName.c_str(), Aux, PIC16Dbg::AuxSize);
      
      // Emit members.
      EmitCompositeTypeElements (CTy, UniqueSuffix);
      
      // Emit mangled Symbol for end of structure/union.
      std::string EOSSymbol = ".eos" + UniqueSuffix;
      EmitSymbol(EOSSymbol.c_str(), PIC16Dbg::C_EOS);
      EmitAuxEntry(EOSSymbol.c_str(), Aux, PIC16Dbg::AuxSize, 
                   MangledCTyName.c_str());
    }
  }
}


/// EmitFunctBeginDI - Emit .bf for function.
///
void PIC16DbgInfo::EmitFunctBeginDI(const Function *F) {
  std::string FunctName = F->getName();
  if (EmitDebugDirectives) {
    std::string FunctBeginSym = ".bf." + FunctName;
    std::string BlockBeginSym = ".bb." + FunctName;

    int BFAux[PIC16Dbg::AuxSize] = {0};
    BFAux[4] = CurLine;
    BFAux[5] = CurLine >> 8;

    // Emit debug directives for beginning of function.
    EmitSymbol(FunctBeginSym, PIC16Dbg::C_FCN);
    EmitAuxEntry(FunctBeginSym, BFAux, PIC16Dbg::AuxSize);

    EmitSymbol(BlockBeginSym, PIC16Dbg::C_BLOCK);
    EmitAuxEntry(BlockBeginSym, BFAux, PIC16Dbg::AuxSize);
  }
}

/// EmitFunctEndDI - Emit .ef for function end.
///
void PIC16DbgInfo::EmitFunctEndDI(const Function *F, unsigned Line) {
  std::string FunctName = F->getName();
  if (EmitDebugDirectives) {
    std::string FunctEndSym = ".ef." + FunctName;
    std::string BlockEndSym = ".eb." + FunctName;

    // Emit debug directives for end of function.
    EmitSymbol(BlockEndSym, PIC16Dbg::C_BLOCK);
    int EFAux[PIC16Dbg::AuxSize] = {0};
    // 5th and 6th byte stand for line number.
    EFAux[4] = CurLine;
    EFAux[5] = CurLine >> 8;
    EmitAuxEntry(BlockEndSym, EFAux, PIC16Dbg::AuxSize);
    EmitSymbol(FunctEndSym, PIC16Dbg::C_FCN);
    EmitAuxEntry(FunctEndSym, EFAux, PIC16Dbg::AuxSize);
  }
}

/// EmitAuxEntry - Emit Auxiliary debug information.
///
void PIC16DbgInfo::EmitAuxEntry(const std::string VarName, int Aux[], int Num,
                                std::string TagName) {
  std::string Tmp;
  // TagName is emitted in case of structure/union objects.
  if (!TagName.empty()) Tmp += ", " + TagName;
  
  for (int i = 0; i<Num; i++)
    Tmp += "," + utostr(Aux[i] && 0xff);
  
  OS.EmitRawText("\n\t.dim " + Twine(VarName) + ", 1" + Tmp);
}

/// EmitSymbol - Emit .def for a symbol. Value is offset for the member.
///
void PIC16DbgInfo::EmitSymbol(std::string Name, short Class,
                              unsigned short Type, unsigned long Value) {
  std::string Tmp;
  if (Value > 0)
    Tmp = ", value = " + utostr(Value);
  
  OS.EmitRawText("\n\t.def " + Twine(Name) + ", type = " + utostr(Type) +
                 ", class = " + utostr(Class) + Tmp);
}

/// EmitVarDebugInfo - Emit debug information for all variables.
///
void PIC16DbgInfo::EmitVarDebugInfo(Module &M) {
  DebugInfoFinder DbgFinder;
  DbgFinder.processModule(M);
  
  for (DebugInfoFinder::iterator I = DbgFinder.global_variable_begin(),
         E = DbgFinder.global_variable_end(); I != E; ++I) {
    DIGlobalVariable DIGV(*I);
    DIType Ty = DIGV.getType();
    unsigned short TypeNo = 0;
    bool HasAux = false;
    int Aux[PIC16Dbg::AuxSize] = { 0 };
    std::string TagName = "";
    std::string VarName = DIGV.getName();
    VarName = MAI->getGlobalPrefix() + VarName;
    PopulateDebugInfo(Ty, TypeNo, HasAux, Aux, TagName);
    // Emit debug info only if type information is availaible.
    if (TypeNo != PIC16Dbg::T_NULL) {
      OS.EmitRawText("\t.type " + Twine(VarName) + ", " + Twine(TypeNo));
      short ClassNo = getStorageClass(DIGV);
      OS.EmitRawText("\t.class " + Twine(VarName) + ", " + Twine(ClassNo));
      if (HasAux)
        EmitAuxEntry(VarName, Aux, PIC16Dbg::AuxSize, TagName);
    }
  }
}

/// SwitchToCU - Switch to a new compilation unit.
///
void PIC16DbgInfo::SwitchToCU(MDNode *CU) {
  // Get the file path from CU.
  DICompileUnit cu(CU);
  std::string DirName = cu.getDirectory();
  std::string FileName = cu.getFilename();
  std::string FilePath = DirName + "/" + FileName;

  // Nothing to do if source file is still same.
  if ( FilePath == CurFile ) return;

  // Else, close the current one and start a new.
  if (CurFile != "")
    OS.EmitRawText(StringRef("\t.eof"));
  OS.EmitRawText("\n\t.file\t\"" + Twine(FilePath) + "\"");
  CurFile = FilePath;
  CurLine = 0;
}

/// EmitEOF - Emit .eof for end of file.
///
void PIC16DbgInfo::EmitEOF() {
  if (CurFile != "")
    OS.EmitRawText(StringRef("\t.EOF"));
}

