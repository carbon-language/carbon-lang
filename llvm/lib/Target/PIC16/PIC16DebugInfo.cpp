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
#include "PIC16DebugInfo.h" 
#include "llvm/GlobalVariable.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void PIC16DbgInfo::PopulateDebugInfo(DIType Ty, unsigned short &TypeNo,
                                     bool &HasAux, int Aux[], 
                                     std::string &TypeName) {
  if (Ty.isBasicType(Ty.getTag())) {
    std::string Name = "";
    Ty.getName(Name);
    unsigned short BaseTy = GetTypeDebugNumber(Name);
    TypeNo = TypeNo << PIC16Dbg::S_BASIC;
    TypeNo = TypeNo | (0xffff & BaseTy);
  }
  else if (Ty.isDerivedType(Ty.getTag())) {
    switch(Ty.getTag())
    {
      case dwarf::DW_TAG_pointer_type:
        TypeNo = TypeNo << PIC16Dbg::S_DERIVED;
        TypeNo = TypeNo | PIC16Dbg::DT_PTR;
        break;
      default:
        TypeNo = TypeNo << PIC16Dbg::S_DERIVED;
    }
    DIType BaseType = DIDerivedType(Ty.getGV()).getTypeDerivedFrom();
    PopulateDebugInfo(BaseType, TypeNo, HasAux, Aux, TypeName);
  }
  else if (Ty.isCompositeType(Ty.getTag())) {
    switch (Ty.getTag()) {
      case dwarf::DW_TAG_array_type: {
        DICompositeType CTy = DICompositeType(Ty.getGV());
        DIArray Elements = CTy.getTypeArray();
        unsigned short size = 1;
        unsigned short Dimension[4]={0,0,0,0};
        for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
          DIDescriptor Element = Elements.getElement(i);
          if (Element.getTag() == dwarf::DW_TAG_subrange_type) {
            TypeNo = TypeNo << PIC16Dbg::S_DERIVED;
            TypeNo = TypeNo | PIC16Dbg::DT_ARY;
            DISubrange SubRange = DISubrange(Element.getGV());
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
        PopulateDebugInfo(BaseType, TypeNo, HasAux, Aux, TypeName);

        break;
      }
      case dwarf:: DW_TAG_union_type:
      case dwarf::DW_TAG_structure_type: {
        DICompositeType CTy = DICompositeType(Ty.getGV());
        TypeNo = TypeNo << PIC16Dbg::S_BASIC;
        if (Ty.getTag() == dwarf::DW_TAG_structure_type)
          TypeNo = TypeNo | PIC16Dbg::T_STRUCT;
        else
          TypeNo = TypeNo | PIC16Dbg::T_UNION;
        CTy.getName(TypeName);
        // UniqueSuffix is .number where number is obtained from 
        // llvm.dbg.composite<number>.
        std::string UniqueSuffix = "." + Ty.getGV()->getName().substr(18);
        TypeName += UniqueSuffix;
        unsigned short size = CTy.getSizeInBits()/8;
        // 7th and 8th byte represent size.   
        HasAux = true;
        Aux[6] = size & 0xff;
        Aux[7] = size >> 8;
        break;
      }
      case dwarf::DW_TAG_enumeration_type: {
        TypeNo = TypeNo << PIC16Dbg::S_BASIC;
        TypeNo = TypeNo | PIC16Dbg::T_ENUM;
        break;
      }
      default:
        TypeNo = TypeNo << PIC16Dbg::S_DERIVED;
    }
  }
  else {
    TypeNo = PIC16Dbg::T_NULL;
    HasAux = false;
  }
  return;
}


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

short PIC16DbgInfo::getClass(DIGlobalVariable DIGV) {
  short ClassNo;
  if (PAN::isLocalName(DIGV.getGlobal()->getName())) {
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

void PIC16DbgInfo::Init(Module &M) {
  // Do all debug related initializations here.
  EmitFileDirective(M);
  EmitCompositeTypeDecls(M);
}

void PIC16DbgInfo::EmitCompositeTypeDecls(Module &M) {
  for(iplist<GlobalVariable>::iterator I = M.getGlobalList().begin(),
      E = M.getGlobalList().end(); I != E; I++) {
    // Structures and union declaration's debug info has llvm.dbg.composite
    // in its name.
    if(I->getName().find("llvm.dbg.composite") != std::string::npos) {
      GlobalVariable *GV = cast<GlobalVariable >(I);
      DICompositeType CTy(GV);
      if (CTy.getTag() == dwarf::DW_TAG_union_type ||
          CTy.getTag() == dwarf::DW_TAG_structure_type ) {
        std::string name;
        CTy.getName(name);
        std::string DIVar = I->getName();
        // Get the number after llvm.dbg.composite and make UniqueSuffix from 
        // it.
        std::string UniqueSuffix = "." + DIVar.substr(18);
        std::string MangledCTyName = name + UniqueSuffix;
        unsigned short size = CTy.getSizeInBits()/8;
        int Aux[PIC16Dbg::AuxSize] = {0};
        // 7th and 8th byte represent size of structure/union.
        Aux[6] = size & 0xff;
        Aux[7] = size >> 8;
        // Emit .def for structure/union tag.
        if( CTy.getTag() == dwarf::DW_TAG_union_type)
          EmitSymbol(MangledCTyName, PIC16Dbg::C_UNTAG);
        else if  (CTy.getTag() == dwarf::DW_TAG_structure_type) 
          EmitSymbol(MangledCTyName, PIC16Dbg::C_STRTAG);

        // Emit auxiliary debug information for structure/union tag. 
        EmitAuxEntry(MangledCTyName, Aux, PIC16Dbg::AuxSize);
        unsigned long Value = 0;
        DIArray Elements = CTy.getTypeArray();
        for (unsigned i = 0, N = Elements.getNumElements(); i < N; i++) {
          DIDescriptor Element = Elements.getElement(i);
          unsigned short TypeNo = 0;
          bool HasAux = false;
          int ElementAux[PIC16Dbg::AuxSize] = { 0 };
          std::string TypeName = "";
          std::string ElementName;
          GlobalVariable *GV = Element.getGV();
          DIDerivedType DITy(GV);
          DITy.getName(ElementName);
          unsigned short ElementSize = DITy.getSizeInBits()/8;
          // Get mangleddd name for this structure/union  element.
          std::string MangMemName = ElementName + UniqueSuffix;
	  PopulateDebugInfo(DITy, TypeNo, HasAux, ElementAux, TypeName);
          short Class;
          if( CTy.getTag() == dwarf::DW_TAG_union_type)
            Class = PIC16Dbg::C_MOU;
          else if  (CTy.getTag() == dwarf::DW_TAG_structure_type)
            Class = PIC16Dbg::C_MOS;
          EmitSymbol(MangMemName, Class, TypeNo, Value);
          if (CTy.getTag() == dwarf::DW_TAG_structure_type)
            Value += ElementSize;
          if (HasAux)
            EmitAuxEntry(MangMemName, ElementAux, PIC16Dbg::AuxSize, TypeName);
        }
        // Emit mangled Symbol for end of structure/union.
        std::string EOSSymbol = ".eos" + UniqueSuffix;
        EmitSymbol(EOSSymbol, PIC16Dbg::C_EOS);
        EmitAuxEntry(EOSSymbol, Aux, PIC16Dbg::AuxSize, MangledCTyName);
      }
    }
  }
}

void PIC16DbgInfo::EmitFunctBeginDI(const Function *F) {
  std::string FunctName = F->getName();
  if (EmitDebugDirectives) {
    std::string FunctBeginSym = ".bf." + FunctName;
    std::string BlockBeginSym = ".bb." + FunctName;

    int BFAux[PIC16Dbg::AuxSize] = {0};
    BFAux[4] = FunctBeginLine;
    BFAux[5] = FunctBeginLine >> 8;
    // Emit debug directives for beginning of function.
    EmitSymbol(FunctBeginSym, PIC16Dbg::C_FCN);
    EmitAuxEntry(FunctBeginSym, BFAux, PIC16Dbg::AuxSize);
    EmitSymbol(BlockBeginSym, PIC16Dbg::C_BLOCK);
    EmitAuxEntry(BlockBeginSym, BFAux, PIC16Dbg::AuxSize);
  }
}

void PIC16DbgInfo::EmitFunctEndDI(const Function *F, unsigned Line) {
  std::string FunctName = F->getName();
  if (EmitDebugDirectives) {
    std::string FunctEndSym = ".ef." + FunctName;
    std::string BlockEndSym = ".eb." + FunctName;

    // Emit debug directives for end of function.
    EmitSymbol(BlockEndSym, PIC16Dbg::C_BLOCK);
    int EFAux[PIC16Dbg::AuxSize] = {0};
    // 5th and 6th byte stand for line number.
    EFAux[4] = Line;
    EFAux[5] = Line >> 8;
    EmitAuxEntry(BlockEndSym, EFAux, PIC16Dbg::AuxSize);
    EmitSymbol(FunctEndSym, PIC16Dbg::C_FCN);
    EmitAuxEntry(FunctEndSym, EFAux, PIC16Dbg::AuxSize);
  }
}

/// EmitAuxEntry - Emit Auxiliary debug information.
///
void PIC16DbgInfo::EmitAuxEntry(const std::string VarName, int Aux[], int num,
                                std::string tag) {
  O << "\n\t.dim " << VarName << ", 1" ;
  if (tag != "")
    O << ", " << tag;
  for (int i = 0; i<num; i++)
    O << "," << Aux[i];
}

void PIC16DbgInfo::EmitSymbol(std::string Name, short Class, unsigned short
                              Type, unsigned long Value) {
  O << "\n\t" << ".def "<< Name << ", type = " << Type << ", class = " 
    << Class;
  if (Value > 0)
    O  << ", value = " << Value;
}

void PIC16DbgInfo::EmitVarDebugInfo(Module &M) {
  GlobalVariable *Root = M.getGlobalVariable("llvm.dbg.global_variables");
  if (!Root)
    return;

  Constant *RootC = cast<Constant>(*Root->use_begin());
  for (Value::use_iterator UI = RootC->use_begin(), UE = Root->use_end();
       UI != UE; ++UI) {
    for (Value::use_iterator UUI = UI->use_begin(), UUE = UI->use_end();
         UUI != UUE; ++UUI) {
      DIGlobalVariable DIGV(cast<GlobalVariable>(*UUI));
      DIType Ty = DIGV.getType();
      unsigned short TypeNo = 0;
      bool HasAux = false;
      int Aux[PIC16Dbg::AuxSize] = { 0 };
      std::string TypeName = "";
      std::string VarName = TAI->getGlobalPrefix()+DIGV.getGlobal()->getName();
      PopulateDebugInfo(Ty, TypeNo, HasAux, Aux, TypeName);
      // Emit debug info only if type information is availaible.
      if (TypeNo != PIC16Dbg::T_NULL) {
        O << "\n\t.type " << VarName << ", " << TypeNo;
        short ClassNo = getClass(DIGV);
        O << "\n\t.class " << VarName << ", " << ClassNo;
        if (HasAux) 
          EmitAuxEntry(VarName, Aux, PIC16Dbg::AuxSize, TypeName);
      }
    }
  }
  O << "\n";
}

void PIC16DbgInfo::EmitFileDirective(Module &M) {
  GlobalVariable *CU = M.getNamedGlobal("llvm.dbg.compile_unit");
  if (CU) {
    EmitDebugDirectives = true;
    EmitFileDirective(CU, false);
  }
}

void PIC16DbgInfo::EmitFileDirective(GlobalVariable *CU, bool EmitEof) {
  std::string Dir, FN;
  DICompileUnit DIUnit(CU);
  std::string File = DIUnit.getDirectory(Dir) + "/" + DIUnit.getFilename(FN);
  if ( File != CurFile ) {
    if (EmitEof)
      EmitEOF();
    O << "\n\t.file\t\"" << File << "\"\n" ;
    CurFile = File;
  }
}

void PIC16DbgInfo::EmitEOF() {
  if (CurFile != "")
    O << "\n\t.EOF";
}

void PIC16DbgInfo::SetFunctBeginLine(unsigned line) {
  FunctBeginLine = line;
}
