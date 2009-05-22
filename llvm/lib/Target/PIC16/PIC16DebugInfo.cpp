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
        Aux[6] = size;
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
        unsigned size = CTy.getSizeInBits()/8;
        // 7th and 8th byte represent size.   
        HasAux = true;
        Aux[6] = size;
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
