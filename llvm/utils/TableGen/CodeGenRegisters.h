//===- CodeGenRegisters.h - Register and RegisterClass Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate information gleaned from the
// target register and register class definitions.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_REGISTERS_H
#define CODEGEN_REGISTERS_H

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/ADT/DenseMap.h"
#include <string>
#include <vector>
#include <set>
#include <cstdlib>

namespace llvm {
  class Record;

  /// CodeGenRegister - Represents a register definition.
  struct CodeGenRegister {
    Record *TheDef;
    const std::string &getName() const;
    unsigned EnumValue;
    unsigned CostPerUse;
    CodeGenRegister(Record *R);
  };


  struct CodeGenRegisterClass {
    Record *TheDef;
    std::string Namespace;
    std::vector<Record*> Elements;
    std::vector<MVT::SimpleValueType> VTs;
    unsigned SpillSize;
    unsigned SpillAlignment;
    int CopyCost;
    bool Allocatable;
    // Map SubRegIndex -> RegisterClass
    DenseMap<Record*,Record*> SubRegClasses;
    std::string MethodProtos, MethodBodies;

    const std::string &getName() const;
    const std::vector<MVT::SimpleValueType> &getValueTypes() const {return VTs;}
    unsigned getNumValueTypes() const { return VTs.size(); }

    MVT::SimpleValueType getValueTypeNum(unsigned VTNum) const {
      if (VTNum < VTs.size())
        return VTs[VTNum];
      assert(0 && "VTNum greater than number of ValueTypes in RegClass!");
      abort();
    }

    bool containsRegister(Record *R) const {
      for (unsigned i = 0, e = Elements.size(); i != e; ++i)
        if (Elements[i] == R) return true;
      return false;
    }

    // Returns true if RC is a strict subclass.
    // RC is a sub-class of this class if it is a valid replacement for any
    // instruction operand where a register of this classis required. It must
    // satisfy these conditions:
    //
    // 1. All RC registers are also in this.
    // 2. The RC spill size must not be smaller than our spill size.
    // 3. RC spill alignment must be compatible with ours.
    //
    bool hasSubClass(const CodeGenRegisterClass *RC) const {

      if (RC->Elements.size() > Elements.size() ||
          (SpillAlignment && RC->SpillAlignment % SpillAlignment) ||
          SpillSize > RC->SpillSize)
        return false;

      std::set<Record*> RegSet;
      for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
        Record *Reg = Elements[i];
        RegSet.insert(Reg);
      }

      for (unsigned i = 0, e = RC->Elements.size(); i != e; ++i) {
        Record *Reg = RC->Elements[i];
        if (!RegSet.count(Reg))
          return false;
      }

      return true;
    }

    CodeGenRegisterClass(Record *R);
  };
}

#endif
