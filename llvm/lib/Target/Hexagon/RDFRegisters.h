//===--- RDFRegisters.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_RDFREGISTERS_H
#define LLVM_LIB_TARGET_HEXAGON_RDFREGISTERS_H

#include "llvm/ADT/BitVector.h"
#include "llvm/Target/TargetRegisterInfo.h"

#include <unordered_map>
#include <vector>

namespace llvm {
namespace rdf {

  typedef uint32_t RegisterId;

  struct RegisterRef {
    RegisterId Reg = 0;
    LaneBitmask Mask = LaneBitmask::getNone();

    RegisterRef() = default;
    explicit RegisterRef(RegisterId R, LaneBitmask M = LaneBitmask::getAll())
      : Reg(R), Mask(R != 0 ? M : LaneBitmask::getNone()) {}

    operator bool() const {
      return Reg != 0 && Mask.any();
    }
    bool operator== (const RegisterRef &RR) const {
      return Reg == RR.Reg && Mask == RR.Mask;
    }
    bool operator!= (const RegisterRef &RR) const {
      return !operator==(RR);
    }
    bool operator< (const RegisterRef &RR) const {
      return Reg < RR.Reg || (Reg == RR.Reg && Mask < RR.Mask);
    }
  };


  struct PhysicalRegisterInfo {
    PhysicalRegisterInfo(const TargetRegisterInfo &tri,
                         const MachineFunction &mf);

    bool alias(RegisterRef RA, RegisterRef RB) const;

    const TargetRegisterInfo &getTRI() const { return TRI; }

//  private:
    struct RegInfo {
      unsigned MaxSuper = 0;
      const TargetRegisterClass *RegClass = nullptr;
    };

    const TargetRegisterInfo &TRI;
    std::vector<RegInfo> RegInfos;
  };


  struct RegisterAggr {
    RegisterAggr(const PhysicalRegisterInfo &pri)
        : ExpAliasUnits(pri.getTRI().getNumRegUnits()), PRI(pri) {}
    RegisterAggr(const RegisterAggr &RG) = default;

    bool empty() const { return Masks.empty(); }
    bool hasAliasOf(RegisterRef RR) const;
    bool hasCoverOf(RegisterRef RR) const;
    static bool isCoverOf(RegisterRef RA, RegisterRef RB,
                          const PhysicalRegisterInfo &PRI) {
      return RegisterAggr(PRI).insert(RA).hasCoverOf(RB);
    }

    RegisterAggr &insert(RegisterRef RR);
    RegisterAggr &insert(const RegisterAggr &RG);
    RegisterAggr &clear(RegisterRef RR);
    RegisterAggr &clear(const RegisterAggr &RG);

    RegisterRef clearIn(RegisterRef RR) const;

    void print(raw_ostream &OS) const;

  private:
    typedef std::unordered_map<RegisterId, LaneBitmask> MapType;

  public:
    typedef MapType::const_iterator iterator;
    iterator begin() const { return Masks.begin(); }
    iterator end() const { return Masks.end(); }
    RegisterRef normalize(RegisterRef RR) const;

  private:
    MapType Masks;
    BitVector ExpAliasUnits; // Register units for explicit aliases.
    bool CheckUnits = false;
    const PhysicalRegisterInfo &PRI;
  };

  // Optionally print the lane mask, if it is not ~0.
  struct PrintLaneMaskOpt {
    PrintLaneMaskOpt(LaneBitmask M) : Mask(M) {}
    LaneBitmask Mask;
  };
  raw_ostream &operator<< (raw_ostream &OS, const PrintLaneMaskOpt &P);

} // namespace rdf
} // namespace llvm

#endif

