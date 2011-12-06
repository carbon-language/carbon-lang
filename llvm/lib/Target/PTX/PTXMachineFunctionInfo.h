//===- PTXMachineFuctionInfo.h - PTX machine function info -------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares PTX-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_MACHINE_FUNCTION_INFO_H
#define PTX_MACHINE_FUNCTION_INFO_H

#include "PTX.h"
#include "PTXParamManager.h"
#include "PTXRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// PTXMachineFunctionInfo - This class is derived from MachineFunction and
/// contains private PTX target-specific information for each MachineFunction.
///
class PTXMachineFunctionInfo : public MachineFunctionInfo {
private:
  bool IsKernel;
  DenseSet<unsigned> RegArgs;
  DenseSet<unsigned> RegRets;

  typedef DenseMap<int, std::string> FrameMap;

  FrameMap FrameSymbols;

  struct RegisterInfo {
    unsigned Reg;
    unsigned Type;
    unsigned Space;
    unsigned Offset;
    unsigned Encoded;
  };

  typedef DenseMap<unsigned, RegisterInfo> RegisterInfoMap;

  RegisterInfoMap RegInfo;

  PTXParamManager ParamManager;

public:
  typedef DenseSet<unsigned>::const_iterator reg_iterator;

  PTXMachineFunctionInfo(MachineFunction &MF)
    : IsKernel(false) {
  }

  /// getParamManager - Returns the PTXParamManager instance for this function.
  PTXParamManager& getParamManager() { return ParamManager; }
  const PTXParamManager& getParamManager() const { return ParamManager; }

  /// setKernel/isKernel - Gets/sets a flag that indicates if this function is
  /// a PTX kernel function.
  void setKernel(bool _IsKernel=true) { IsKernel = _IsKernel; }
  bool isKernel() const { return IsKernel; }

  /// argreg_begin/argreg_end - Returns iterators to the set of registers
  /// containing function arguments.
  reg_iterator argreg_begin() const { return RegArgs.begin(); }
  reg_iterator argreg_end()   const { return RegArgs.end(); }

  /// retreg_begin/retreg_end - Returns iterators to the set of registers
  /// containing the function return values.
  reg_iterator retreg_begin() const { return RegRets.begin(); }
  reg_iterator retreg_end()   const { return RegRets.end(); }

  /// addRegister - Adds a virtual register to the set of all used registers
  void addRegister(unsigned Reg, unsigned RegType, unsigned RegSpace) {
    if (!RegInfo.count(Reg)) {
      RegisterInfo Info;
      Info.Reg = Reg;
      Info.Type = RegType;
      Info.Space = RegSpace;

      // Determine register offset
      Info.Offset = 0;
      for(RegisterInfoMap::const_iterator i = RegInfo.begin(),
          e = RegInfo.end(); i != e; ++i) {
        const RegisterInfo& RI = i->second;
        if (RI.Space == RegSpace)
          if (RI.Space != PTXRegisterSpace::Reg || RI.Type == Info.Type)
            Info.Offset++;
      }

      // Encode the register data into a single register number
      Info.Encoded = (Info.Offset << 6) | (Info.Type << 3) | Info.Space;

      RegInfo[Reg] = Info;

      if (RegSpace == PTXRegisterSpace::Argument)
        RegArgs.insert(Reg);
      else if (RegSpace == PTXRegisterSpace::Return)
        RegRets.insert(Reg);
    }
  }

  /// countRegisters - Returns the number of registers of the given type and
  /// space.
  unsigned countRegisters(unsigned RegType, unsigned RegSpace) const {
    unsigned Count = 0;
    for(RegisterInfoMap::const_iterator i = RegInfo.begin(), e = RegInfo.end();
        i != e; ++i) {
      const RegisterInfo& RI = i->second;
      if (RI.Type == RegType && RI.Space == RegSpace)
        Count++;
    }
    return Count;
  }

  /// getEncodedRegister - Returns the encoded value of the register.
  unsigned getEncodedRegister(unsigned Reg) const {
    return RegInfo.lookup(Reg).Encoded;
  }

  /// addRetReg - Adds a register to the set of return-value registers.
  void addRetReg(unsigned Reg) {
    if (!RegRets.count(Reg)) {
      RegRets.insert(Reg);
    }
  }

  /// addArgReg - Adds a register to the set of function argument registers.
  void addArgReg(unsigned Reg) {
    RegArgs.insert(Reg);
  }

  /// getRegisterName - Returns the name of the specified virtual register. This
  /// name is used during PTX emission.
  std::string getRegisterName(unsigned Reg) const {
    if (RegInfo.count(Reg)) {
      const RegisterInfo& RI = RegInfo.lookup(Reg);
      std::string Name;
      raw_string_ostream NameStr(Name);
      decodeRegisterName(NameStr, RI.Encoded);
      NameStr.flush();
      return Name;
    }
    else if (Reg == PTX::NoRegister)
      return "%noreg";
    else
      llvm_unreachable("Register not in register name map");
  }

  /// getEncodedRegisterName - Returns the name of the encoded register.
  std::string getEncodedRegisterName(unsigned EncodedReg) const {
    std::string Name;
    raw_string_ostream NameStr(Name);
    decodeRegisterName(NameStr, EncodedReg);
    NameStr.flush();
    return Name;
  }

  /// getRegisterType - Returns the type of the specified virtual register.
  unsigned getRegisterType(unsigned Reg) const {
    if (RegInfo.count(Reg))
      return RegInfo.lookup(Reg).Type;
    else
      llvm_unreachable("Unknown register");
  }

  /// getOffsetForRegister - Returns the offset of the virtual register
  unsigned getOffsetForRegister(unsigned Reg) const {
    if (RegInfo.count(Reg))
      return RegInfo.lookup(Reg).Offset;
    else
      return 0;
  }

  /// getFrameSymbol - Returns the symbol name for the given FrameIndex.
  const char* getFrameSymbol(int FrameIndex) {
    if (FrameSymbols.count(FrameIndex)) {
      return FrameSymbols.lookup(FrameIndex).c_str();
    } else {
      std::string Name          = "__local";
      Name                     += utostr(FrameIndex);
      // The whole point of caching this name is to ensure the pointer we pass
      // to any getExternalSymbol() calls will remain valid for the lifetime of
      // the back-end instance. This is to work around an issue in SelectionDAG
      // where symbol names are expected to be life-long strings.
      FrameSymbols[FrameIndex]  = Name;
      return FrameSymbols[FrameIndex].c_str();
    }
  }
}; // class PTXMachineFunctionInfo
} // namespace llvm

#endif // PTX_MACHINE_FUNCTION_INFO_H
