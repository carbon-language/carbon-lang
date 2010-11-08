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
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {
/// PTXMachineFunctionInfo - This class is derived from MachineFunction and
/// contains private PTX target-specific information for each MachineFunction.
///
class PTXMachineFunctionInfo : public MachineFunctionInfo {
private:
  bool is_kernel;
  std::vector<unsigned> reg_arg, reg_local_var;
  unsigned reg_ret;
  bool _isDoneAddArg;

public:
  PTXMachineFunctionInfo(MachineFunction &MF)
    : is_kernel(false), reg_ret(PTX::NoRegister), _isDoneAddArg(false) {
      reg_arg.reserve(32);
      reg_local_var.reserve(64);
    }

  void setKernel(bool _is_kernel=true) { is_kernel = _is_kernel; }

  void addArgReg(unsigned reg) { reg_arg.push_back(reg); }
  void addLocalVarReg(unsigned reg) { reg_local_var.push_back(reg); }
  void setRetReg(unsigned reg) { reg_ret = reg; }

  void doneAddArg(void) {
    std::sort(reg_arg.begin(), reg_arg.end());
    _isDoneAddArg = true;
  }
  void doneAddLocalVar(void) {
    std::sort(reg_local_var.begin(), reg_local_var.end());
  }

  bool isDoneAddArg(void) { return _isDoneAddArg; }

  bool isKernel() const { return is_kernel; }

  typedef std::vector<unsigned>::const_iterator reg_iterator;

  bool argRegEmpty() const { return reg_arg.empty(); }
  reg_iterator argRegBegin() const { return reg_arg.begin(); }
  reg_iterator argRegEnd()   const { return reg_arg.end(); }

  bool localVarRegEmpty() const { return reg_local_var.empty(); }
  reg_iterator localVarRegBegin() const { return reg_local_var.begin(); }
  reg_iterator localVarRegEnd()   const { return reg_local_var.end(); }

  unsigned retReg() const { return reg_ret; }

  bool isArgReg(unsigned reg) const {
    return std::binary_search(reg_arg.begin(), reg_arg.end(), reg);
  }

  bool isLocalVarReg(unsigned reg) const {
    return std::binary_search(reg_local_var.begin(), reg_local_var.end(), reg);
  }
}; // class PTXMachineFunctionInfo
} // namespace llvm

#endif // PTX_MACHINE_FUNCTION_INFO_H
