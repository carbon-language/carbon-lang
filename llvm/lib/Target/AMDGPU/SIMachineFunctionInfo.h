//===- SIMachineFunctionInfo.h - SIMachineFunctionInfo interface -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SIMachineFunctionInfo is used to keep track of the spi_sp_input_addr config
// register, which is to tell the hardware which interpolation parameters to
// load.
//
//===----------------------------------------------------------------------===//


#ifndef _SIMACHINEFUNCTIONINFO_H_
#define _SIMACHINEFUNCTIONINFO_H_

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class SIMachineFunctionInfo : public MachineFunctionInfo {

  private:

  public:
    SIMachineFunctionInfo(const MachineFunction &MF);
    unsigned spi_ps_input_addr;

};

} // End namespace llvm


#endif //_SIMACHINEFUNCTIONINFO_H_
