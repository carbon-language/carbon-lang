//===-- ArmUnwindInfo.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ArmUnwindInfo_h_
#define liblldb_ArmUnwindInfo_h_

#include <vector>

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/lldb-private.h"

/*
 * Unwind information reader and parser for the ARM exception handling ABI
 *
 * Implemented based on:
 *     Exception Handling ABI for the ARM Architecture
 *     Document number: ARM IHI 0038A (current through ABI r2.09)
 *     Date of Issue: 25th January 2007, reissued 30th November 2012
 *     http://infocenter.arm.com/help/topic/com.arm.doc.ihi0038a/IHI0038A_ehabi.pdf
 */

namespace lldb_private {

class ArmUnwindInfo
{
public:
    ArmUnwindInfo (ObjectFile& objfile, lldb::SectionSP& arm_exidx, lldb::SectionSP& arm_extab);
    ~ArmUnwindInfo();

    bool
    GetUnwindPlan (Target &target, const Address& addr, UnwindPlan& unwind_plan);

private:
    const uint8_t*
    GetExceptionHandlingTableEntry(const Address& addr);
    
    lldb::SectionSP m_arm_exidx_sp; // .ARM.exidx section
    lldb::SectionSP m_arm_extab_sp; // .ARM.extab section

    DataExtractor m_arm_exidx_data; // .ARM.exidx section data
    DataExtractor m_arm_extab_data; // .ARM.extab section data
};

} // namespace lldb_private

#endif  // liblldb_ArmUnwindInfo_h_
