//
//  MemoryGauge.cpp
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/6/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#include "MemoryGauge.h"
#include <assert.h>
#include <mach/mach.h>
#include <mach/task.h>
#include <mach/mach_traps.h>

using namespace lldb_perf;

MemoryStats::MemoryStats () : MemoryStats(0,0,0) {}
MemoryStats::MemoryStats (mach_vm_size_t vs,mach_vm_size_t rs, mach_vm_size_t mrs) :
m_virtual_size(vs),
m_resident_size(rs),
m_max_resident_size(mrs)
{}

MemoryStats::MemoryStats (const MemoryStats& rhs) : MemoryStats(rhs.m_virtual_size,rhs.m_resident_size,rhs.m_max_resident_size)
{}

MemoryStats&
MemoryStats::operator = (const MemoryStats& rhs)
{
    if (&rhs != this)
    {
        m_virtual_size = rhs.m_virtual_size;
        m_resident_size = rhs.m_resident_size;
        m_max_resident_size = rhs.m_max_resident_size;
    }
    return *this;
}

MemoryStats&
MemoryStats::operator += (const MemoryStats& rhs)
{
    m_virtual_size += rhs.m_virtual_size;
    m_resident_size += rhs.m_resident_size;
    m_max_resident_size += rhs.m_max_resident_size;
    return *this;
}

MemoryStats
MemoryStats::operator - (const MemoryStats& rhs)
{
    return MemoryStats(m_virtual_size - rhs.m_virtual_size,
                       m_resident_size - rhs.m_resident_size,
                       m_max_resident_size - rhs.m_max_resident_size);
}

MemoryStats&
MemoryStats::operator / (size_t rhs)
{
    m_virtual_size /= rhs;
    m_resident_size /= rhs;
    m_max_resident_size /= rhs;
    return *this;
}

MemoryGauge::SizeType
MemoryGauge::now ()
{
    task_t task = mach_task_self();
    mach_task_basic_info_data_t taskBasicInfo;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    auto task_info_ret = task_info(task, MACH_TASK_BASIC_INFO, (task_info_t) & taskBasicInfo, &count);
    if (task_info_ret == KERN_SUCCESS) {
        return MemoryStats(taskBasicInfo.virtual_size, taskBasicInfo.resident_size, taskBasicInfo.resident_size_max);
    }
    return 0;
}

MemoryGauge::MemoryGauge () :
m_start(),
m_state(MemoryGauge::State::eMSNeverUsed)
{
}

void
MemoryGauge::start ()
{
	m_state = MemoryGauge::State::eMSCounting;
	m_start = now();
}

MemoryGauge::SizeType
MemoryGauge::stop ()
{
	auto stop = now();
	assert(m_state == MemoryGauge::State::eMSCounting && "cannot stop a non-started gauge");
	m_state = MemoryGauge::State::eMSStopped;
	return (m_value = stop-m_start);
}

MemoryGauge::SizeType
MemoryGauge::value ()
{
	assert(m_state == MemoryGauge::State::eMSStopped && "gauge must be used before you can evaluate it");
	return m_value;
}
