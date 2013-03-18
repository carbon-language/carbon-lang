//
//  MemoryGauge.cpp
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/6/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#include "MemoryGauge.h"
#include <assert.h>
#include <mach/task.h>

using namespace lldb_perf;

MemoryGauge::SizeType
MemoryGauge::now ()
{
    task_t task = MACH_PORT_NULL;
    mach_task_basic_info_data_t taskBasicInfo;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(task, MACH_TASK_BASIC_INFO, (task_info_t) & taskBasicInfo, &count) == KERN_SUCCESS) {
        return taskBasicInfo.virtual_size;
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
