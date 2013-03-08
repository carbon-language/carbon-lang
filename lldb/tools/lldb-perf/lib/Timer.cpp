//
//  Timer.cpp
//  PerfTestDriver
//
//  Created by Enrico Granata on 3/6/13.
//  Copyright (c) 2013 Apple Inc. All rights reserved.
//

#include "Timer.h"
#include <assert.h>

using namespace lldb::perf;

TimeGauge::HPTime
TimeGauge::now ()
{
	return high_resolution_clock::now();
}

TimeGauge::TimeGauge () :
m_start(),
m_state(TimeGauge::State::eTSNeverUsed)
{
}

void
TimeGauge::start ()
{
	m_state = TimeGauge::State::eTSCounting;
	m_start = now();
}

double
TimeGauge::stop ()
{
	auto stop = now();
	assert(m_state == TimeGauge::State::eTSCounting && "cannot stop a non-started clock");
	m_state = TimeGauge::State::eTSStopped;
	return (m_value = duration_cast<duration<double>>(stop-m_start).count());
}

double
TimeGauge::value ()
{
	assert(m_state == TimeGauge::State::eTSStopped && "clock must be used before you can evaluate it");
	return m_value;
}
