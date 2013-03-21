//===-- Timer.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Timer.h"
#include <assert.h>

using namespace lldb_perf;

TimeGauge::TimeType
TimeGauge::Now ()
{
	return high_resolution_clock::now();
}

TimeGauge::TimeGauge () :
m_start(),
    m_state(TimeGauge::State::eNeverUsed)
{
}

void
TimeGauge::Start ()
{
	m_state = TimeGauge::State::eCounting;
	m_start = Now();
}

double
TimeGauge::Stop ()
{
	auto stop = Now();
	assert(m_state == TimeGauge::State::eCounting && "cannot stop a non-started clock");
	m_state = TimeGauge::State::eStopped;
	return (m_value = duration_cast<duration<double>>(stop-m_start).count());
}

double
TimeGauge::GetValue ()
{
	assert(m_state == TimeGauge::State::eStopped && "clock must be used before you can evaluate it");
	return m_value;
}
