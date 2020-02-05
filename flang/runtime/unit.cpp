//===-- runtime/unit.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "unit.h"
#include "lock.h"
#include "memory.h"
#include "tools.h"
#include <algorithm>
#include <type_traits>

namespace Fortran::runtime::io {

static Lock mapLock;
static Terminator mapTerminator;
static Map<int, ExternalFileUnit> unitMap{
    MapAllocator<int, ExternalFileUnit>{mapTerminator}};
static ExternalFileUnit *defaultOutput{nullptr};

void FlushOutputOnCrash(const Terminator &terminator) {
  if (defaultOutput) {
    IoErrorHandler handler{terminator};
    handler.HasIoStat();  // prevent nested crash if flush has error
    defaultOutput->Flush(handler);
  }
}

ExternalFileUnit *ExternalFileUnit::LookUp(int unit) {
  CriticalSection criticalSection{mapLock};
  auto iter{unitMap.find(unit)};
  return iter == unitMap.end() ? nullptr : &iter->second;
}

ExternalFileUnit &ExternalFileUnit::LookUpOrCrash(
    int unit, const Terminator &terminator) {
  CriticalSection criticalSection{mapLock};
  ExternalFileUnit *file{LookUp(unit)};
  if (!file) {
    terminator.Crash("Not an open I/O unit number: %d", unit);
  }
  return *file;
}

ExternalFileUnit &ExternalFileUnit::LookUpOrCreate(int unit, bool *wasExtant) {
  CriticalSection criticalSection{mapLock};
  auto pair{unitMap.emplace(unit, unit)};
  if (wasExtant) {
    *wasExtant = !pair.second;
  }
  return pair.first->second;
}

int ExternalFileUnit::NewUnit() {
  CriticalSection criticalSection{mapLock};
  static int nextNewUnit{-1000};  // see 12.5.6.12 in Fortran 2018
  return --nextNewUnit;
}

void ExternalFileUnit::OpenUnit(OpenStatus status, Position position,
    OwningPtr<char> &&newPath, std::size_t newPathLength,
    IoErrorHandler &handler) {
  CriticalSection criticalSection{lock()};
  if (IsOpen()) {
    if (status == OpenStatus::Old &&
        (!newPath.get() ||
            (path() && pathLength() == newPathLength &&
                std::memcmp(path(), newPath.get(), newPathLength) == 0))) {
      // OPEN of existing unit, STATUS='OLD', not new FILE=
      newPath.reset();
      return;
    }
    // Otherwise, OPEN on open unit with new FILE= implies CLOSE
    Flush(handler);
    Close(CloseStatus::Keep, handler);
  }
  set_path(std::move(newPath), newPathLength);
  Open(status, position, handler);
}

void ExternalFileUnit::CloseUnit(CloseStatus status, IoErrorHandler &handler) {
  {
    CriticalSection criticalSection{lock()};
    Flush(handler);
    Close(status, handler);
  }
  CriticalSection criticalSection{mapLock};
  auto iter{unitMap.find(unitNumber_)};
  if (iter != unitMap.end()) {
    unitMap.erase(iter);
  }
}

void ExternalFileUnit::InitializePredefinedUnits() {
  ExternalFileUnit &out{ExternalFileUnit::LookUpOrCreate(6)};
  out.Predefine(1);
  out.set_mayRead(false);
  out.set_mayWrite(true);
  out.set_mayPosition(false);
  defaultOutput = &out;
  ExternalFileUnit &in{ExternalFileUnit::LookUpOrCreate(5)};
  in.Predefine(0);
  in.set_mayRead(true);
  in.set_mayWrite(false);
  in.set_mayPosition(false);
  // TODO: Set UTF-8 mode from the environment
}

void ExternalFileUnit::CloseAll(IoErrorHandler &handler) {
  CriticalSection criticalSection{mapLock};
  defaultOutput = nullptr;
  while (!unitMap.empty()) {
    auto &pair{*unitMap.begin()};
    pair.second.CloseUnit(CloseStatus::Keep, handler);
  }
}

bool ExternalFileUnit::SetPositionInRecord(
    std::int64_t n, IoErrorHandler &handler) {
  n = std::max<std::int64_t>(0, n);
  bool ok{true};
  if (n > static_cast<std::int64_t>(recordLength.value_or(n))) {
    handler.SignalEor();
    n = *recordLength;
    ok = false;
  }
  if (n > furthestPositionInRecord) {
    if (!isReading_ && ok) {
      WriteFrame(recordOffsetInFile, n, handler);
      std::fill_n(Frame() + furthestPositionInRecord,
          n - furthestPositionInRecord, ' ');
    }
    furthestPositionInRecord = n;
  }
  positionInRecord = n;
  return ok;
}

bool ExternalFileUnit::Emit(
    const char *data, std::size_t bytes, IoErrorHandler &handler) {
  auto furthestAfter{std::max(furthestPositionInRecord,
      positionInRecord + static_cast<std::int64_t>(bytes))};
  WriteFrame(recordOffsetInFile, furthestAfter, handler);
  std::memcpy(Frame() + positionInRecord, data, bytes);
  positionInRecord += bytes;
  furthestPositionInRecord = furthestAfter;
  return true;
}

void ExternalFileUnit::SetLeftTabLimit() {
  leftTabLimit = furthestPositionInRecord;
  positionInRecord = furthestPositionInRecord;
}

bool ExternalFileUnit::AdvanceRecord(IoErrorHandler &handler) {
  bool ok{true};
  if (recordLength.has_value()) {  // fill fixed-size record
    ok &= SetPositionInRecord(*recordLength, handler);
  } else if (!isUnformatted && !isReading_) {
    ok &= SetPositionInRecord(furthestPositionInRecord, handler) &&
        Emit("\n", 1, handler);
  }
  recordOffsetInFile += furthestPositionInRecord;
  ++currentRecordNumber;
  positionInRecord = 0;
  furthestPositionInRecord = 0;
  leftTabLimit.reset();
  return ok;
}

bool ExternalFileUnit::HandleAbsolutePosition(
    std::int64_t n, IoErrorHandler &handler) {
  return SetPositionInRecord(
      std::max(n, std::int64_t{0}) + leftTabLimit.value_or(0), handler);
}

bool ExternalFileUnit::HandleRelativePosition(
    std::int64_t n, IoErrorHandler &handler) {
  return HandleAbsolutePosition(positionInRecord + n, handler);
}

void ExternalFileUnit::FlushIfTerminal(IoErrorHandler &handler) {
  if (isTerminal()) {
    Flush(handler);
  }
}

void ExternalFileUnit::EndIoStatement() {
  io_.reset();
  u_.emplace<std::monostate>();
}
}
