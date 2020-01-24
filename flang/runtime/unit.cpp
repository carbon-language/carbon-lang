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
#include <cerrno>
#include <type_traits>

namespace Fortran::runtime::io {

static Lock mapLock;
static Terminator mapTerminator;
static Map<int, ExternalFile> unitMap{MapAllocator<int, ExternalFile>{mapTerminator}};

ExternalFile *ExternalFile::LookUp(int unit) {
  CriticalSection criticalSection{mapLock};
  auto iter{unitMap.find(unit)};
  return iter == unitMap.end() ? nullptr : &iter->second;
}

ExternalFile &ExternalFile::LookUpOrCrash(int unit, Terminator &terminator) {
  CriticalSection criticalSection{mapLock};
  ExternalFile *file{LookUp(unit)};
  if (!file) {
    terminator.Crash("Not an open I/O unit number: %d", unit);
  }
  return *file;
}

ExternalFile &ExternalFile::Create(int unit, Terminator &terminator) {
  CriticalSection criticalSection{mapLock};
  auto pair{unitMap.emplace(unit, unit)};
  if (!pair.second) {
    terminator.Crash("Already opened I/O unit number: %d", unit);
  }
  return pair.first->second;
}

void ExternalFile::CloseUnit(IoErrorHandler &handler) {
  CriticalSection criticalSection{mapLock};
  Flush(handler);
  auto iter{unitMap.find(unitNumber_)};
  if (iter != unitMap.end()) {
    unitMap.erase(iter);
  }
}

void ExternalFile::InitializePredefinedUnits(Terminator &terminator) {
  ExternalFile &out{ExternalFile::Create(6, terminator)};
  out.Predefine(1);
  out.set_mayRead(false);
  out.set_mayWrite(true);
  out.set_mayPosition(false);
  ExternalFile &in{ExternalFile::Create(5, terminator)};
  in.Predefine(0);
  in.set_mayRead(true);
  in.set_mayWrite(false);
  in.set_mayPosition(false);
  // TODO: Set UTF-8 mode from the environment
}

void ExternalFile::CloseAll(IoErrorHandler &handler) {
  CriticalSection criticalSection{mapLock};
  while (!unitMap.empty()) {
    auto &pair{*unitMap.begin()};
    pair.second.CloseUnit(handler);
  }
}

bool ExternalFile::SetPositionInRecord(std::int64_t n, IoErrorHandler &handler) {
  n = std::max(std::int64_t{0}, n);
  bool ok{true};
  if (n > recordLength.value_or(n)) {
    handler.SignalEor();
    n = *recordLength;
    ok = false;
  }
  if (n > furthestPositionInRecord) {
    if (!isReading_ && ok) {
      WriteFrame(recordOffsetInFile, n, handler);
      std::fill_n(Frame() + furthestPositionInRecord, n - furthestPositionInRecord, ' ');
    }
    furthestPositionInRecord = n;
  }
  positionInRecord = n;
  return ok;
}

bool ExternalFile::Emit(const char *data, std::size_t bytes, IoErrorHandler &handler) {
  auto furthestAfter{std::max(furthestPositionInRecord, positionInRecord + static_cast<std::int64_t>(bytes))};
  WriteFrame(recordOffsetInFile, furthestAfter, handler);
  std::memcpy(Frame() + positionInRecord, data, bytes);
  positionInRecord += bytes;
  furthestPositionInRecord = furthestAfter;
  return true;
}

void ExternalFile::SetLeftTabLimit() {
  leftTabLimit = furthestPositionInRecord;
  positionInRecord = furthestPositionInRecord;
}

bool ExternalFile::NextOutputRecord(IoErrorHandler &handler) {
  bool ok{true};
  if (recordLength.has_value()) {  // fill fixed-size record
    ok &= SetPositionInRecord(*recordLength, handler);
  } else if (!unformatted && !isReading_) {
    ok &= SetPositionInRecord(furthestPositionInRecord, handler) &&
      Emit("\n", 1, handler);
  }
  recordOffsetInFile += furthestPositionInRecord;
  ++currentRecordNumber;
  positionInRecord = 0;
  positionInRecord = furthestPositionInRecord = 0;
  leftTabLimit.reset();
  return ok;
}

bool ExternalFile::HandleAbsolutePosition(std::int64_t n, IoErrorHandler &handler) {
  return SetPositionInRecord(std::max(n, std::int64_t{0}) + leftTabLimit.value_or(0), handler);
}

bool ExternalFile::HandleRelativePosition(std::int64_t n, IoErrorHandler &handler) {
  return HandleAbsolutePosition(positionInRecord + n, handler);
}

void ExternalFile::EndIoStatement() {
  u_.emplace<std::monostate>();
}
}
