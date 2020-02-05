//===-- runtime/io-stmt.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Representations of the state of an I/O statement in progress

#ifndef FORTRAN_RUNTIME_IO_STMT_H_
#define FORTRAN_RUNTIME_IO_STMT_H_

#include "descriptor.h"
#include "file.h"
#include "format.h"
#include "internal-unit.h"
#include "io-error.h"
#include <functional>
#include <type_traits>
#include <variant>

namespace Fortran::runtime::io {

struct ConnectionState;
class ExternalFileUnit;

class OpenStatementState;
class CloseStatementState;
class NoopCloseStatementState;
template<bool isInput, typename CHAR = char>
class InternalFormattedIoStatementState;
template<bool isInput, typename CHAR = char> class InternalListIoStatementState;
template<bool isInput, typename CHAR = char>
class ExternalFormattedIoStatementState;
template<bool isInput> class ExternalListIoStatementState;
template<bool isInput> class UnformattedIoStatementState;

// The Cookie type in the I/O API is a pointer (for C) to this class.
class IoStatementState {
public:
  template<typename A> explicit IoStatementState(A &x) : u_{x} {}

  // These member functions each project themselves into the active alternative.
  // They're used by per-data-item routines in the I/O API(e.g., OutputReal64)
  // to interact with the state of the I/O statement in progress.
  // This design avoids virtual member functions and function pointers,
  // which may not have good support in some use cases.
  DataEdit GetNextDataEdit(int = 1);
  bool Emit(const char *, std::size_t);
  bool AdvanceRecord(int = 1);
  int EndIoStatement();
  ConnectionState &GetConnectionState();
  MutableModes &mutableModes();

  // N.B.: this also works with base classes
  template<typename A> A *get_if() const {
    return std::visit(
        [](auto &x) -> A * {
          if constexpr (std::is_convertible_v<decltype(x.get()), A &>) {
            return &x.get();
          }
          return nullptr;
        },
        u_);
  }
  IoErrorHandler &GetIoErrorHandler() const;

  bool EmitRepeated(char, std::size_t);
  bool EmitField(const char *, std::size_t length, std::size_t width);

private:
  std::variant<std::reference_wrapper<OpenStatementState>,
      std::reference_wrapper<CloseStatementState>,
      std::reference_wrapper<NoopCloseStatementState>,
      std::reference_wrapper<InternalFormattedIoStatementState<false>>,
      std::reference_wrapper<InternalFormattedIoStatementState<true>>,
      std::reference_wrapper<InternalListIoStatementState<false>>,
      std::reference_wrapper<ExternalFormattedIoStatementState<false>>,
      std::reference_wrapper<ExternalListIoStatementState<false>>,
      std::reference_wrapper<UnformattedIoStatementState<false>>>
      u_;
};

// Base class for all per-I/O statement state classes.
// Inherits IoErrorHandler from its base.
struct IoStatementBase : public DefaultFormatControlCallbacks {
  using DefaultFormatControlCallbacks::DefaultFormatControlCallbacks;
  int EndIoStatement();
  DataEdit GetNextDataEdit(int = 1);  // crashing default
};

struct InputStatementState {};
struct OutputStatementState {};
template<bool isInput>
using IoDirectionState =
    std::conditional_t<isInput, InputStatementState, OutputStatementState>;

struct FormattedStatementState {};

template<bool isInput> struct ListDirectedStatementState {};
template<> struct ListDirectedStatementState<false /*output*/> {
  static std::size_t RemainingSpaceInRecord(const ConnectionState &);
  bool NeedAdvance(const ConnectionState &, std::size_t) const;
  bool EmitLeadingSpaceOrAdvance(
      IoStatementState &, std::size_t, bool isCharacter = false);
  bool lastWasUndelimitedCharacter{false};
};

template<bool isInput, typename CHAR = char>
class InternalIoStatementState : public IoStatementBase,
                                 public IoDirectionState<isInput> {
public:
  using CharType = CHAR;
  using Buffer = std::conditional_t<isInput, const CharType *, CharType *>;
  InternalIoStatementState(Buffer, std::size_t,
      const char *sourceFile = nullptr, int sourceLine = 0);
  InternalIoStatementState(
      const Descriptor &, const char *sourceFile = nullptr, int sourceLine = 0);
  int EndIoStatement();
  bool Emit(const CharType *, std::size_t chars /* not bytes */);
  bool AdvanceRecord(int = 1);
  ConnectionState &GetConnectionState() { return unit_; }
  MutableModes &mutableModes() { return unit_.modes; }

protected:
  bool free_{true};
  InternalDescriptorUnit<isInput> unit_;
};

template<bool isInput, typename CHAR>
class InternalFormattedIoStatementState
  : public InternalIoStatementState<isInput, CHAR>,
    public FormattedStatementState {
public:
  using CharType = CHAR;
  using typename InternalIoStatementState<isInput, CharType>::Buffer;
  InternalFormattedIoStatementState(Buffer internal, std::size_t internalLength,
      const CharType *format, std::size_t formatLength,
      const char *sourceFile = nullptr, int sourceLine = 0);
  InternalFormattedIoStatementState(const Descriptor &, const CharType *format,
      std::size_t formatLength, const char *sourceFile = nullptr,
      int sourceLine = 0);
  IoStatementState &ioStatementState() { return ioStatementState_; }
  int EndIoStatement();
  DataEdit GetNextDataEdit(int maxRepeat = 1) {
    return format_.GetNextDataEdit(*this, maxRepeat);
  }
  bool HandleRelativePosition(std::int64_t);
  bool HandleAbsolutePosition(std::int64_t);

private:
  IoStatementState ioStatementState_;  // points to *this
  using InternalIoStatementState<isInput, CharType>::unit_;
  // format_ *must* be last; it may be partial someday
  FormatControl<InternalFormattedIoStatementState> format_;
};

template<bool isInput, typename CHAR>
class InternalListIoStatementState
  : public InternalIoStatementState<isInput, CHAR>,
    public ListDirectedStatementState<isInput> {
public:
  using CharType = CHAR;
  using typename InternalIoStatementState<isInput, CharType>::Buffer;
  InternalListIoStatementState(Buffer internal, std::size_t internalLength,
      const char *sourceFile = nullptr, int sourceLine = 0);
  InternalListIoStatementState(
      const Descriptor &, const char *sourceFile = nullptr, int sourceLine = 0);
  IoStatementState &ioStatementState() { return ioStatementState_; }
  DataEdit GetNextDataEdit(int maxRepeat = 1) {
    DataEdit edit;
    edit.descriptor = DataEdit::ListDirected;
    edit.repeat = maxRepeat;
    edit.modes = InternalIoStatementState<isInput, CharType>::mutableModes();
    return edit;
  }

private:
  using InternalIoStatementState<isInput, CharType>::unit_;
  IoStatementState ioStatementState_;  // points to *this
};

class ExternalIoStatementBase : public IoStatementBase {
public:
  ExternalIoStatementBase(
      ExternalFileUnit &, const char *sourceFile = nullptr, int sourceLine = 0);
  ExternalFileUnit &unit() { return unit_; }
  MutableModes &mutableModes();
  ConnectionState &GetConnectionState();
  int EndIoStatement();

private:
  ExternalFileUnit &unit_;
};

template<bool isInput>
class ExternalIoStatementState : public ExternalIoStatementBase,
                                 public IoDirectionState<isInput> {
public:
  using ExternalIoStatementBase::ExternalIoStatementBase;
  int EndIoStatement();
  bool Emit(const char *, std::size_t chars /* not bytes */);
  bool Emit(const char16_t *, std::size_t chars /* not bytes */);
  bool Emit(const char32_t *, std::size_t chars /* not bytes */);
  bool AdvanceRecord(int = 1);
  bool HandleRelativePosition(std::int64_t);
  bool HandleAbsolutePosition(std::int64_t);
};

template<bool isInput, typename CHAR>
class ExternalFormattedIoStatementState
  : public ExternalIoStatementState<isInput>,
    public FormattedStatementState {
public:
  using CharType = CHAR;
  ExternalFormattedIoStatementState(ExternalFileUnit &, const CharType *format,
      std::size_t formatLength, const char *sourceFile = nullptr,
      int sourceLine = 0);
  MutableModes &mutableModes() { return mutableModes_; }
  int EndIoStatement();
  DataEdit GetNextDataEdit(int maxRepeat = 1) {
    return format_.GetNextDataEdit(*this, maxRepeat);
  }

private:
  // These are forked from ConnectionState's modes at the beginning
  // of each formatted I/O statement so they may be overridden by control
  // edit descriptors during the statement.
  MutableModes mutableModes_;
  FormatControl<ExternalFormattedIoStatementState> format_;
};

template<bool isInput>
class ExternalListIoStatementState
  : public ExternalIoStatementState<isInput>,
    public ListDirectedStatementState<isInput> {
public:
  using ExternalIoStatementState<isInput>::ExternalIoStatementState;
  DataEdit GetNextDataEdit(int maxRepeat = 1) {
    DataEdit edit;
    edit.descriptor = DataEdit::ListDirected;
    edit.repeat = maxRepeat;
    edit.modes = ExternalIoStatementState<isInput>::mutableModes();
    return edit;
  }
};

template<bool isInput>
class UnformattedIoStatementState : public ExternalIoStatementState<isInput> {
public:
  using ExternalIoStatementState<isInput>::ExternalIoStatementState;
  int EndIoStatement();
};

class OpenStatementState : public ExternalIoStatementBase {
public:
  OpenStatementState(ExternalFileUnit &unit, bool wasExtant,
      const char *sourceFile = nullptr, int sourceLine = 0)
    : ExternalIoStatementBase{unit, sourceFile, sourceLine}, wasExtant_{
                                                                 wasExtant} {}
  bool wasExtant() const { return wasExtant_; }
  void set_status(OpenStatus status) { status_ = status; }
  void set_path(const char *, std::size_t, int kind);  // FILE=
  void set_position(Position position) { position_ = position; }  // POSITION=
  int EndIoStatement();

private:
  bool wasExtant_;
  OpenStatus status_{OpenStatus::Unknown};
  Position position_{Position::AsIs};
  OwningPtr<char> path_;
  std::size_t pathLength_;
};

class CloseStatementState : public ExternalIoStatementBase {
public:
  CloseStatementState(ExternalFileUnit &unit, const char *sourceFile = nullptr,
      int sourceLine = 0)
    : ExternalIoStatementBase{unit, sourceFile, sourceLine} {}
  void set_status(CloseStatus status) { status_ = status; }
  int EndIoStatement();

private:
  CloseStatus status_{CloseStatus::Keep};
};

class NoopCloseStatementState : public IoStatementBase {
public:
  NoopCloseStatementState(const char *sourceFile, int sourceLine)
    : IoStatementBase{sourceFile, sourceLine}, ioStatementState_{*this} {}
  IoStatementState &ioStatementState() { return ioStatementState_; }
  void set_status(CloseStatus) {}  // discards
  MutableModes &mutableModes() { return connection_.modes; }
  ConnectionState &GetConnectionState() { return connection_; }
  int EndIoStatement();

private:
  IoStatementState ioStatementState_;  // points to *this
  ConnectionState connection_;
};

extern template class InternalIoStatementState<false>;
extern template class InternalIoStatementState<true>;
extern template class InternalFormattedIoStatementState<false>;
extern template class InternalFormattedIoStatementState<true>;
extern template class InternalListIoStatementState<false>;
extern template class ExternalIoStatementState<false>;
extern template class ExternalFormattedIoStatementState<false>;
extern template class ExternalListIoStatementState<false>;
extern template class UnformattedIoStatementState<false>;
extern template class FormatControl<InternalFormattedIoStatementState<false>>;
extern template class FormatControl<InternalFormattedIoStatementState<true>>;
extern template class FormatControl<ExternalFormattedIoStatementState<false>>;

}
#endif  // FORTRAN_RUNTIME_IO_STMT_H_
