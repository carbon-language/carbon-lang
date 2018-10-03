// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "check-do-concurrent.h"
#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"

namespace Fortran::semantics {

using namespace parser::literals;

// 11.1.7.5 - enforce semantics constraints on a DO CONCURRENT loop body
class DoConcurrentEnforcement {
public:
  DoConcurrentEnforcement(parser::Messages &messages) : messages_{messages} {}
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}
  template<typename T> bool Pre(const parser::Statement<T> &statement) {
    charBlock_ = statement.source;
    return true;
  }
  // C1136
  void Post(const parser::ReturnStmt &) {
    messages_.Say(charBlock_,
        parser::MessageFormattedText{
            "RETURN not allowed in DO CONCURRENT"_err_en_US});
  }
  // C1137
  void NoImageControl() {
    messages_.Say(charBlock_,
        parser::MessageFormattedText{
            "image control statement not allowed in DO CONCURRENT"_err_en_US});
  }
  void Post(const parser::SyncAllStmt &) { NoImageControl(); }
  void Post(const parser::SyncImagesStmt &) { NoImageControl(); }
  void Post(const parser::SyncMemoryStmt &) { NoImageControl(); }
  void Post(const parser::SyncTeamStmt &) { NoImageControl(); }
  void Post(const parser::ChangeTeamConstruct &) { NoImageControl(); }
  void Post(const parser::CriticalConstruct &) { NoImageControl(); }
  void Post(const parser::EventPostStmt &) { NoImageControl(); }
  void Post(const parser::EventWaitStmt &) { NoImageControl(); }
  void Post(const parser::FormTeamStmt &) { NoImageControl(); }
  void Post(const parser::LockStmt &) { NoImageControl(); }
  void Post(const parser::UnlockStmt &) { NoImageControl(); }
  void Post(const parser::StopStmt &) { NoImageControl(); }
  void Post(const parser::EndProgramStmt &) { NoImageControl(); }

  void Post(const parser::AllocateStmt &) {
    if (ObjectIsCoarray()) {
      messages_.Say(charBlock_,
          parser::MessageFormattedText{
              "ALLOCATE coarray not allowed in DO CONCURRENT"_err_en_US});
    }
  }
  void Post(const parser::DeallocateStmt &) {
    if (ObjectIsCoarray()) {
      messages_.Say(charBlock_,
          parser::MessageFormattedText{
              "DEALLOCATE coarray not allowed in DO CONCURRENT"_err_en_US});
    }
    // C1140: deallocation of polymorphic objects
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    if (EndTDeallocatesCoarray()) {
      messages_.Say(charBlock_,
          parser::MessageFormattedText{
              "implicit deallocation of coarray not allowed"
              " in DO CONCURRENT"_err_en_US});
    }
  }
  void Post(const parser::CallStmt &) {
    // C1137: call move_alloc with coarray arguments
    // C1139: call to impure procedure
  }
  // C1141
  void Post(const parser::ProcedureDesignator &procedureDesignator) {
#if 0
    if (auto *name{std::get_if<parser::Name>(&procedureDesignator.u)}) {
      auto upperName{parser::ToUpperCaseLetters(name->ToString())};
      // FIXME: These names need to resolve to symbols imported from the
      // intrinsic module IEEE_EXCEPTIONS. Do some symbol lookup here?
      if (upperName == "IEEE_GET_FLAG"s) {
        messages_.Say(charBlock_,
            parser::MessageFormattedText{
                "IEEE_GET_FLAG not allowed in DO CONCURRENT"_err_en_US});
      } else if (upperName == "IEEE_SET_HALTING_MODE"s) {
        messages_.Say(charBlock_,
            parser::MessageFormattedText{"IEEE_SET_HALTING_MODE not allowed"
                                         " in DO CONCURRENT"_err_en_US});
      } else if (upperName == "IEEE_GET_HALTING_MODE"s) {
        messages_.Say(charBlock_,
            parser::MessageFormattedText{"IEEE_GET_HALTING_MODE not allowed"
                                         " in DO CONCURRENT"_err_en_US});
      }
    }
#endif
  }

  // C1138: extended ranges in DOs should be errors, not warnings

  // 11.1.7.5
  void Post(const parser::IoControlSpec &ioControlSpec) {
    if (auto *charExpr{
            std::get_if<parser::IoControlSpec::CharExpr>(&ioControlSpec.u)}) {
      if (std::get<parser::IoControlSpec::CharExpr::Kind>(charExpr->t) ==
          parser::IoControlSpec::CharExpr::Kind::Advance) {
        messages_.Say(charBlock_,
            parser::MessageFormattedText{
                "ADVANCE specifier not allowed in DO CONCURRENT"_err_en_US});
      }
    }
  }

private:
  bool ObjectIsCoarray() { return false; }  // placeholder
  bool EndTDeallocatesCoarray() { return false; }  // placeholder

  parser::CharBlock charBlock_;
  parser::Messages &messages_;
};

// Find a canonical DO CONCURRENT and enforce semantics checks on its body
class FindDoConcurrentLoops {
public:
  FindDoConcurrentLoops(parser::Messages &messages) : messages_{messages} {}
  template<typename T> constexpr bool Pre(const T &) { return true; }
  template<typename T> constexpr void Post(const T &) {}
  template<typename T> constexpr bool Pre(const parser::Statement<T> &) {
    return false;
  }
  bool Pre(const parser::DoConstruct &doConstruct) {
    if (std::get<std::optional<parser::LoopControl>>(
            std::get<parser::Statement<parser::NonLabelDoStmt>>(doConstruct.t)
                .statement.t)
            .has_value() &&
        std::holds_alternative<parser::LoopControl::Concurrent>(
            std::get<std::optional<parser::LoopControl>>(
                std::get<parser::Statement<parser::NonLabelDoStmt>>(
                    doConstruct.t)
                    .statement.t)
                ->u)) {
      DoConcurrentEnforcement doConcurrentEnforcement{messages_};
      parser::Walk(
          std::get<parser::Block>(doConstruct.t), doConcurrentEnforcement);
    }
    return true;
  }

private:
  parser::Messages &messages_;
};

// DO loops must be canonicalized prior to calling
void CheckDoConcurrentConstraints(
    parser::Messages &messages, const parser::Program &program) {
  FindDoConcurrentLoops findDoConcurrentLoops{messages};
  Walk(program, findDoConcurrentLoops);
}

}  // namespace Fortran::semantics
