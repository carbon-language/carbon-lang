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

#ifndef FORTRAN_PARSER_BASIC_PARSERS_H_
#define FORTRAN_PARSER_BASIC_PARSERS_H_

// Let a "parser" be an instance of any class that supports this
// type definition and member (or static) function:
//
//   using resultType = ...;
//   std::optional<resultType> Parse(ParseState &) const;
//
// which either returns a value to signify a successful recognition or else
// returns {} to signify failure.  On failure, the state cannot be assumed
// to still be valid, in general -- see below for exceptions.
//
// This header defines the fundamental parser class templates and helper
// template functions.  See parser-combinators.txt for documentation.

#include "char-block.h"
#include "features.h"
#include "message.h"
#include "parse-state.h"
#include "provenance.h"
#include "user-state.h"
#include "../common/idioms.h"
#include "../common/indirection.h"
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>

namespace Fortran::parser {

// fail<A>("..."_err_en_US) returns a parser that never succeeds.  It reports an
// error message at the current position.  The result type is unused,
// but might have to be specified at the point of call for satisfy
// the type checker.  The state remains valid.
template<typename A> class FailParser {
public:
  using resultType = A;
  constexpr FailParser(const FailParser &) = default;
  constexpr explicit FailParser(MessageFixedText t) : text_{t} {}
  std::optional<A> Parse(ParseState &state) const {
    state.Say(text_);
    return std::nullopt;
  }

private:
  const MessageFixedText text_;
};

template<typename A = Success> inline constexpr auto fail(MessageFixedText t) {
  return FailParser<A>{t};
}

// pure(x) returns a parsers that always succeeds, does not advance the
// parse, and returns a captured value whose type must be copy-constructible.
template<typename A> class PureParser {
public:
  using resultType = A;
  constexpr PureParser(const PureParser &) = default;
  constexpr explicit PureParser(A &&x) : value_(std::move(x)) {}
  std::optional<A> Parse(ParseState &) const { return {value_}; }

private:
  const A value_;
};

template<typename A> inline constexpr auto pure(A x) {
  return PureParser<A>(std::move(x));
}

// If a is a parser, attempt(a) is the same parser, but on failure
// the ParseState is guaranteed to have been restored to its initial value.
template<typename A> class BacktrackingParser {
public:
  using resultType = typename A::resultType;
  constexpr BacktrackingParser(const BacktrackingParser &) = default;
  constexpr BacktrackingParser(const A &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    Messages messages{std::move(state.messages())};
    ParseState backtrack{state};
    std::optional<resultType> result{parser_.Parse(state)};
    if (result.has_value()) {
      state.messages().Restore(std::move(messages));
    } else {
      state = std::move(backtrack);
      state.messages() = std::move(messages);
    }
    return result;
  }

private:
  const A parser_;
};

template<typename A> inline constexpr auto attempt(const A &parser) {
  return BacktrackingParser<A>{parser};
}

// For any parser x, the parser returned by !x is one that succeeds when
// x fails, returning a useless (but present) result.  !x fails when x succeeds.
template<typename PA> class NegatedParser {
public:
  using resultType = Success;
  constexpr NegatedParser(const NegatedParser &) = default;
  constexpr NegatedParser(const PA &p) : parser_{p} {}
  std::optional<Success> Parse(ParseState &state) const {
    ParseState forked{state};
    forked.set_deferMessages(true);
    if (parser_.Parse(forked)) {
      return std::nullopt;
    }
    return {Success{}};
  }

private:
  const PA parser_;
};

template<typename PA> inline constexpr auto operator!(const PA &p) {
  return NegatedParser<PA>(p);
}

// For any parser x, the parser returned by lookAhead(x) is one that succeeds
// or fails if x does, but the state is not modified.
template<typename PA> class LookAheadParser {
public:
  using resultType = Success;
  constexpr LookAheadParser(const LookAheadParser &) = default;
  constexpr LookAheadParser(const PA &p) : parser_{p} {}
  std::optional<Success> Parse(ParseState &state) const {
    ParseState forked{state};
    forked.set_deferMessages(true);
    if (parser_.Parse(forked).has_value()) {
      return {Success{}};
    }
    return std::nullopt;
  }

private:
  const PA parser_;
};

template<typename PA> inline constexpr auto lookAhead(const PA &p) {
  return LookAheadParser<PA>{p};
}

// If a is a parser, inContext("..."_en_US, a) runs it in a nested message
// context.
template<typename PA> class MessageContextParser {
public:
  using resultType = typename PA::resultType;
  constexpr MessageContextParser(const MessageContextParser &) = default;
  constexpr MessageContextParser(MessageFixedText t, const PA &p)
    : text_{t}, parser_{p} {}
  std::optional<resultType> Parse(ParseState &state) const {
    state.PushContext(text_);
    std::optional<resultType> result{parser_.Parse(state)};
    state.PopContext();
    return result;
  }

private:
  const MessageFixedText text_;
  const PA parser_;
};

template<typename PA>
inline constexpr auto inContext(MessageFixedText context, const PA &parser) {
  return MessageContextParser{context, parser};
}

// If a is a parser, withMessage("..."_en_US, a) runs it unchanged if it
// succeeds, and overrides its messages with a specific one if it fails and
// has matched no tokens.
template<typename PA> class WithMessageParser {
public:
  using resultType = typename PA::resultType;
  constexpr WithMessageParser(const WithMessageParser &) = default;
  constexpr WithMessageParser(MessageFixedText t, const PA &p)
    : text_{t}, parser_{p} {}
  std::optional<resultType> Parse(ParseState &state) const {
    Messages messages{std::move(state.messages())};
    ParseState backtrack{state};
    state.set_anyTokenMatched(false);
    std::optional<resultType> result{parser_.Parse(state)};
    bool emitMessage{false};
    if (result.has_value()) {
      messages.Annex(std::move(state.messages()));
      if (backtrack.anyTokenMatched()) {
        state.set_anyTokenMatched();
      }
    } else if (state.anyTokenMatched()) {
      messages.Annex(std::move(state.messages()));
      backtrack.set_anyTokenMatched();
      if (state.anyDeferredMessages()) {
        backtrack.set_anyDeferredMessages(true);
      }
      state = std::move(backtrack);
    } else {
      emitMessage = true;
    }
    state.messages() = std::move(messages);
    if (emitMessage) {
      state.Say(text_);
    }
    return result;
  }

private:
  const MessageFixedText text_;
  const PA parser_;
};

template<typename PA>
inline constexpr auto withMessage(MessageFixedText msg, const PA &parser) {
  return WithMessageParser{msg, parser};
}

// If a and b are parsers, then a >> b returns a parser that succeeds when
// b succeeds after a does so, but fails when either a or b does.  The
// result is taken from b.  Similarly, a / b also succeeds if both a and b
// do so, but the result is that returned by a.
template<typename PA, typename PB> class SequenceParser {
public:
  using resultType = typename PB::resultType;
  constexpr SequenceParser(const SequenceParser &) = default;
  constexpr SequenceParser(const PA &pa, const PB &pb) : pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    std::optional<resultType> result;
    if (pa_.Parse(state)) {
      result = pb_.Parse(state);
    }
    return result;
  }

private:
  const PA pa_;
  const PB pb_;
};

template<typename PA, typename PB>
inline constexpr auto operator>>(const PA &pa, const PB &pb) {
  return SequenceParser<PA, PB>{pa, pb};
}

template<typename PA, typename PB> class InvertedSequenceParser {
public:
  using resultType = typename PA::resultType;
  constexpr InvertedSequenceParser(const InvertedSequenceParser &) = default;
  constexpr InvertedSequenceParser(const PA &pa, const PB &pb)
    : pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    std::optional<resultType> result;
    if (std::optional<resultType> ax{pa_.Parse(state)}) {
      if (pb_.Parse(state)) {
        result = std::move(ax);
      }
    }
    return result;
  }

private:
  const PA pa_;
  const PB pb_;
};

template<typename PA, typename PB>
inline constexpr auto operator/(const PA &pa, const PB &pb) {
  return InvertedSequenceParser<PA, PB>{pa, pb};
}

template<typename PA, typename... Ps> class AlternativesParser {
public:
  using resultType = typename PA::resultType;
  constexpr AlternativesParser(const PA &pa, const Ps &... ps)
    : ps_{pa, ps...} {}
  constexpr AlternativesParser(const AlternativesParser &) = default;
  std::optional<resultType> Parse(ParseState &state) const {
    Messages messages{std::move(state.messages())};
    ParseState backtrack{state};
    std::optional<resultType> result{std::get<0>(ps_).Parse(state)};
    if (!result.has_value()) {
      ParseRest<1>(result, state, backtrack);
    }
    state.messages().Restore(std::move(messages));
    return result;
  }

private:
  template<int J>
  void ParseRest(std::optional<resultType> &result, ParseState &state,
      ParseState &backtrack) const {
    if constexpr (J <= sizeof...(Ps)) {
      ParseState prevState{std::move(state)};
      state = backtrack;
      const auto &parser{std::get<J>(ps_)};
      static_assert(std::is_same_v<resultType,
          typename std::decay<decltype(parser)>::type::resultType>);
      result = parser.Parse(state);
      if (!result.has_value()) {
        state.CombineFailedParses(std::move(prevState));
        ParseRest<J + 1>(result, state, backtrack);
      }
    }
  }

  const std::tuple<PA, Ps...> ps_;
};

template<typename... Ps> inline constexpr auto first(const Ps &... ps) {
  return AlternativesParser<Ps...>{ps...};
}

#if !__GNUC__ || __clang__ || ((100 * __GNUC__ + __GNUC__MINOR__) >= 802)
// Implement operator|| with first(), unless compiling with g++,
// which can segfault at compile time and needs to continue to use
// the original implementation of operator|| as of gcc-8.1.0.
template<typename PA, typename PB>
inline constexpr auto operator||(const PA &pa, const PB &pb) {
  return first(pa, pb);
}
#else  // g++ <= 8.1.0 only: original implementation
// If a and b are parsers, then a || b returns a parser that succeeds if
// a does so, or if a fails and b succeeds.  The result types of the parsers
// must be the same type.  If a succeeds, b is not attempted.
// TODO: remove this code when no longer needed
template<typename PA, typename PB> class AlternativeParser {
public:
  using resultType = typename PA::resultType;
  static_assert(std::is_same_v<resultType, typename PB::resultType>);
  constexpr AlternativeParser(const AlternativeParser &) = default;
  constexpr AlternativeParser(const PA &pa, const PB &pb) : pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    Messages messages{std::move(state.messages())};
    ParseState backtrack{state};
    if (std::optional<resultType> ax{pa_.Parse(state)}) {
      state.messages().Restore(std::move(messages));
      return ax;
    }
    ParseState paState{std::move(state)};
    state = std::move(backtrack);
    if (std::optional<resultType> bx{pb_.Parse(state)}) {
      state.messages().Restore(std::move(messages));
      return bx;
    }
    state.CombineFailedParses(std::move(paState));
    state.messages().Restore(std::move(messages));
    std::optional<resultType> result;
    return result;
  }

private:
  const PA pa_;
  const PB pb_;
};

template<typename PA, typename PB>
inline constexpr auto operator||(const PA &pa, const PB &pb) {
  return AlternativeParser<PA, PB>{pa, pb};
}
#endif  // clang vs. g++ on operator|| implementations

// If a and b are parsers, then recovery(a,b) returns a parser that succeeds if
// a does so, or if a fails and b succeeds.  If a succeeds, b is not attempted.
// All messages from the first parse are retained.
// The two parsers must return values of the same type.
template<typename PA, typename PB> class RecoveryParser {
public:
  using resultType = typename PA::resultType;
  static_assert(std::is_same_v<resultType, typename PB::resultType>);
  constexpr RecoveryParser(const RecoveryParser &) = default;
  constexpr RecoveryParser(const PA &pa, const PB &pb) : pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    bool originallyDeferred{state.deferMessages()};
    ParseState backtrack{state};
    if (!originallyDeferred && state.messages().empty() &&
        !state.anyErrorRecovery()) {
      // Fast path.  There are no messages or recovered errors in the incoming
      // state.  Attempt to parse with messages deferred, expecting that the
      // parse will succeed silently.
      state.set_deferMessages(true);
      if (std::optional<resultType> ax{pa_.Parse(state)}) {
        if (!state.anyDeferredMessages() && !state.anyErrorRecovery()) {
          state.set_deferMessages(false);
          return ax;
        }
      }
      state = backtrack;
    }
    Messages messages{std::move(state.messages())};
    if (std::optional<resultType> ax{pa_.Parse(state)}) {
      state.messages().Restore(std::move(messages));
      return ax;
    }
    messages.Annex(std::move(state.messages()));
    bool hadDeferredMessages{state.anyDeferredMessages()};
    bool anyTokenMatched{state.anyTokenMatched()};
    state = std::move(backtrack);
    state.set_deferMessages(true);
    std::optional<resultType> bx{pb_.Parse(state)};
    state.messages() = std::move(messages);
    state.set_deferMessages(originallyDeferred);
    if (anyTokenMatched) {
      state.set_anyTokenMatched();
    }
    if (hadDeferredMessages) {
      state.set_anyDeferredMessages();
    }
    if (bx.has_value()) {
      // Error recovery situations must also produce messages.
      CHECK(state.anyDeferredMessages() || state.messages().AnyFatalError());
      state.set_anyErrorRecovery();
    }
    return bx;
  }

private:
  const PA pa_;
  const PB pb_;
};

template<typename PA, typename PB>
inline constexpr auto recovery(const PA &pa, const PB &pb) {
  return RecoveryParser<PA, PB>{pa, pb};
}

// If x is a parser, then many(x) returns a parser that always succeeds
// and whose value is a list, possibly empty, of the values returned from
// repeated application of x until it fails or does not advance the parse.
template<typename PA> class ManyParser {
  using paType = typename PA::resultType;

public:
  using resultType = std::list<paType>;
  constexpr ManyParser(const ManyParser &) = default;
  constexpr ManyParser(const PA &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    resultType result;
    auto at{state.GetLocation()};
    while (std::optional<paType> x{parser_.Parse(state)}) {
      result.emplace_back(std::move(*x));
      if (state.GetLocation() <= at) {
        break;  // no forward progress, don't loop
      }
      at = state.GetLocation();
    }
    return {std::move(result)};
  }

private:
  const BacktrackingParser<PA> parser_;
};

template<typename PA> inline constexpr auto many(const PA &parser) {
  return ManyParser<PA>{parser};
}

// If x is a parser, then some(x) returns a parser that succeeds if x does
// and whose value is a nonempty list of the values returned from repeated
// application of x until it fails or does not advance the parse.  In other
// words, some(x) is a variant of many(x) that has to succeed at least once.
template<typename PA> class SomeParser {
  using paType = typename PA::resultType;

public:
  using resultType = std::list<paType>;
  constexpr SomeParser(const SomeParser &) = default;
  constexpr SomeParser(const PA &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    auto start{state.GetLocation()};
    if (std::optional<paType> first{parser_.Parse(state)}) {
      resultType result;
      result.emplace_back(std::move(*first));
      if (state.GetLocation() > start) {
        result.splice(result.end(), *many(parser_).Parse(state));
      }
      return {std::move(result)};
    }
    return std::nullopt;
  }

private:
  const PA parser_;
};

template<typename PA> inline constexpr auto some(const PA &parser) {
  return SomeParser<PA>{parser};
}

// If x is a parser, skipMany(x) is equivalent to many(x) but with no result.
template<typename PA> class SkipManyParser {
public:
  using resultType = Success;
  constexpr SkipManyParser(const SkipManyParser &) = default;
  constexpr SkipManyParser(const PA &parser) : parser_{parser} {}
  std::optional<Success> Parse(ParseState &state) const {
    for (auto at{state.GetLocation()};
         parser_.Parse(state) && state.GetLocation() > at;
         at = state.GetLocation()) {
    }
    return {Success{}};
  }

private:
  const BacktrackingParser<PA> parser_;
};

template<typename PA> inline constexpr auto skipMany(const PA &parser) {
  return SkipManyParser<PA>{parser};
}

// If x is a parser, skipManyFast(x) is equivalent to skipMany(x).
// The parser x must always advance on success and never invalidate the
// state on failure.
template<typename PA> class SkipManyFastParser {
public:
  using resultType = Success;
  constexpr SkipManyFastParser(const SkipManyFastParser &) = default;
  constexpr SkipManyFastParser(const PA &parser) : parser_{parser} {}
  std::optional<Success> Parse(ParseState &state) const {
    while (parser_.Parse(state)) {
    }
    return {Success{}};
  }

private:
  const PA parser_;
};

template<typename PA> inline constexpr auto skipManyFast(const PA &parser) {
  return SkipManyFastParser<PA>{parser};
}

// If x is a parser returning some type A, then maybe(x) returns a
// parser that returns std::optional<A>, always succeeding.
template<typename PA> class MaybeParser {
  using paType = typename PA::resultType;

public:
  using resultType = std::optional<paType>;
  constexpr MaybeParser(const MaybeParser &) = default;
  constexpr MaybeParser(const PA &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (resultType result{parser_.Parse(state)}) {
      return {std::move(result)};
    }
    return {resultType{}};
  }

private:
  const BacktrackingParser<PA> parser_;
};

template<typename PA> inline constexpr auto maybe(const PA &parser) {
  return MaybeParser<PA>{parser};
}

// If x is a parser, then defaulted(x) returns a parser that always
// succeeds.  When x succeeds, its result is that of x; otherwise, its
// result is a default-constructed value of x's result type.
template<typename PA> class DefaultedParser {
public:
  using resultType = typename PA::resultType;
  constexpr DefaultedParser(const DefaultedParser &) = default;
  constexpr DefaultedParser(const PA &p) : parser_{p} {}
  std::optional<resultType> Parse(ParseState &state) const {
    std::optional<std::optional<resultType>> ax{maybe(parser_).Parse(state)};
    CHECK(ax.has_value());  // maybe() always succeeds
    if (ax.value().has_value()) {
      return std::move(*ax);
    }
    return {resultType{}};
  }

private:
  const BacktrackingParser<PA> parser_;
};

template<typename PA> inline constexpr auto defaulted(const PA &p) {
  return DefaultedParser<PA>(p);
}

// If a is a parser, and f is a function mapping an rvalue of a's result type
// to some other type T, then applyFunction(f, a) returns a parser that succeeds
// iff a does, and whose result value ax has been passed through the function;
// the final result is that returned by the call f(std::move(ax)).
//
// Function application is generalized to functions with more than one
// argument with applyFunction(f, a, b, ...) succeeding if all of the parsers
// a, b, &c. do so, and the result is the value of applying f to their
// results.
//
// applyLambda(f, ...) is the same concept extended to std::function<> functors.
// It is not constexpr.
//
// Member function application is supported by applyMem(f, a).  If the
// parser a succeeds and returns some value ax, the result is that returned
// by ax.f().  Additional parser arguments can be specified to supply their
// results to the member function call, so applyMem(f, a, b) succeeds if
// both a and b do so and returns the result of calling ax.f(std::move(bx)).
template<typename PA, typename T> class Apply1 {
  using paType = typename PA::resultType;
  using funcType = T (*)(paType &&);

public:
  using resultType = T;
  constexpr Apply1(const Apply1 &) = default;
  constexpr Apply1(funcType function, const PA &parser)
    : function_{function}, parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<paType> ax{parser_.Parse(state)}) {
      return {function_(std::move(*ax))};
    }
    return std::nullopt;
  }

private:
  const funcType function_;
  const PA parser_;
};

template<typename PA, typename T>
inline constexpr auto applyFunction(
    T (*f)(typename PA::resultType &&), const PA &pa) {
  return Apply1<PA, T>{f, pa};
}

template<typename PA, typename T> class Apply1Functor {
  using paType = typename PA::resultType;
  using funcType = std::function<T(paType &&)>;

public:
  using resultType = T;
  Apply1Functor(const Apply1Functor &) = default;
  Apply1Functor(const funcType &functor, const PA &parser)
    : functor_{functor}, parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<paType> ax{parser_.Parse(state)}) {
      return {functor_(std::move(*ax))};
    }
    return std::nullopt;
  }

private:
  const funcType &functor_;
  const PA parser_;
};

template<typename PA, typename T>
inline auto applyLambda(
    const std::function<T(typename PA::resultType &&)> &f, const PA &pa) {
  return Apply1Functor<PA, T>{f, pa};
}

template<typename PA> class Apply1Mem {
public:
  using resultType = typename PA::resultType;
  using funcType = void (resultType::*)();
  constexpr Apply1Mem(const Apply1Mem &) = default;
  constexpr Apply1Mem(funcType function, const PA &pa)
    : function_{function}, pa_{pa} {}
  std::optional<resultType> Parse(ParseState &state) const {
    std::optional<resultType> result{pa_.Parse(state)};
    if (result) {
      ((*result).*function_)();
    }
    return result;
  }

private:
  const funcType function_;
  const PA pa_;
};

template<typename PA>
inline constexpr auto applyMem(
    typename Apply1Mem<PA>::funcType f, const PA &pa) {
  return Apply1Mem<PA>{f, pa};
}

template<typename PA, typename PB, typename T> class Apply2 {
  using paType = typename PA::resultType;
  using pbType = typename PB::resultType;
  using funcType = T (*)(paType &&, pbType &&);

public:
  using resultType = T;
  constexpr Apply2(const Apply2 &) = default;
  constexpr Apply2(funcType function, const PA &pa, const PB &pb)
    : function_{function}, pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<paType> ax{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        return {function_(std::move(*ax), std::move(*bx))};
      }
    }
    return std::nullopt;
  }

private:
  const funcType function_;
  const PA pa_;
  const PB pb_;
};

template<typename PA, typename PB, typename T>
inline constexpr auto applyFunction(
    T (*f)(typename PA::resultType &&, typename PB::resultType &&),
    const PA &pa, const PB &pb) {
  return Apply2<PA, PB, T>{f, pa, pb};
}

template<typename PA, typename PB, typename T> class Apply2Functor {
  using paType = typename PA::resultType;
  using pbType = typename PB::resultType;
  using funcType = std::function<T(paType &&, pbType &&)>;

public:
  using resultType = T;
  Apply2Functor(const Apply2Functor &) = default;
  Apply2Functor(const funcType &function, const PA &pa, const PB &pb)
    : function_{function}, pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<paType> ax{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        return {function_(std::move(*ax), std::move(*bx))};
      }
    }
    return std::nullopt;
  }

private:
  const funcType &function_;
  const PA pa_;
  const PB pb_;
};

template<typename PA, typename PB, typename T>
inline auto applyLambda(const std::function<T(typename PA::resultType &&,
                            typename PB::resultType &&)> &f,
    const PA &pa, const PB &pb) {
  return Apply2Functor<PA, PB, T>{f, pa, pb};
}

template<typename PA, typename PB> class Apply2Mem {
  using pbType = typename PB::resultType;

public:
  using resultType = typename PA::resultType;
  using funcType = void (resultType::*)(pbType &&);
  constexpr Apply2Mem(const Apply2Mem &) = default;
  constexpr Apply2Mem(funcType function, const PA &pa, const PB &pb)
    : function_{function}, pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<resultType> result{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        ((*result).*function_)(std::move(*bx));
        return result;
      }
    }
    return std::nullopt;
  }

private:
  const funcType function_;
  const PA pa_;
  const PB pb_;
};

template<typename PA, typename PB>
inline constexpr auto applyMem(
    typename Apply2Mem<PA, PB>::funcType f, const PA &pa, const PB &pb) {
  return Apply2Mem<PA, PB>{f, pa, pb};
}

template<typename PA, typename PB, typename PC, typename T> class Apply3 {
  using paType = typename PA::resultType;
  using pbType = typename PB::resultType;
  using pcType = typename PC::resultType;
  using funcType = T (*)(paType &&, pbType &&, pcType &&);

public:
  using resultType = T;
  constexpr Apply3(const Apply3 &) = default;
  constexpr Apply3(funcType function, const PA &pa, const PB &pb, const PC &pc)
    : function_{function}, pa_{pa}, pb_{pb}, pc_{pc} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<paType> ax{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        if (std::optional<pcType> cx{pc_.Parse(state)}) {
          return {function_(std::move(*ax), std::move(*bx), std::move(*cx))};
        }
      }
    }
    return std::nullopt;
  }

private:
  const funcType function_;
  const PA pa_;
  const PB pb_;
  const PC pc_;
};

template<typename PA, typename PB, typename PC, typename T>
inline constexpr auto applyFunction(
    T (*f)(typename PA::resultType &&, typename PB::resultType &&,
        typename PC::resultType &&),
    const PA &pa, const PB &pb, const PC &pc) {
  return Apply3<PA, PB, PC, T>{f, pa, pb, pc};
}

template<typename PA, typename PB, typename PC> class Apply3Mem {
  using pbType = typename PB::resultType;
  using pcType = typename PC::resultType;

public:
  using resultType = typename PA::resultType;
  using funcType = void (resultType::*)(pbType &&, pcType &&);
  constexpr Apply3Mem(const Apply3Mem &) = default;
  constexpr Apply3Mem(
      funcType function, const PA &pa, const PB &pb, const PC &pc)
    : function_{function}, pa_{pa}, pb_{pb}, pc_{pc} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<resultType> result{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        if (std::optional<pcType> cx{pc_.Parse(state)}) {
          ((*result).*function_)(std::move(*bx), std::move(*cx));
          return result;
        }
      }
    }
    return std::nullopt;
  }

private:
  const funcType function_;
  const PA pa_;
  const PB pb_;
  const PC pc_;
};

template<typename PA, typename PB, typename PC>
inline constexpr auto applyMem(typename Apply3Mem<PA, PB, PC>::funcType f,
    const PA &pa, const PB &pb, const PC &pc) {
  return Apply3Mem<PA, PB, PC>{f, pa, pb, pc};
}

template<typename PA, typename PB, typename PC, typename PD, typename T>
class Apply4 {
  using paType = typename PA::resultType;
  using pbType = typename PB::resultType;
  using pcType = typename PC::resultType;
  using pdType = typename PD::resultType;
  using funcType = T (*)(paType &&, pbType &&, pcType &&, pdType &&);

public:
  using resultType = T;
  constexpr Apply4(const Apply4 &) = default;
  constexpr Apply4(
      funcType function, const PA &pa, const PB &pb, const PC &pc, const PD &pd)
    : function_{function}, pa_{pa}, pb_{pb}, pc_{pc}, pd_{pd} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<paType> ax{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        if (std::optional<pcType> cx{pc_.Parse(state)}) {
          if (std::optional<pdType> dx{pd_.Parse(state)}) {
            return {function_(std::move(*ax), std::move(*bx), std::move(*cx),
                std::move(*dx))};
          }
        }
      }
    }
    return std::nullopt;
  }

private:
  const funcType function_;
  const PA pa_;
  const PB pb_;
  const PC pc_;
  const PD pd_;
};

template<typename PA, typename PB, typename PC, typename PD, typename T>
inline constexpr auto applyFunction(
    T (*f)(typename PA::resultType &&, typename PB::resultType &&,
        typename PC::resultType &&, typename PD::resultType &&),
    const PA &pa, const PB &pb, const PC &pc, const PD &pd) {
  return Apply4<PA, PB, PC, PD, T>{f, pa, pb, pc, pd};
}

template<typename PA, typename PB, typename PC, typename PD> class Apply4Mem {
  using pbType = typename PB::resultType;
  using pcType = typename PC::resultType;
  using pdType = typename PD::resultType;

public:
  using resultType = typename PA::resultType;
  using funcType = void (resultType::*)(pbType &&, pcType &&, pdType &&);
  constexpr Apply4Mem(const Apply4Mem &) = default;
  constexpr Apply4Mem(
      funcType function, const PA &pa, const PB &pb, const PC &pc, const PD &pd)
    : function_{function}, pa_{pa}, pb_{pb}, pc_{pc}, pd_{pd} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (std::optional<resultType> result{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        if (std::optional<pcType> cx{pc_.Parse(state)}) {
          if (std::optional<pdType> dx{pd_.Parse(state)}) {
            ((*result).*function_)(
                std::move(*bx), std::move(*cx), std::move(*dx));
            return result;
          }
        }
      }
    }
    return std::nullopt;
  }

private:
  const funcType function_;
  const PA pa_;
  const PB pb_;
  const PC pc_;
  const PD pd_;
};

template<typename PA, typename PB, typename PC, typename PD>
inline constexpr auto applyMem(typename Apply4Mem<PA, PB, PC, PD>::funcType f,
    const PA &pa, const PB &pb, const PC &pc, const PD &pd) {
  return Apply4Mem<PA, PB, PC, PD>{f, pa, pb, pc, pd};
}

// As is done with function application via applyFunction() above, class
// instance construction can also be based upon the results of successful
// parses.  For some type T and zero or more parsers a, b, &c., the call
// construct<T>(a, b, ...) returns a parser that succeeds if all of
// its argument parsers do so in succession, and whose result is an
// instance of T constructed upon the values they returned.
template<class T> struct Construct0 {
  using resultType = T;
  constexpr Construct0() {}
  constexpr Construct0(const Construct0 &) = default;
  std::optional<T> Parse(ParseState &state) const { return {T{}}; }
};

template<class T> constexpr Construct0<T> construct() {
  return Construct0<T>{};
}

template<class T, typename PA> struct Construct01 {
  using resultType = T;
  constexpr explicit Construct01(const PA &parser) : parser_{parser} {}
  constexpr Construct01(const Construct01 &) = default;
  std::optional<T> Parse(ParseState &state) const {
    if (std::optional<Success>{parser_.Parse(state)}) {
      return {T{}};
    }
    return std::nullopt;
  }

private:
  const PA parser_;
};

template<typename T, typename PA> class Construct1 {
public:
  using resultType = T;
  constexpr explicit Construct1(const PA &parser) : parser_{parser} {}
  constexpr Construct1(const Construct1 &) = default;
  std::optional<T> Parse(ParseState &state) const {
    if (auto ax{parser_.Parse(state)}) {
      return {T(std::move(*ax))};
    }
    return std::nullopt;
  }

private:
  const PA parser_;
};

// With a single argument that is a parser with no usable value,
// construct<T>(p) invokes T's default nullary constructor T(){}.
// With a single argument that is a parser with a usable value of
// type A, construct<T>(p) invokes T's explicit constructor T(A &&).
template<class T, typename PA>
constexpr std::enable_if_t<std::is_same_v<Success, typename PA::resultType>,
    Construct01<T, PA>>
construct(const PA &parser) {
  return Construct01<T, PA>{parser};
}

template<typename T, typename PA>
constexpr std::enable_if_t<!std::is_same_v<Success, typename PA::resultType>,
    Construct1<T, PA>>
construct(const PA &parser) {
  return Construct1<T, PA>{parser};
}

template<typename T, typename PA, typename PB> class Construct2 {
public:
  using resultType = T;
  constexpr Construct2(const PA &pa, const PB &pb) : pa_{pa}, pb_{pb} {}
  constexpr Construct2(const Construct2 &) = default;
  std::optional<T> Parse(ParseState &state) const {
    if (auto ax{pa_.Parse(state)}) {
      if (auto bx{pb_.Parse(state)}) {
        return {T{std::move(*ax), std::move(*bx)}};
      }
    }
    return std::nullopt;
  }

private:
  const PA pa_;
  const PB pb_;
};

template<typename T, typename PA, typename PB>
constexpr Construct2<T, PA, PB> construct(const PA &pa, const PB &pb) {
  return Construct2<T, PA, PB>{pa, pb};
}

template<typename T, typename PA, typename PB, typename PC> class Construct3 {
public:
  using resultType = T;
  constexpr Construct3(const PA &pa, const PB &pb, const PC &pc)
    : pa_{pa}, pb_{pb}, pc_{pc} {}
  constexpr Construct3(const Construct3 &) = default;
  std::optional<resultType> Parse(ParseState &state) const {
    if (auto ax{pa_.Parse(state)}) {
      if (auto bx{pb_.Parse(state)}) {
        if (auto cx{pc_.Parse(state)}) {
          return {T{std::move(*ax), std::move(*bx), std::move(*cx)}};
        }
      }
    }
    return std::nullopt;
  }

private:
  const PA pa_;
  const PB pb_;
  const PC pc_;
};

template<typename T, typename PA, typename PB, typename PC>
constexpr Construct3<T, PA, PB, PC> construct(
    const PA &pa, const PB &pb, const PC &pc) {
  return Construct3<T, PA, PB, PC>{pa, pb, pc};
}

template<typename T, typename PA, typename PB, typename PC, typename PD>
class Construct4 {
public:
  using resultType = T;
  constexpr Construct4(const PA &pa, const PB &pb, const PC &pc, const PD &pd)
    : pa_{pa}, pb_{pb}, pc_{pc}, pd_{pd} {}
  constexpr Construct4(const Construct4 &) = default;
  std::optional<resultType> Parse(ParseState &state) const {
    if (auto ax{pa_.Parse(state)}) {
      if (auto bx{pb_.Parse(state)}) {
        if (auto cx{pc_.Parse(state)}) {
          if (auto dx{pd_.Parse(state)}) {
            return {T{std::move(*ax), std::move(*bx), std::move(*cx),
                std::move(*dx)}};
          }
        }
      }
    }
    return std::nullopt;
  }

private:
  const PA pa_;
  const PB pb_;
  const PC pc_;
  const PD pd_;
};

template<typename T, typename PA, typename PB, typename PC, typename PD>
constexpr Construct4<T, PA, PB, PC, PD> construct(
    const PA &pa, const PB &pb, const PC &pc, const PD &pd) {
  return Construct4<T, PA, PB, PC, PD>{pa, pb, pc, pd};
}

template<typename T, typename PA, typename PB, typename PC, typename PD,
    typename PE>
class Construct5 {
public:
  using resultType = T;
  constexpr Construct5(
      const PA &pa, const PB &pb, const PC &pc, const PD &pd, const PE &pe)
    : pa_{pa}, pb_{pb}, pc_{pc}, pd_{pd}, pe_{pe} {}
  constexpr Construct5(const Construct5 &) = default;
  std::optional<resultType> Parse(ParseState &state) const {
    if (auto ax{pa_.Parse(state)}) {
      if (auto bx{pb_.Parse(state)}) {
        if (auto cx{pc_.Parse(state)}) {
          if (auto dx{pd_.Parse(state)}) {
            if (auto ex{pe_.Parse(state)}) {
              return {T{std::move(*ax), std::move(*bx), std::move(*cx),
                  std::move(*dx), std::move(*ex)}};
            }
          }
        }
      }
    }
    return std::nullopt;
  }

private:
  const PA pa_;
  const PB pb_;
  const PC pc_;
  const PD pd_;
  const PE pe_;
};

template<typename T, typename PA, typename PB, typename PC, typename PD,
    typename PE>
constexpr Construct5<T, PA, PB, PC, PD, PE> construct(
    const PA &pa, const PB &pb, const PC &pc, const PD &pd, const PE &pe) {
  return Construct5<T, PA, PB, PC, PD, PE>{pa, pb, pc, pd, pe};
}

template<typename T, typename PA, typename PB, typename PC, typename PD,
    typename PE, typename PF>
class Construct6 {
public:
  using resultType = T;
  constexpr Construct6(const PA &pa, const PB &pb, const PC &pc, const PD &pd,
      const PE &pe, const PF &pf)
    : pa_{pa}, pb_{pb}, pc_{pc}, pd_{pd}, pe_{pe}, pf_{pf} {}
  constexpr Construct6(const Construct6 &) = default;
  std::optional<resultType> Parse(ParseState &state) const {
    if (auto ax{pa_.Parse(state)}) {
      if (auto bx{pb_.Parse(state)}) {
        if (auto cx{pc_.Parse(state)}) {
          if (auto dx{pd_.Parse(state)}) {
            if (auto ex{pe_.Parse(state)}) {
              if (auto fx{pf_.Parse(state)}) {
                return {T{std::move(*ax), std::move(*bx), std::move(*cx),
                    std::move(*dx), std::move(*ex), std::move(*fx)}};
              }
            }
          }
        }
      }
    }
    return std::nullopt;
  }

private:
  const PA pa_;
  const PB pb_;
  const PC pc_;
  const PD pd_;
  const PE pe_;
  const PF pf_;
};

template<typename T, typename PA, typename PB, typename PC, typename PD,
    typename PE, typename PF>
constexpr Construct6<T, PA, PB, PC, PD, PE, PF> construct(const PA &pa,
    const PB &pb, const PC &pc, const PD &pd, const PE &pe, const PF &pf) {
  return Construct6<T, PA, PB, PC, PD, PE, PF>{pa, pb, pc, pd, pe, pf};
}

// For a parser p, indirect(p) returns a parser that builds an indirect
// reference to p's return type.
template<typename PA> inline constexpr auto indirect(const PA &p) {
  return construct<common::Indirection<typename PA::resultType>>(p);
}

// If a and b are parsers, then nonemptySeparated(a, b) returns a parser
// that succeeds if a does.  If a succeeds, it then applies many(b >> a).
// The result is the list of the values returned from all of the applications
// of a.
template<typename T> std::list<T> prepend(T &&head, std::list<T> &&rest) {
  rest.push_front(std::move(head));
  return std::move(rest);
}

template<typename PA, typename PB> class NonemptySeparated {
private:
  using paType = typename PA::resultType;

public:
  using resultType = std::list<paType>;
  constexpr NonemptySeparated(const NonemptySeparated &) = default;
  constexpr NonemptySeparated(const PA &p, const PB &sep)
    : parser_{p}, separator_{sep} {}
  std::optional<resultType> Parse(ParseState &state) const {
    return applyFunction(prepend<paType>, parser_, many(separator_ >> parser_))
        .Parse(state);
  }

private:
  const PA parser_;
  const PB separator_;
};

template<typename PA, typename PB>
inline constexpr auto nonemptySeparated(const PA &p, const PB &sep) {
  return NonemptySeparated<PA, PB>{p, sep};
}

// ok is a parser that always succeeds.  It is useful when a parser
// must discard its result in order to be compatible in type with other
// parsers in an alternative, e.g. "x >> ok || y >> ok" is type-safe even
// when x and y have distinct result types.
//
// cut is a parser that always fails.  It is useful when a parser must
// have its type implicitly set; one use is the idiom "defaulted(cut >> x)",
// which is essentially what "pure(T{})" would be able to do for x's
// result type T, but without requiring that T have a default constructor
// or a non-trivial destructor.  The state is preserved.
template<bool pass> struct FixedParser {
  using resultType = Success;
  constexpr FixedParser() {}
  static constexpr std::optional<Success> Parse(ParseState &) {
    if (pass) {
      return {Success{}};
    }
    return std::nullopt;
  }
};

constexpr FixedParser<true> ok;
constexpr FixedParser<false> cut;

// A variant of recovery() above for convenience.
template<typename PA, typename PB>
inline constexpr auto localRecovery(
    MessageFixedText msg, const PA &pa, const PB &pb) {
  return recovery(withMessage(msg, pa), pb >> defaulted(cut >> pa));
}

// nextCh is a parser that succeeds if the parsing state is not
// at the end of its input, returning the next character location and
// advancing the parse when it does so.
constexpr struct NextCh {
  using resultType = const char *;
  constexpr NextCh() {}
  std::optional<const char *> Parse(ParseState &state) const {
    if (std::optional<const char *> result{state.GetNextChar()}) {
      return result;
    }
    state.Say("end of file"_err_en_US);
    return std::nullopt;
  }
} nextCh;

// If a is a parser for some nonstandard language feature LF, extension<LF>(a)
// is a parser that optionally enabled, sets a strict conformance violation
// flag, and may emit a warning message, if those are enabled.
template<LanguageFeature LF, typename PA> class NonstandardParser {
public:
  using resultType = typename PA::resultType;
  constexpr NonstandardParser(const NonstandardParser &) = default;
  constexpr NonstandardParser(const PA &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (UserState * ustate{state.userState()}) {
      if (!ustate->features().IsEnabled(LF)) {
        return std::nullopt;
      }
    }
    auto at{state.GetLocation()};
    auto result{parser_.Parse(state)};
    if (result.has_value()) {
      state.Nonstandard(
          CharBlock{at, state.GetLocation()}, LF, "nonstandard usage"_en_US);
    }
    return result;
  }

private:
  const PA parser_;
};

template<LanguageFeature LF, typename PA>
inline constexpr auto extension(const PA &parser) {
  return NonstandardParser<LF, PA>(parser);
}

// If a is a parser for some deprecated or deleted language feature LF,
// deprecated<LF>(a) is a parser that is optionally enabled, sets a strict
// conformance violation flag, and may emit a warning message, if enabled.
template<LanguageFeature LF, typename PA> class DeprecatedParser {
public:
  using resultType = typename PA::resultType;
  constexpr DeprecatedParser(const DeprecatedParser &) = default;
  constexpr DeprecatedParser(const PA &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    if (UserState * ustate{state.userState()}) {
      if (!ustate->features().IsEnabled(LF)) {
        return std::nullopt;
      }
    }
    auto at{state.GetLocation()};
    auto result{parser_.Parse(state)};
    if (result.has_value()) {
      state.Nonstandard(
          CharBlock{at, state.GetLocation()}, LF, "deprecated usage"_en_US);
    }
    return result;
  }

private:
  const PA parser_;
};

template<LanguageFeature LF, typename PA>
inline constexpr auto deprecated(const PA &parser) {
  return DeprecatedParser<LF, PA>(parser);
}

// Parsing objects with "source" members.
template<typename PA> class SourcedParser {
public:
  using resultType = typename PA::resultType;
  constexpr SourcedParser(const SourcedParser &) = default;
  constexpr SourcedParser(const PA &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState &state) const {
    const char *start{state.GetLocation()};
    auto result{parser_.Parse(state)};
    if (result.has_value()) {
      result->source = CharBlock{start, state.GetLocation()};
    }
    return result;
  }

private:
  const PA parser_;
};

template<typename PA> inline constexpr auto sourced(const PA &parser) {
  return SourcedParser<PA>{parser};
}

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_BASIC_PARSERS_H_
