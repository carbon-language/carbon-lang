#ifndef FORTRAN_PARSER_BASIC_PARSERS_H_
#define FORTRAN_PARSER_BASIC_PARSERS_H_

// Let a "parser" be an instance of any class that supports this
// type definition and member (or static) function:
//
//   using resultType = ...;
//   std::optional<resultType> Parse(ParseState *) const;
//
// which either returns a value to signify a successful recognition or else
// returns {} to signify failure.  On failure, the state cannot be assumed
// to still be valid, in general -- see below for exceptions.
//
// This header defines the fundamental parser template classes and helper
// template functions.  See parser-combinators.txt for documentation.

#include "char-block.h"
#include "idioms.h"
#include "message.h"
#include "parse-state.h"
#include "provenance.h"
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>

namespace Fortran {
namespace parser {

// fail<A>("..."_err_en_US) returns a parser that never succeeds.  It reports an
// error message at the current position.  The result type is unused,
// but might have to be specified at the point of call for satisfy
// the type checker.  The state remains valid.
template<typename A> class FailParser {
public:
  using resultType = A;
  constexpr FailParser(const FailParser &) = default;
  constexpr explicit FailParser(MessageFixedText t) : text_{t} {}
  std::optional<A> Parse(ParseState *state) const {
    state->Say(text_);
    return {};
  }

private:
  const MessageFixedText text_;
};

class Success {};  // for when one must return something that's present

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
  std::optional<A> Parse(ParseState *) const { return {value_}; }

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
  std::optional<resultType> Parse(ParseState *state) const {
    Messages messages{std::move(state->messages())};
    ParseState backtrack{*state};
    std::optional<resultType> result{parser_.Parse(state)};
    if (result.has_value()) {
      messages.Annex(state->messages());
      state->messages() = std::move(messages);
    } else {
      *state = std::move(backtrack);
      state->messages() = std::move(messages);
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
  std::optional<Success> Parse(ParseState *state) const {
    ParseState forked{*state};
    forked.set_deferMessages(true);
    if (parser_.Parse(&forked)) {
      return {};
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
  std::optional<Success> Parse(ParseState *state) const {
    ParseState forked{*state};
    forked.set_deferMessages(true);
    if (parser_.Parse(&forked).has_value()) {
      return {Success{}};
    }
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
    state->PushContext(text_);
    std::optional<resultType> result{parser_.Parse(state)};
    state->PopContext();
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

// If a and b are parsers, then a >> b returns a parser that succeeds when
// b succeeds after a does so, but fails when either a or b does.  The
// result is taken from b.  Similarly, a / b also succeeds if both a and b
// do so, but the result is that returned by a.
template<typename PA, typename PB> class SequenceParser {
public:
  using resultType = typename PB::resultType;
  constexpr SequenceParser(const SequenceParser &) = default;
  constexpr SequenceParser(const PA &pa, const PB &pb) : pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState *state) const {
    if (pa_.Parse(state)) {
      return pb_.Parse(state);
    }
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
    if (std::optional<resultType> ax{pa_.Parse(state)}) {
      if (pb_.Parse(state)) {
        return ax;
      }
    }
    return {};
  }

private:
  const PA pa_;
  const PB pb_;
};

template<typename PA, typename PB>
inline constexpr auto operator/(const PA &pa, const PB &pb) {
  return InvertedSequenceParser<PA, PB>{pa, pb};
}

// If a and b are parsers, then a || b returns a parser that succeeds if
// a does so, or if a fails and b succeeds.  The result types of the parsers
// must be the same type.  If a succeeds, b is not attempted.
template<typename PA, typename PB> class AlternativeParser {
public:
  using resultType = typename PA::resultType;
  static_assert(std::is_same_v<resultType, typename PB::resultType>);
  constexpr AlternativeParser(const AlternativeParser &) = default;
  constexpr AlternativeParser(const PA &pa, const PB &pb) : pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState *state) const {
    Messages messages{std::move(state->messages())};
    ParseState backtrack{*state};
    if (std::optional<resultType> ax{pa_.Parse(state)}) {
      messages.Annex(state->messages());
      state->messages() = std::move(messages);
      return ax;
    }
    ParseState paState{std::move(*state)};
    *state = std::move(backtrack);
    if (std::optional<resultType> bx{pb_.Parse(state)}) {
      messages.Annex(state->messages());
      state->messages() = std::move(messages);
      return bx;
    }
    // Both alternatives failed.  Retain the state (and messages) from the
    // alternative parse that went the furthest.
    auto paEnd = paState.GetLocation();
    auto pbEnd = state->GetLocation();
    if (paEnd > pbEnd) {
      messages.Annex(paState.messages());
      *state = std::move(paState);
    } else if (paEnd < pbEnd) {
      messages.Annex(state->messages());
    } else {
      // It's a tie.
      paState.messages().Incorporate(state->messages());
      messages.Annex(paState.messages());
    }
    state->messages() = std::move(messages);
    return {};
  }

private:
  const PA pa_;
  const PB pb_;
};

template<typename PA, typename PB>
inline constexpr auto operator||(const PA &pa, const PB &pb) {
  return AlternativeParser<PA, PB>{pa, pb};
}

#if 0
// Should have been a big speed-up, but instead produced a slow-down.
// TODO: Further investigate rebinding alternatives to the right.
template<typename PA, typename PB, typename PC>
inline constexpr auto operator||(const AlternativeParser<PA,PB> &papb,
                                 const PC &pc) {
  return papb.pa_ || (papb.pb_ || pc);  // bind to the right for performance
}
#endif

// If a and b are parsers, then recovery(a,b) returns a parser that succeeds if
// a does so, or if a fails and b succeeds.  If a succeeds, b is not attempted.
// All messages from the first parse are retained.
template<typename PA, typename PB> class RecoveryParser {
public:
  using resultType = typename PA::resultType;
  static_assert(std::is_same_v<resultType, typename PB::resultType>);
  constexpr RecoveryParser(const RecoveryParser &) = default;
  constexpr RecoveryParser(const PA &pa, const PB &pb) : pa_{pa}, pb_{pb} {}
  std::optional<resultType> Parse(ParseState *state) const {
    bool originallyDeferred{state->deferMessages()};
    ParseState backtrack{*state};
    if (!originallyDeferred && state->messages().empty() &&
        !state->anyErrorRecovery()) {
      state->set_deferMessages(true);
      if (std::optional<resultType> ax{pa_.Parse(state)}) {
        if (!state->anyDeferredMessages() && !state->anyErrorRecovery()) {
          state->set_deferMessages(false);
          return ax;
        }
      }
      *state = backtrack;
    }
    Messages messages{std::move(state->messages())};
    if (std::optional<resultType> ax{pa_.Parse(state)}) {
      messages.Annex(state->messages());
      state->messages() = std::move(messages);
      return ax;
    }
    messages.Annex(state->messages());
    *state = std::move(backtrack);
    state->set_deferMessages(true);
    std::optional<resultType> bx{pb_.Parse(state)};
    state->messages() = std::move(messages);
    state->set_deferMessages(originallyDeferred);
    if (bx.has_value()) {
      state->set_anyErrorRecovery();
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
  std::optional<resultType> Parse(ParseState *state) const {
    resultType result;
    auto at = state->GetLocation();
    while (std::optional<paType> x{parser_.Parse(state)}) {
      result.emplace_back(std::move(*x));
      if (state->GetLocation() <= at) {
        break;  // no forward progress, don't loop
      }
      at = state->GetLocation();
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
  std::optional<resultType> Parse(ParseState *state) const {
    auto start = state->GetLocation();
    if (std::optional<paType> first{parser_.Parse(state)}) {
      resultType result;
      result.emplace_back(std::move(*first));
      if (state->GetLocation() > start) {
        result.splice(result.end(), *many(parser_).Parse(state));
      }
      return {std::move(result)};
    }
    return {};
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
  std::optional<Success> Parse(ParseState *state) const {
    for (auto at = state->GetLocation();
         parser_.Parse(state) && state->GetLocation() > at;
         at = state->GetLocation()) {
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
  std::optional<Success> Parse(ParseState *state) const {
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
  std::optional<resultType> Parse(ParseState *state) const {
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
  std::optional<resultType> Parse(ParseState *state) const {
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
  std::optional<resultType> Parse(ParseState *state) const {
    if (std::optional<paType> ax{parser_.Parse(state)}) {
      return {function_(std::move(*ax))};
    }
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
    if (std::optional<paType> ax{parser_.Parse(state)}) {
      return {functor_(std::move(*ax))};
    }
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
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
  std::optional<resultType> Parse(ParseState *state) const {
    if (std::optional<paType> ax{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        return {function_(std::move(*ax), std::move(*bx))};
      }
    }
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
    if (std::optional<paType> ax{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        return {function_(std::move(*ax), std::move(*bx))};
      }
    }
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
    if (std::optional<resultType> result{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        ((*result).*function_)(std::move(*bx));
        return result;
      }
    }
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
    if (std::optional<paType> ax{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        if (std::optional<pcType> cx{pc_.Parse(state)}) {
          return {function_(std::move(*ax), std::move(*bx), std::move(*cx))};
        }
      }
    }
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
    if (std::optional<resultType> result{pa_.Parse(state)}) {
      if (std::optional<pbType> bx{pb_.Parse(state)}) {
        if (std::optional<pcType> cx{pc_.Parse(state)}) {
          ((*result).*function_)(std::move(*bx), std::move(*cx));
          return result;
        }
      }
    }
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
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
    return {};
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
  std::optional<resultType> Parse(ParseState *state) const {
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
    return {};
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
// construct<T>{}(a, b, ...) returns a parser that succeeds if all of
// its argument parsers do so in succession, and whose result is an
// instance of T constructed upon the values they returned.
template<class T> struct construct {

  using resultType = T;
  constexpr construct(const construct &) = default;
  std::optional<T> Parse(ParseState *state) const { return {T{}}; }

  constexpr construct operator()() const { return *this; }

  template<typename PA> class Construct1 {
  public:
    using resultType = T;
    constexpr Construct1(const Construct1 &) = default;
    constexpr explicit Construct1(const PA &parser) : parser_{parser} {}
    std::optional<T> Parse(ParseState *state) const {
      if (auto ax = parser_.Parse(state)) {
        return {T(std::move(*ax))};
      }
      return {};
    }

  private:
    const PA parser_;
  };

  template<typename PA>
  constexpr Construct1<PA> operator()(const PA &pa) const {
    return Construct1<PA>{pa};
  }

  template<typename PA, typename PB> class Construct2 {
  public:
    using resultType = T;
    constexpr Construct2(const Construct2 &) = default;
    constexpr Construct2(const PA &pa, const PB &pb) : pa_{pa}, pb_{pb} {}
    std::optional<T> Parse(ParseState *state) const {
      if (auto ax = pa_.Parse(state)) {
        if (auto bx = pb_.Parse(state)) {
          return {T{std::move(*ax), std::move(*bx)}};
        }
      }
      return {};
    }

  private:
    const PA pa_;
    const PB pb_;
  };

  template<typename PA, typename PB>
  constexpr Construct2<PA, PB> operator()(const PA &pa, const PB &pb) const {
    return Construct2<PA, PB>{pa, pb};
  }

  template<typename PA, typename PB, typename PC> class Construct3 {
  public:
    using resultType = T;
    constexpr Construct3(const Construct3 &) = default;
    constexpr Construct3(const PA &pa, const PB &pb, const PC &pc)
      : pa_{pa}, pb_{pb}, pc_{pc} {}
    std::optional<resultType> Parse(ParseState *state) const {
      if (auto ax = pa_.Parse(state)) {
        if (auto bx = pb_.Parse(state)) {
          if (auto cx = pc_.Parse(state)) {
            return {T{std::move(*ax), std::move(*bx), std::move(*cx)}};
          }
        }
      }
      return {};
    }

  private:
    const PA pa_;
    const PB pb_;
    const PC pc_;
  };

  template<typename PA, typename PB, typename PC>
  constexpr Construct3<PA, PB, PC> operator()(
      const PA &pa, const PB &pb, const PC &pc) const {
    return Construct3<PA, PB, PC>{pa, pb, pc};
  }

  template<typename PA, typename PB, typename PC, typename PD>
  class Construct4 {
  public:
    using resultType = T;
    constexpr Construct4(const Construct4 &) = default;
    constexpr Construct4(const PA &pa, const PB &pb, const PC &pc, const PD &pd)
      : pa_{pa}, pb_{pb}, pc_{pc}, pd_{pd} {}
    std::optional<resultType> Parse(ParseState *state) const {
      if (auto ax = pa_.Parse(state)) {
        if (auto bx = pb_.Parse(state)) {
          if (auto cx = pc_.Parse(state)) {
            if (auto dx = pd_.Parse(state)) {
              return {T{std::move(*ax), std::move(*bx), std::move(*cx),
                  std::move(*dx)}};
            }
          }
        }
      }
      return {};
    }

  private:
    const PA pa_;
    const PB pb_;
    const PC pc_;
    const PD pd_;
  };

  template<typename PA, typename PB, typename PC, typename PD>
  constexpr Construct4<PA, PB, PC, PD> operator()(
      const PA &pa, const PB &pb, const PC &pc, const PD &pd) const {
    return Construct4<PA, PB, PC, PD>{pa, pb, pc, pd};
  }

  template<typename PA, typename PB, typename PC, typename PD, typename PE>
  class Construct5 {
  public:
    using resultType = T;
    constexpr Construct5(const Construct5 &) = default;
    constexpr Construct5(
        const PA &pa, const PB &pb, const PC &pc, const PD &pd, const PE &pe)
      : pa_{pa}, pb_{pb}, pc_{pc}, pd_{pd}, pe_{pe} {}
    std::optional<resultType> Parse(ParseState *state) const {
      if (auto ax = pa_.Parse(state)) {
        if (auto bx = pb_.Parse(state)) {
          if (auto cx = pc_.Parse(state)) {
            if (auto dx = pd_.Parse(state)) {
              if (auto ex = pe_.Parse(state)) {
                return {T{std::move(*ax), std::move(*bx), std::move(*cx),
                    std::move(*dx), std::move(*ex)}};
              }
            }
          }
        }
      }
      return {};
    }

  private:
    const PA pa_;
    const PB pb_;
    const PC pc_;
    const PD pd_;
    const PE pe_;
  };

  template<typename PA, typename PB, typename PC, typename PD, typename PE>
  constexpr Construct5<PA, PB, PC, PD, PE> operator()(const PA &pa,
      const PB &pb, const PC &pc, const PD &pd, const PE &pe) const {
    return Construct5<PA, PB, PC, PD, PE>{pa, pb, pc, pd, pe};
  }

  template<typename PA, typename PB, typename PC, typename PD, typename PE,
      typename PF>
  class Construct6 {
  public:
    using resultType = T;
    constexpr Construct6(const Construct6 &) = default;
    constexpr Construct6(const PA &pa, const PB &pb, const PC &pc, const PD &pd,
        const PE &pe, const PF &pf)
      : pa_{pa}, pb_{pb}, pc_{pc}, pd_{pd}, pe_{pe}, pf_{pf} {}
    std::optional<resultType> Parse(ParseState *state) const {
      if (auto ax = pa_.Parse(state)) {
        if (auto bx = pb_.Parse(state)) {
          if (auto cx = pc_.Parse(state)) {
            if (auto dx = pd_.Parse(state)) {
              if (auto ex = pe_.Parse(state)) {
                if (auto fx = pf_.Parse(state)) {
                  return {T{std::move(*ax), std::move(*bx), std::move(*cx),
                      std::move(*dx), std::move(*ex), std::move(*fx)}};
                }
              }
            }
          }
        }
      }
      return {};
    }

  private:
    const PA pa_;
    const PB pb_;
    const PC pc_;
    const PD pd_;
    const PE pe_;
    const PF pf_;
  };

  template<typename PA, typename PB, typename PC, typename PD, typename PE,
      typename PF>
  constexpr Construct6<PA, PB, PC, PD, PE, PF> operator()(const PA &pa,
      const PB &pb, const PC &pc, const PD &pd, const PE &pe,
      const PF &pf) const {
    return Construct6<PA, PB, PC, PD, PE, PF>{pa, pb, pc, pd, pe, pf};
  }
};

// If f is a function of type bool (*f)(const ParseState &), then
// StatePredicateGuardParser{f} is a parser that succeeds when f() is true
// and fails otherwise.  The state is preserved.
class StatePredicateGuardParser {
public:
  using resultType = Success;
  constexpr StatePredicateGuardParser(
      const StatePredicateGuardParser &) = default;
  constexpr explicit StatePredicateGuardParser(
      bool (*predicate)(const ParseState &))
    : predicate_{predicate} {}
  std::optional<Success> Parse(ParseState *state) const {
    if (predicate_(*state)) {
      return {Success{}};
    }
    return {};
  }

private:
  bool (*const predicate_)(const ParseState &);
};

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
  std::optional<resultType> Parse(ParseState *state) const {
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

// If f is a function of type void (*f)(ParseState *), then
// StateUpdateParser{f} is a parser that always succeeds, possibly with
// side effects on the parsing state.
class StateUpdateParser {
public:
  using resultType = Success;
  constexpr StateUpdateParser(const StateUpdateParser &) = default;
  constexpr StateUpdateParser(void (*function)(ParseState *))
    : function_{function} {}
  std::optional<Success> Parse(ParseState *state) const {
    function_(state);
    return {Success{}};
  }

private:
  void (*const function_)(ParseState *);
};

// If a is a parser with some result type A, and f is a function of A&& that
// returns another parser, then a >>= f returns a parser that succeeds
// when a does so and then f(ax) also does so; the final result is that of
// applying the parser that was returned by f(ax).
template<typename PA, typename T> class BoundMoveParser {
  using paType = typename PA::resultType;
  using funcType = T (*)(paType &&);

public:
  using resultType = T;
  constexpr BoundMoveParser(const BoundMoveParser &) = default;
  constexpr BoundMoveParser(const PA &pa, funcType f) : pa_{pa}, f_{f} {}
  std::optional<T> Parse(ParseState *state) const {
    if (std::optional<paType> ax{pa_.Parse(state)}) {
      return f_(std::move(*ax)).Parse(state);
    }
    return {};
  }

private:
  const PA pa_;
  const funcType f_;
};

template<typename PA, typename T>
inline constexpr auto operator>>=(
    const PA &pa, T (*f)(typename PA::resultType &&)) {
  return BoundMoveParser<PA, T>(pa, f);
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
  static constexpr std::optional<Success> Parse(ParseState *) {
    if (pass) {
      return {Success{}};
    }
    return {};
  }
};

constexpr FixedParser<true> ok;
constexpr FixedParser<false> cut;

// nextCh is a parser that succeeds if the parsing state is not
// at the end of its input, returning the next character location and
// advancing the parse when it does so.
constexpr struct NextCh {
  using resultType = const char *;
  constexpr NextCh() {}
  std::optional<const char *> Parse(ParseState *state) const {
    if (std::optional<const char *> result{state->GetNextChar()}) {
      return result;
    }
    state->Say("end of file"_err_en_US);
    return {};
  }
} nextCh;

// If a is a parser for nonstandard usage, extension(a) is a parser that
// is disabled in strict conformance mode and otherwise sets a violation flag
// and may emit a warning message, if those are enabled.
template<typename PA> class NonstandardParser {
public:
  using resultType = typename PA::resultType;
  constexpr NonstandardParser(const NonstandardParser &) = default;
  constexpr NonstandardParser(const PA &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState *state) const {
    if (state->strictConformance()) {
      return {};
    }
    auto at = state->GetLocation();
    auto result = parser_.Parse(state);
    if (result.has_value()) {
      state->set_anyConformanceViolation();
      if (state->warnOnNonstandardUsage()) {
        state->Say(at, "nonstandard usage"_en_US);
      }
    }
    return result;
  }

private:
  const PA parser_;
};

template<typename PA> inline constexpr auto extension(const PA &parser) {
  return NonstandardParser<PA>(parser);
}

// If a is a parser for deprecated usage, deprecated(a) is a parser that
// is disabled if strict standard compliance is enforced,and otherwise
// sets of violation flag and may emit a warning.
template<typename PA> class DeprecatedParser {
public:
  using resultType = typename PA::resultType;
  constexpr DeprecatedParser(const DeprecatedParser &) = default;
  constexpr DeprecatedParser(const PA &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState *state) const {
    if (state->strictConformance()) {
      return {};
    }
    auto at = state->GetLocation();
    auto result = parser_.Parse(state);
    if (result) {
      state->set_anyConformanceViolation();
      if (state->warnOnDeprecatedUsage()) {
        state->Say(at, "deprecated usage"_en_US);
      }
    }
    return result;
  }

private:
  const PA parser_;
};

template<typename PA> inline constexpr auto deprecated(const PA &parser) {
  return DeprecatedParser<PA>(parser);
}

// Parsing objects with "source" members.
template<typename PA> class SourcedParser {
public:
  using resultType = typename PA::resultType;
  constexpr SourcedParser(const SourcedParser &) = default;
  constexpr SourcedParser(const PA &parser) : parser_{parser} {}
  std::optional<resultType> Parse(ParseState *state) const {
    const char *start{state->GetLocation()};
    auto result = parser_.Parse(state);
    if (result.has_value()) {
      result->source = CharBlock{start, state->GetLocation()};
    }
    return result;
  }

private:
  const PA parser_;
};

template<typename PA> inline constexpr auto sourced(const PA &parser) {
  return SourcedParser<PA>{parser};
}

constexpr struct GetUserState {
  using resultType = UserState *;
  constexpr GetUserState() {}
  static std::optional<resultType> Parse(ParseState *state) {
    return {state->userState()};
  }
} getUserState;
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_BASIC_PARSERS_H_
