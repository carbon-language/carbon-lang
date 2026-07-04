// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BENCHMARKING_SOURCE_GEN_H_
#define CARBON_TOOLCHAIN_BENCHMARKING_SOURCE_GEN_H_

#include <string>

#include "absl/random/random.h"
#include "common/map.h"
#include "common/set.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

namespace Carbon::Testing {

// Provides source code generation facilities.
//
// This class works to generate valid but random & meaningless source code in
// interesting patterns for benchmarking. It is very incomplete. A high level
// set of long-term goals:
//
// - Generate interesting patterns and structures of code that have emerged as
//   toolchain performance bottlenecks in practice in C++ codebases.
// - Generate code that includes most Carbon language features (and whatever
//   reasonable C++ analogs could be used for comparative purposes):
//   - Functions
//   - Classes with class functions, methods, and fields
//   - Interfaces
//   - Checked generics and templates
//   - Nested and unnested impls
//   - Nested classes
//   - Inline and out-of-line function and method definitions
//   - Imports and exports
//   - API files and impl files.
// - Be random but deterministic. The goal is benchmarking and so while this
//   code should strive for not producing trivially predictable patterns, it
//   should also strive to be consistent and suitable for benchmarking. Wherever
//   possible, it should permute the order and content without randomizing the
//   total count, size, or complexity.
//
// Note that the default and primary generation target is interesting Carbon
// source code. We have a best-effort to alternatively generate comparable C++
// constructs to the Carbon ones for comparative benchmarking, but there is no
// goal to cover all the interesting C++ patterns we might want to benchmark,
// and we don't aim for perfectly synthesizing C++ analogs. We can always drop
// fidelity for the C++ code path if needed for simplicity.
//
// TODO: There are numerous places where we hard code a fixed quantity. Instead,
// we should build a rich but general system to easily encode a discrete
// distribution that is sampled. We have a specialized version of this for
// identifiers that should be generalized.
class SourceGen {
 public:
  enum class Language : uint8_t {
    Carbon,
    Cpp,
  };

  struct FunctionDeclParams {
    // TODD: Arbitrary default, should switch to a distribution from data.
    int max_params = 4;
  };

  struct MethodDeclParams {
    // TODD: Arbitrary default, should switch to a distribution from data.
    int max_params = 4;
  };

  // Parameters used to generate a class in a generated file.
  //
  // Currently, this uses a fixed number of each kind of declaration, with
  // arbitrary defaults chosen. The defaults currently skew towards large
  // classes with lots of nested declarations.
  // TODO: Switch these to distributions based on data.
  //
  // TODO: Add heuristic for how many functions have return types.
  struct ClassParams {
    int public_function_decls = 4;
    FunctionDeclParams public_function_decl_params = {.max_params = 8};

    int public_method_decls = 10;
    MethodDeclParams public_method_decl_params;

    int private_function_decls = 2;
    FunctionDeclParams private_function_decl_params = {.max_params = 6};

    int private_method_decls = 8;
    MethodDeclParams private_method_decl_params = {.max_params = 6};

    int private_field_decls = 6;

    // The number of public class functions to define inline, with a body that
    // computes over its parameters.
    int inline_function_defs = 0;
    FunctionDeclParams inline_function_decl_params = {.max_params = 4};

    // The maximum number of local variables in each computation. The counts are
    // evenly distributed over `[0, max_body_locals]`.
    int max_body_locals = 4;

    // The number of public methods to define inline that return a field. Each
    // one returns a field of its own, in addition to `private_field_decls`.
    int inline_getters = 0;

    // The number of public methods to define inline that test a field with the
    // field type's predicate template. Each one tests a field of its own.
    int inline_predicates = 0;

    // The number of public methods to define inline that pass their parameters
    // on to a private method declaration with the same signature.
    int inline_forwarders = 0;
    MethodDeclParams inline_forwarder_params = {.max_params = 4};
  };

  // Parameters used to select type _uses_, as opposed to definitions.
  //
  // These govern what distribution of types are used for function parameters,
  // returns, and fields.
  //
  // Mainly these provide a coarse histogram of weights to shape the
  // distribution of different type options, and try to fit that as closely as
  // possible.
  //
  // The default weights in the histogram were arbitrarily selected based on
  // intuition about importance for benchmarking and not based on any
  // measurement. We arrange for them to sum to 100 so that the weights can be
  // view as %s of the type uses.
  //
  // The specific builtin type options used in the weights were also selected
  // arbitrarily.
  //
  // TODO: Improve the set of builtin types and the weighting if real world code
  // ends up sharply different.
  //
  // TODO: Add a heuristic to make some % of type references via pointers (or
  // other compound types).
  struct TypeUseParams {
    // The weights in the histogram start with a sequence fixed types described
    // with a Carbon and C++ string, and their associated weight.
    //
    // Each type also has a value expression, which a body uses to produce a
    // value of the type, such as a return value. Types without one, like
    // pointers, are only used where no value is produced.
    //
    // Each type also has consumer templates, which a body uses to read a
    // parameter of the type so that no parameter is unused. A template is an
    // `i32` expression (`int` in C++) with a `{0}` placeholder for the
    // parameter name. Every type needs at least one.
    //
    // A type can also have a predicate template, a `bool` expression with a
    // `{0}` placeholder for a value of the type. Only types with a predicate
    // template and a value expression are used for the fields that predicates
    // test.
    struct FixedTypeWeight {
      llvm::StringRef carbon_spelling;
      llvm::StringRef cpp_spelling;
      int weight;
      llvm::StringRef carbon_value;
      llvm::StringRef cpp_value;
      llvm::SmallVector<llvm::StringRef> carbon_consumers;
      llvm::SmallVector<llvm::StringRef> cpp_consumers;
      llvm::StringRef carbon_predicate;
      llvm::StringRef cpp_predicate;
    };

    llvm::SmallVector<FixedTypeWeight> fixed_type_weights = {
        // Combined weight of 65 for a core set of builtin types.
        {.carbon_spelling = "bool",
         .cpp_spelling = "bool",
         .weight = 25,
         .carbon_value = "true",
         .cpp_value = "true",
         .carbon_consumers = {"(if {0} then 1 else 0)",
                              "(if {0} then 2 else 3)"},
         .cpp_consumers = {"({0} ? 1 : 0)", "({0} ? 2 : 3)"},
         .carbon_predicate = "not {0}",
         .cpp_predicate = "!{0}"},
        {.carbon_spelling = "i32",
         .cpp_spelling = "int",
         .weight = 20,
         .carbon_value = "0",
         .cpp_value = "0",
         .carbon_consumers = {"{0} + 1", "{0} * 2", "({0} % 7) + 1"},
         .cpp_consumers = {"{0} + 1", "{0} * 2", "({0} % 7) + 1"},
         .carbon_predicate = "{0} == 0",
         .cpp_predicate = "{0} == 0"},
        {.carbon_spelling = "i64",
         .cpp_spelling = "std::int64_t",
         .weight = 10,
         .carbon_value = "0",
         .cpp_value = "0",
         .carbon_consumers = {"({0} as i32) - 2", "(({0} + 1) as i32)"},
         .cpp_consumers = {"static_cast<int>({0}) - 2",
                           "static_cast<int>({0} + 1)"},
         .carbon_predicate = "{0} > 0",
         .cpp_predicate = "{0} > 0"},
        {.carbon_spelling = "i32*",
         .cpp_spelling = "int*",
         .weight = 5,
         .carbon_consumers = {"*{0} + 1", "*{0} * 2"},
         .cpp_consumers = {"*{0} + 1", "*{0} * 2"}},
        {.carbon_spelling = "i64*",
         .cpp_spelling = "std::int64_t*",
         .weight = 5,
         .carbon_consumers = {"(*{0} as i32) - 2", "((*{0} * 2) as i32)"},
         .cpp_consumers = {"static_cast<int>(*{0}) - 2",
                           "static_cast<int>(*{0} * 2)"}},

        // A weight of 5 distributed across tuple structures
        {.carbon_spelling = "(bool, i64)",
         .cpp_spelling = "std::pair<bool, std::int64_t>",
         .weight = 2,
         .carbon_value = "(true, 0)",
         .cpp_value = "{true, 0}",
         .carbon_consumers = {"(if {0}.0 then 1 else 0)", "({0}.1 as i32) + 1"},
         .cpp_consumers = {"({0}.first ? 1 : 0)",
                           "static_cast<int>({0}.second) + 1"},
         .carbon_predicate = "{0}.1 > 0",
         .cpp_predicate = "{0}.second > 0"},
        {.carbon_spelling = "(i32, i64*)",
         .cpp_spelling = "std::pair<int, std::int64_t*>",
         .weight = 3,
         .carbon_consumers = {"{0}.0 + 1", "{0}.0 * 2", "(*{0}.1 as i32) - 1"},
         .cpp_consumers = {"{0}.first + 1", "{0}.first * 2",
                           "static_cast<int>(*{0}.second) - 1"}},
    };

    // Consumer templates for class types, in both languages. These call the
    // generated `Checksum` method, which is the same in every class.
    llvm::SmallVector<llvm::StringRef> class_consumers = {
        "{0}.Checksum()", "{0}.Checksum() + 1", "{0}.Checksum() * 2"};

    // The weight for using types declared in the file. These will be randomly
    // shuffled references, and when there are more type references than
    // declared, include repeated references.
    int declared_types_weight = 30;
  };

  // Parameters used to generate a file with dense declarations.
  struct DenseDeclParams {
    // TODO: Add more parameters to control generating top-level constructs
    // other than class definitions.

    // Parameters used when generating class definitions.
    ClassParams class_params = {};

    // Parameters used to guide the selection of types for use in declarations.
    TypeUseParams type_use_params = {};
  };

  // Access a global instance of this type to generate Carbon code for
  // benchmarks, tests, or other places where sharing a common instance is
  // useful. Note that there is nothing thread safe about this instance or type.
  static auto Global() -> SourceGen&;

  // Construct a source generator for the provided language, by default Carbon.
  explicit SourceGen(Language language = Language::Carbon);

  // Generate an API file with dense classes containing function forward
  // declarations.
  //
  // Accepts a number of `target_lines` for the resulting source code. This is a
  // rough approximation used to scale all the other constructs up and down
  // accordingly. For C++ source generation, we work to generate the same number
  // of constructs as Carbon would for the given line count over keeping the
  // actual line count close to the target.
  //
  // TODO: Currently, the formatting and line breaks of generating code are
  // extremely rough still, and those are a large factor in adherence to
  // `target_lines`. Long term, the goal is to get as close as we can to any
  // automatically formatted code while still keeping the stability of
  // benchmarking.
  auto GenApiFileDenseDecls(int target_lines, const DenseDeclParams& params)
      -> std::string;

  // Get some number of randomly shuffled identifiers.
  //
  // The identifiers start with a character [A-Za-z], other characters may also
  // include [0-9_]. Both Carbon and C++ keywords are excluded along with any
  // other non-identifier syntaxes that overlap to ensure all of these can be
  // used as identifiers.
  //
  // The order will be different for each call to this function, but the
  // specific identifiers may remain the same in order to reduce the cost of
  // repeated calls. However, the sum of the identifier sizes returned is
  // guaranteed to be the same for every call with the same number of
  // identifiers so that benchmarking all of these identifiers has predictable
  // and stable cost.
  //
  // Optionally, callers can request a minimum and maximum length. By default,
  // the length distribution used across the identifiers will mirror the
  // observed distribution of identifiers in C++ source code and our expectation
  // of them in Carbon source code. The maximum length in this default
  // distribution cannot be more than 64.
  //
  // Callers can request a uniform distribution across [min_length, max_length],
  // and when it is requested there is no limit on `max_length`.
  auto GetShuffledIdentifiers(int number, int min_length = 1,
                              int max_length = 64, bool uniform = false)
      -> llvm::SmallVector<llvm::StringRef>;

  // Same as `GetShuffledIdentifiers`, but ensures there are no collisions.
  auto GetShuffledUniqueIdentifiers(int number, int min_length = 4,
                                    int max_length = 64, bool uniform = false)
      -> llvm::SmallVector<llvm::StringRef>;

  // Returns a collection of un-shuffled identifiers, otherwise the same as
  // `GetShuffledIdentifiers`.
  //
  // Usually, benchmarks should use the shuffled version. However, this is
  // useful when deterministic access to the identifiers is needed to avoid
  // introducing noise, or if there is already a post-processing step to shuffle
  // things, since shuffling is very expensive in debug builds.
  auto GetIdentifiers(int number, int min_length = 1, int max_length = 64,
                      bool uniform = false)
      -> llvm::SmallVector<llvm::StringRef>;

  // Returns a collection of un-shuffled unique identifiers, otherwise the same
  // as `GetShuffledUniqueIdentifiers`.
  //
  // Usually, benchmarks should use the shuffled version. However, this is
  // useful when deterministic access to the identifiers is needed to avoid
  // introducing noise, or if there is already a post-processing step to shuffle
  // things, since shuffling is very expensive in debug builds.
  auto GetUniqueIdentifiers(int number, int min_length = 1, int max_length = 64,
                            bool uniform = false)
      -> llvm::SmallVector<llvm::StringRef>;

  // Returns a shared collection of random identifiers of a specific length.
  //
  // For a single, exact length, we have an even cheaper routine to return
  // access to a shared collection of identifiers. The order of these is a
  // single fixed random order for a given execution. The returned array
  // reference is only valid until the next call any method on this generator.
  auto GetSingleLengthIdentifiers(int length, int number)
      -> llvm::ArrayRef<llvm::StringRef>;

 private:
  class ClassGenState;
  friend ClassGenState;

  class UniqueIdentifierPopper;
  friend UniqueIdentifierPopper;

  using AppendFn = auto(int length, int number,
                        llvm::SmallVectorImpl<llvm::StringRef>& dest) -> void;

  auto IsCpp() -> bool { return language_ == Language::Cpp; }

  auto GenerateRandomIdentifier(llvm::MutableArrayRef<char> dest_storage)
      -> void;
  auto AppendUniqueIdentifiers(int length, int number,
                               llvm::SmallVectorImpl<llvm::StringRef>& dest)
      -> void;
  auto GetIdentifiersImpl(int number, int min_length, int max_length,
                          bool uniform, llvm::function_ref<AppendFn> append)
      -> llvm::SmallVector<llvm::StringRef>;

  auto GetShuffledInts(int number, int min, int max) -> llvm::SmallVector<int>;

  // A parameter or a field: its name, and the spelling of its type.
  struct TypedName {
    llvm::StringRef name;
    llvm::StringRef type;
  };

  auto EmitParams(bool is_method, llvm::ArrayRef<TypedName> params,
                  llvm::StringRef indent, llvm::raw_ostream& os) -> void;
  auto EmitFunctionDecl(llvm::StringRef name, bool is_private, bool is_method,
                        llvm::ArrayRef<TypedName> params,
                        llvm::StringRef return_type, llvm::StringRef indent,
                        llvm::raw_ostream& os) -> void;
  auto GenerateFunctionDecl(ClassGenState& state, llvm::StringRef name,
                            bool is_private, bool is_method, int param_count,
                            llvm::StringRef indent, llvm::raw_ostream& os)
      -> void;
  auto GenerateInlineFunctionDef(ClassGenState& state, llvm::StringRef name,
                                 int param_count, int local_count,
                                 llvm::StringRef indent, llvm::raw_ostream& os)
      -> void;
  auto GenerateGetter(llvm::StringRef name, TypedName field,
                      llvm::raw_ostream& os) -> void;
  auto GeneratePredicate(llvm::StringRef name, TypedName field,
                         llvm::StringRef predicate, llvm::raw_ostream& os)
      -> void;
  auto GenerateForwarder(llvm::StringRef name, llvm::ArrayRef<TypedName> params,
                         llvm::StringRef return_type, llvm::raw_ostream& os)
      -> void;
  auto GenerateMakeFunction(ClassGenState& state, llvm::StringRef class_name,
                            llvm::ArrayRef<TypedName> fields,
                            llvm::raw_ostream& os) -> void;
  auto GenerateChecksumFunction(llvm::raw_ostream& os) -> void;
  auto GenerateClassDef(const ClassParams& params, ClassGenState& state,
                        llvm::raw_ostream& os) -> void;

  absl::BitGen rng_;
  llvm::BumpPtrAllocator storage_;

  Map<int, llvm::SmallVector<llvm::StringRef>> identifiers_by_length_;
  Map<int, std::pair<int, Set<llvm::StringRef>>> unique_identifiers_by_length_;

  Language language_;
};

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_BENCHMARKING_SOURCE_GEN_H_
