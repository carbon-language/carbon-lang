// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "resolve-names-utils.h"
#include "semantics.h"
#include "symbol.h"
#include "type.h"
#include "../common/idioms.h"
#include "../parser/char-block.h"
#include "../parser/features.h"
#include "../parser/parse-tree.h"
#include <ostream>
#include <variant>

namespace Fortran::semantics {

using IntrinsicOperator = parser::DefinedOperator::IntrinsicOperator;

static GenericKind MapIntrinsicOperator(IntrinsicOperator);

Symbol *Resolve(const parser::Name &name, Symbol *symbol) {
  if (symbol && !name.symbol) {
    name.symbol = symbol;
  }
  return symbol;
}
Symbol &Resolve(const parser::Name &name, Symbol &symbol) {
  return *Resolve(name, &symbol);
}

parser::MessageFixedText WithIsFatal(
    const parser::MessageFixedText &msg, bool isFatal) {
  return parser::MessageFixedText{
      msg.text().begin(), msg.text().size(), isFatal};
}

bool IsDefinedOperator(const SourceName &name) {
  const char *begin{name.begin()};
  const char *end{name.end()};
  return begin != end && begin[0] == '.' && end[-1] == '.';
}

bool IsInstrinsicOperator(
    const SemanticsContext &context, const SourceName &name) {
  std::string str{name.ToString()};
  std::set<std::string> intrinsics{".and.", ".eq.", ".eqv.", ".ge.", ".gt.",
      ".le.", ".lt.", ".ne.", ".neqv.", ".not.", ".or."};
  if (intrinsics.count(str) > 0) {
    return true;
  }
  if (context.IsEnabled(parser::LanguageFeature::XOROperator) &&
      str == ".xor.") {
    return true;
  }
  if (context.IsEnabled(parser::LanguageFeature::LogicalAbbreviations) &&
      (str == ".n." || str == ".a" || str == ".o." || str == ".x.")) {
    return true;
  }
  return false;
}

bool IsLogicalConstant(
    const SemanticsContext &context, const SourceName &name) {
  std::string str{name.ToString()};
  return str == ".true." || str == ".false." ||
      (context.IsEnabled(parser::LanguageFeature::LogicalAbbreviations) &&
          (str == ".t" || str == ".f."));
}

void GenericSpecInfo::Resolve(Symbol *symbol) {
  if (symbol) {
    if (auto *details{symbol->detailsIf<GenericDetails>()}) {
      details->set_kind(kind_);
    } else if (auto *details{symbol->detailsIf<GenericBindingDetails>()}) {
      details->set_kind(kind_);
    }
    if (parseName_) {
      semantics::Resolve(*parseName_, symbol);
    }
  }
}

void GenericSpecInfo::Analyze(const parser::DefinedOpName &name) {
  kind_ = GenericKind::DefinedOp;
  parseName_ = &name.v;
  symbolName_ = &name.v.source;
}

void GenericSpecInfo::Analyze(const parser::GenericSpec &x) {
  symbolName_ = &x.source;
  kind_ = std::visit(
      common::visitors{
          [&](const parser::Name &y) {
            parseName_ = &y;
            symbolName_ = &y.source;
            return GenericKind::Name;
          },
          [&](const parser::DefinedOperator &y) {
            return std::visit(
                common::visitors{
                    [&](const parser::DefinedOpName &z) {
                      Analyze(z);
                      return GenericKind::DefinedOp;
                    },
                    [&](const IntrinsicOperator &z) {
                      return MapIntrinsicOperator(z);
                    },
                },
                y.u);
          },
          [&](const parser::GenericSpec::Assignment &y) {
            return GenericKind::Assignment;
          },
          [&](const parser::GenericSpec::ReadFormatted &y) {
            return GenericKind::ReadFormatted;
          },
          [&](const parser::GenericSpec::ReadUnformatted &y) {
            return GenericKind::ReadUnformatted;
          },
          [&](const parser::GenericSpec::WriteFormatted &y) {
            return GenericKind::WriteFormatted;
          },
          [&](const parser::GenericSpec::WriteUnformatted &y) {
            return GenericKind::WriteUnformatted;
          },
      },
      x.u);
}

// parser::DefinedOperator::IntrinsicOperator -> GenericKind
static GenericKind MapIntrinsicOperator(IntrinsicOperator op) {
  switch (op) {
  case IntrinsicOperator::Power: return GenericKind::OpPower;
  case IntrinsicOperator::Multiply: return GenericKind::OpMultiply;
  case IntrinsicOperator::Divide: return GenericKind::OpDivide;
  case IntrinsicOperator::Add: return GenericKind::OpAdd;
  case IntrinsicOperator::Subtract: return GenericKind::OpSubtract;
  case IntrinsicOperator::Concat: return GenericKind::OpConcat;
  case IntrinsicOperator::LT: return GenericKind::OpLT;
  case IntrinsicOperator::LE: return GenericKind::OpLE;
  case IntrinsicOperator::EQ: return GenericKind::OpEQ;
  case IntrinsicOperator::NE: return GenericKind::OpNE;
  case IntrinsicOperator::GE: return GenericKind::OpGE;
  case IntrinsicOperator::GT: return GenericKind::OpGT;
  case IntrinsicOperator::NOT: return GenericKind::OpNOT;
  case IntrinsicOperator::AND: return GenericKind::OpAND;
  case IntrinsicOperator::OR: return GenericKind::OpOR;
  case IntrinsicOperator::XOR: return GenericKind::OpXOR;
  case IntrinsicOperator::EQV: return GenericKind::OpEQV;
  case IntrinsicOperator::NEQV: return GenericKind::OpNEQV;
  default: CRASH_NO_CASE;
  }
}

}
