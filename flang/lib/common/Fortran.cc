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

#include "Fortran.h"

namespace Fortran::common {

const char *AsFortran(NumericOperator opr) {
  switch (opr) {
    SWITCH_COVERS_ALL_CASES
  case NumericOperator::Power: return "**";
  case NumericOperator::Multiply: return "*";
  case NumericOperator::Divide: return "/";
  case NumericOperator::Add: return "+";
  case NumericOperator::Subtract: return "-";
  }
}

const char *AsFortran(LogicalOperator opr) {
  switch (opr) {
    SWITCH_COVERS_ALL_CASES
  case LogicalOperator::And: return ".and.";
  case LogicalOperator::Or: return ".or.";
  case LogicalOperator::Eqv: return ".eqv.";
  case LogicalOperator::Neqv: return ".neqv.";
  case LogicalOperator::Not: return ".not.";
  }
}

const char *AsFortran(RelationalOperator opr) {
  switch (opr) {
    SWITCH_COVERS_ALL_CASES
  case RelationalOperator::LT: return "<";
  case RelationalOperator::LE: return "<=";
  case RelationalOperator::EQ: return "==";
  case RelationalOperator::NE: return "/=";
  case RelationalOperator::GE: return ">=";
  case RelationalOperator::GT: return ">";
  }
}

}
