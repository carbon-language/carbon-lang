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

#include "symbol.h"
#include "scope.h"
#include "../parser/idioms.h"
#include <memory>

namespace Fortran::semantics {

std::ostream &operator<<(std::ostream &os, const parser::Name &name) {
  return os << name.ToString();
}
std::ostream &operator<<(std::ostream &os, const parser::CharBlock &name) {
  return os << name.ToString();
}

void EntityDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = type;
}

void ObjectEntityDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = type;
}

void ObjectEntityDetails::set_shape(const ArraySpec &shape) {
  CHECK(shape_.empty());
  for (const auto &shapeSpec : shape) {
    shape_.push_back(shapeSpec);
  }
}

ProcEntityDetails::ProcEntityDetails(const EntityDetails &d) {
  if (auto &type = d.type()) {
    interface_.set_type(*type);
  }
}

const Symbol &UseDetails::module() const {
  // owner is a module so it must have a symbol:
  return *symbol_->owner().symbol();
}

GenericDetails::GenericDetails(const listType &specificProcs) {
  for (const auto *proc : specificProcs) {
    add_specificProc(proc);
  }
}

void GenericDetails::set_specific(Symbol &&specific) {
  CHECK(!specific_);
  specific_ = std::unique_ptr<Symbol>(new Symbol(std::move(specific)));
}

const Symbol *GenericDetails::CheckSpecific() const {
  if (const auto *specific = specific_.get()) {
    for (const auto *proc : specificProcs_) {
      if (proc == specific) {
        return nullptr;
      }
    }
    std::cerr << "specific: " << *specific << "\n";
    return specific;
  } else {
    return nullptr;
  }
}

// The name of the kind of details for this symbol.
// This is primarily for debugging.
static std::string DetailsToString(const Details &details) {
  return std::visit(
      parser::visitors{
          [](const UnknownDetails &) { return "Unknown"; },
          [](const MainProgramDetails &) { return "MainProgram"; },
          [](const ModuleDetails &) { return "Module"; },
          [](const SubprogramDetails &) { return "Subprogram"; },
          [](const SubprogramNameDetails &) { return "SubprogramName"; },
          [](const EntityDetails &) { return "Entity"; },
          [](const ObjectEntityDetails &) { return "ObjectEntity"; },
          [](const ProcEntityDetails &) { return "ProcEntity"; },
          [](const UseDetails &) { return "Use"; },
          [](const UseErrorDetails &) { return "UseError"; },
          [](const GenericDetails &) { return "Generic"; },
          [](const auto &) { return "unknown"; },
      },
      details);
}

const std::string Symbol::GetDetailsName() const {
  return DetailsToString(details_);
}

void Symbol::set_details(Details &&details) {
  CHECK(CanReplaceDetails(details));
  details_.swap(details);
}

bool Symbol::CanReplaceDetails(const Details &details) const {
  if (has<UnknownDetails>()) {
    return true;  // can always replace UnknownDetails
  } else {
    return std::visit(
        parser::visitors{
            [](const UseErrorDetails &) { return true; },
            [=](const ObjectEntityDetails &) { return has<EntityDetails>(); },
            [=](const ProcEntityDetails &) { return has<EntityDetails>(); },
            [=](const SubprogramDetails &) {
              return has<SubprogramNameDetails>();
            },
            [](const auto &) { return false; },
        },
        details);
  }
}

Symbol &Symbol::GetUltimate() {
  return const_cast<Symbol &>(static_cast<const Symbol *>(this)->GetUltimate());
}
const Symbol &Symbol::GetUltimate() const {
  if (const auto *details = detailsIf<UseDetails>()) {
    return details->symbol().GetUltimate();
  } else {
    return *this;
  }
}

bool Symbol::isSubprogram() const {
  return std::visit(
      parser::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const GenericDetails &) { return true; },
          [](const UseDetails &x) { return x.symbol().isSubprogram(); },
          [](const auto &) { return false; },
      },
      details_);
}

bool Symbol::HasExplicitInterface() const {
  return std::visit(
      parser::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const ProcEntityDetails &x) { return x.HasExplicitInterface(); },
          [](const UseDetails &x) { return x.symbol().HasExplicitInterface(); },
          [](const auto &) { return false; },
      },
      details_);
}

ObjectEntityDetails::ObjectEntityDetails(const EntityDetails &d)
  : isDummy_{d.isDummy()} {
  if (auto &type = d.type()) {
    set_type(*type);
  }
}

std::ostream &operator<<(std::ostream &os, const EntityDetails &x) {
  if (x.type()) {
    os << " type: " << *x.type();
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const ObjectEntityDetails &x) {
  if (x.type()) {
    os << " type: " << *x.type();
  }
  if (!x.shape().empty()) {
    os << " shape:";
    for (const auto &s : x.shape()) {
      os << ' ' << s;
    }
  }
  return os;
}

bool ProcEntityDetails::HasExplicitInterface() const {
  if (auto *symbol = interface_.symbol()) {
    return symbol->HasExplicitInterface();
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, const ProcEntityDetails &x) {
  if (auto *symbol = x.interface_.symbol()) {
    os << ' ' << symbol->name().ToString();
  } else if (auto *type = x.interface_.type()) {
    os << ' ' << *type;
  }
  return os;
}


static std::ostream &DumpType(std::ostream &os, const Symbol &symbol) {
  if (const auto *details = symbol.detailsIf<EntityDetails>()) {
    if (details->type()) {
      os << *details->type() << ' ';
    }
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const Details &details) {
  os << DetailsToString(details);
  std::visit(
      parser::visitors{
          [&](const UnknownDetails &x) {},
          [&](const MainProgramDetails &x) {},
          [&](const ModuleDetails &x) {},
          [&](const SubprogramDetails &x) {
            os << " (";
            int n = 0;
            for (const auto &dummy : x.dummyArgs()) {
              if (n++ > 0) os << ", ";
              DumpType(os, *dummy);
              os << dummy->name();
            }
            os << ')';
            if (x.isFunction()) {
              os << " result(";
              DumpType(os, x.result());
              os << x.result().name() << ')';
            }
            if (x.isInterface()) {
              os << " interface";
            }
          },
          [&](const SubprogramNameDetails &x) {
            os << ' ' << EnumToString(x.kind());
          },
          [&](const EntityDetails &x) { os << x; },
          [&](const ObjectEntityDetails &x) { os << x; },
          [&](const ProcEntityDetails &x) { os << x; },
          [&](const UseDetails &x) {
            os << " from " << x.symbol().name() << " in " << x.module().name();
          },
          [&](const UseErrorDetails &x) {
            os << " uses:";
            for (const auto &pair : x.occurrences()) {
              os << " from " << pair.second->name() << " at " << *pair.first;
            }
          },
          [&](const GenericDetails &x) {
            for (const auto *proc : x.specificProcs()) {
              os << ' ' << proc->name();
            }
          },
      },
      details);
  return os;
}

std::ostream &operator<<(std::ostream &o, Symbol::Flag flag) {
  return o << Symbol::EnumToString(flag);
}

std::ostream &operator<<(std::ostream &o, const Symbol::Flags &flags) {
  std::size_t n{flags.count()};
  std::size_t seen{0};
  for (std::size_t j{0}; seen < n; ++j) {
    Symbol::Flag flag{static_cast<Symbol::Flag>(j)};
    if (flags.test(flag)) {
      if (seen++ > 0) {
        o << ", ";
      }
      o << flag;
    }
  }
  return o;
}

std::ostream &operator<<(std::ostream &os, const Symbol &symbol) {
  os << symbol.name();
  if (!symbol.attrs().empty()) {
    os << ", " << symbol.attrs();
  }
  if (!symbol.flags().empty()) {
    os << " (" << symbol.flags() << ')';
  }
  os << ": " << symbol.details_;
  return os;
}

}  // namespace Fortran::semantics
