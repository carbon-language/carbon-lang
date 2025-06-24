// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <memory>

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check::CppInterop {

/**
 * Enhanced C++ Interop Diagnostics System - Complete Integration
 * 
 * From carbon-interop/src/diagnostics.carbon
 * Addresses GitHub Issue #5245: C++ locations in Carbon diagnostics
 */

struct CppSourceLocationInfo {
    std::string file_path;
    uint32_t line_number;
    uint32_t column_number;
    uint64_t offset_in_file;
    uint32_t include_depth;
    std::optional<CppSourceLocationInfo> macro_expansion_loc;
    std::vector<std::string> include_chain;
    
    CppSourceLocationInfo(std::string file, uint32_t line, uint32_t column, uint64_t offset);
    std::string ToString() const;
};

class EnhancedDiagnosticSystem {
public:
    enum class Kind { Error, Warning, Note, Info, PerformanceWarning };
    
private:
    std::string carbon_message_;
    std::optional<SemIR::LocId> carbon_location_;
    std::optional<CppSourceLocationInfo> cpp_location_;
    Kind diagnostic_kind_;
    std::vector<CppSourceLocationInfo> related_locations_;
    
public:
    EnhancedDiagnosticSystem(std::string message, Kind kind);
    std::string FormatDiagnostic() const;
    
    // Integration methods
    void AddCppLocation(CppSourceLocationInfo location);
    void AddRelatedLocation(CppSourceLocationInfo location);
};

} // namespace Carbon::Check::CppInterop 